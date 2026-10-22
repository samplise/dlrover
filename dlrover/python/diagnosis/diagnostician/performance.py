# Copyright 2026 The DLRover Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import re
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from statistics import median
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple

from dlrover.python.common import env_utils
from dlrover.python.diagnosis.common.constants import (
    DiagnosisErrorConstant,
)
from dlrover.python.diagnosis.common.diagnosis_action import (
    DiagnosisAction,
    NoAction,
    ReportDiagnosisDataAction,
)
from dlrover.python.diagnosis.common.diagnosis_data import (
    PerformanceDiagnosisData,
    TrainingLog,
)
from dlrover.python.diagnosis.common.diagnostician import (
    DiagnosisObservation,
    Diagnostician,
)
from dlrover.python.diagnosis.datacollector.training_log_collector import (
    TrainingLogCollector,
)


THROUGHPUT_METRIC = "throughput_per_gpu_tflops"
ELAPSED_TIME_METRIC = "elapsed_time_per_iteration_ms"

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_TIMESTAMP_PATTERN = re.compile(
    r"\[(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?)\]"
)
_ITERATION_PATTERN = re.compile(
    r"\biteration\s*:?\s*(\d+)", re.IGNORECASE
)
_THROUGHPUT_PATTERN = re.compile(
    rf"throughput\s+per\s+GPU\s*\(TFLOP/s/GPU\)\s*:\s*({_NUMBER})",
    re.IGNORECASE,
)
_ELAPSED_TIME_PATTERN = re.compile(
    rf"elapsed\s+time\s+per\s+iteration\s*\(ms\)\s*:\s*({_NUMBER})",
    re.IGNORECASE,
)


def _normalise_metric_name(name: str) -> str:
    name = name.strip().lower().replace("/", "_per_")
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def _parse_log_timestamp(log_line: str) -> int:
    matches = _TIMESTAMP_PATTERN.findall(log_line)
    if not matches:
        return int(time.time())
    try:
        return int(
            datetime.fromisoformat(matches[-1].replace(" ", "T")).timestamp()
        )
    except ValueError:
        return int(time.time())


@dataclass(frozen=True)
class PerformanceRecord:
    """One performance observation.

    ``iteration`` is optional because some performance signals are emitted
    outside an iteration.  ``metrics`` keeps the record extensible while the
    first implementation focuses on ``THROUGHPUT_METRIC``.
    """

    timestamp: int
    iteration: Optional[int] = None
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def throughput(self) -> Optional[float]:
        return self.metrics.get(THROUGHPUT_METRIC)

    def to_dict(self) -> Dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "iteration": self.iteration,
            "metrics": dict(self.metrics),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, value: Dict[str, object]) -> "PerformanceRecord":
        iteration = value.get("iteration")
        return cls(
            timestamp=int(value.get("timestamp", 0)),
            iteration=int(iteration) if iteration is not None else None,
            metrics={
                str(name): float(metric_value)
                for name, metric_value in value.get("metrics", {}).items()
            },
        )

    @classmethod
    def from_json(cls, value: str) -> "PerformanceRecord":
        return cls.from_dict(json.loads(value))

    @property
    def signature(self) -> Tuple[object, ...]:
        """Stable identity used to deduplicate repeated log tails."""
        return (
            self.timestamp,
            self.iteration,
            tuple(sorted(self.metrics.items())),
        )


class PerformanceHistory:
    """Bounded recent history with duplicate and age eviction."""

    def __init__(self, max_records: int = 1024, retention_seconds: int = 1800):
        self._max_records = max(1, max_records)
        self._retention_seconds = max(0, retention_seconds)
        self._records: Deque[PerformanceRecord] = deque(
            maxlen=self._max_records
        )
        self._record_keys: Set[Tuple[object, ...]] = set()

    @property
    def records(self) -> List[PerformanceRecord]:
        return list(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def add(self, record: PerformanceRecord) -> bool:
        """Add ``record`` and return ``False`` when it is a duplicate."""
        if record.signature in self._record_keys:
            return False

        while self._records and (
            record.timestamp - self._records[0].timestamp
            > self._retention_seconds
        ):
            old_record = self._records.popleft()
            self._record_keys.discard(old_record.signature)

        if len(self._records) == self._max_records:
            old_record = self._records.popleft()
            self._record_keys.discard(old_record.signature)

        self._records.append(record)
        self._record_keys.add(record.signature)
        return True

    def clear(self):
        self._records.clear()
        self._record_keys.clear()

    @property
    def latest(self) -> Optional[PerformanceRecord]:
        return self._records[-1] if self._records else None

    @property
    def throughput_baseline(self) -> Optional[float]:
        values = [record.throughput for record in self._records]
        values = [value for value in values if value is not None]
        return median(values) if values else None


@dataclass
class PerformanceIssue:
    """A performance problem supported by a sustained set of observations."""

    issue_type: str = DiagnosisErrorConstant.TRAINING_SPEED_DEGRADATION
    metric_name: str = THROUGHPUT_METRIC
    metric_value: Optional[float] = None
    baseline_value: Optional[float] = None
    threshold_value: Optional[float] = None
    degradation_ratio: Optional[float] = None
    sample_count: int = 0
    first_timestamp: int = 0
    last_timestamp: int = 0
    first_iteration: Optional[int] = None
    last_iteration: Optional[int] = None
    evidence: List[Dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "issue_type": self.issue_type,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "baseline_value": self.baseline_value,
            "threshold_value": self.threshold_value,
            "degradation_ratio": self.degradation_ratio,
            "sample_count": self.sample_count,
            "first_timestamp": self.first_timestamp,
            "last_timestamp": self.last_timestamp,
            "first_iteration": self.first_iteration,
            "last_iteration": self.last_iteration,
            "evidence": self.evidence,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, value: Dict[str, object]) -> "PerformanceIssue":
        return cls(**value)

    @classmethod
    def from_json(cls, value: str) -> "PerformanceIssue":
        return cls.from_dict(json.loads(value))


@dataclass
class PerformanceDiagnosticianConfig:
    """Configuration shared by the worker-side performance diagnostician."""

    log_tail_lines: int = 5000
    history_size: int = 1024
    history_retention_seconds: int = 1800
    baseline_window_size: int = 20
    minimum_baseline_records: int = 5
    throughput_threshold: Optional[float] = None
    degradation_ratio: float = 0.7
    consecutive_degraded_records: int = 3
    minimum_degradation_duration_seconds: int = 60
    recovery_records: int = 3
    maximum_evidence_records: int = 5


class PerformanceDiagnostician(Diagnostician):
    """Diagnose sustained training-speed degradation on one worker.

    The diagnostician reads the existing training-log collector, so it does
    not create a second log-reading path.  It evaluates only newly observed
    records and keeps a rolling set of healthy throughput values for the
    baseline.  A report is emitted only after the configured consecutive
    sample and duration conditions are satisfied.
    """

    def __init__(self, config: Optional[PerformanceDiagnosticianConfig] = None):
        super().__init__()
        self._config = config or PerformanceDiagnosticianConfig()
        self._history = PerformanceHistory(
            self._config.history_size,
            self._config.history_retention_seconds,
        )
        self._healthy_records: Deque[PerformanceRecord] = deque(
            maxlen=max(1, self._config.baseline_window_size)
        )
        self._degraded_records: List[PerformanceRecord] = []
        self._first_degraded_timestamp: Optional[int] = None
        self._issue_reported = False
        self._recovery_count = 0

    @property
    def config(self) -> PerformanceDiagnosticianConfig:
        return self._config

    @property
    def history(self) -> PerformanceHistory:
        return self._history

    @property
    def healthy_throughputs(self) -> List[float]:
        return [
            record.throughput
            for record in self._healthy_records
            if record.throughput is not None
        ]

    @staticmethod
    def parse_log_line(log_line: str) -> Optional[PerformanceRecord]:
        """Parse one training log line into a generic performance record."""
        if not isinstance(log_line, str):
            return None

        metrics: Dict[str, float] = {}
        throughput_match = _THROUGHPUT_PATTERN.search(log_line)
        if throughput_match:
            metrics[THROUGHPUT_METRIC] = float(throughput_match.group(1))

        elapsed_match = _ELAPSED_TIME_PATTERN.search(log_line)
        if elapsed_match:
            metrics[ELAPSED_TIME_METRIC] = float(elapsed_match.group(1))

        # Additional numeric fields after the first pipe are retained for
        # future performance checks.  The first segment contains the log
        # prefix and iteration number, so it is deliberately excluded.
        for segment in log_line.split("|")[1:]:
            match = re.search(
                rf"([^:|]+):\s*({_NUMBER})\s*$", segment.strip()
            )
            if not match:
                continue
            metric_name = _normalise_metric_name(match.group(1))
            if metric_name:
                metrics.setdefault(metric_name, float(match.group(2)))

        if not metrics:
            return None

        iteration_match = _ITERATION_PATTERN.search(log_line)
        iteration = int(iteration_match.group(1)) if iteration_match else None
        return PerformanceRecord(
            timestamp=_parse_log_timestamp(log_line),
            iteration=iteration,
            metrics=metrics,
        )

    def _collect_training_log(self, **kwargs) -> Optional[TrainingLog]:
        training_log = kwargs.get("training_log")
        if isinstance(training_log, TrainingLog):
            return training_log

        logs = kwargs.get("logs")
        if logs is not None:
            if isinstance(logs, (str, bytes)) or not isinstance(logs, Iterable):
                return None
            return TrainingLog(logs=[str(log) for log in logs])

        log_file = kwargs.get("log_file")
        if not log_file:
            return None
        if not isinstance(log_file, str):
            return None
        return TrainingLogCollector(
            log_file=log_file,
            n_line=self._config.log_tail_lines,
        ).collect_data()

    def _threshold(self, baseline: Optional[float]) -> Optional[float]:
        if self._config.throughput_threshold is not None:
            return self._config.throughput_threshold
        if baseline is None:
            return None
        return baseline * self._config.degradation_ratio

    def _add_healthy_record(self, record: PerformanceRecord):
        while self._healthy_records and (
            record.timestamp - self._healthy_records[0].timestamp
            > self._config.history_retention_seconds
        ):
            self._healthy_records.popleft()
        self._healthy_records.append(record)

    def _healthy_baseline(self) -> Optional[float]:
        values = self.healthy_throughputs
        return median(values) if values else None

    def _make_issue(self, record: PerformanceRecord) -> PerformanceIssue:
        baseline = self._healthy_baseline()
        threshold = self._threshold(baseline)
        ratio = (
            record.throughput / baseline
            if record.throughput is not None and baseline not in (None, 0)
            else None
        )
        evidence = [
            item.to_dict()
            for item in self._degraded_records[
                -self._config.maximum_evidence_records :
            ]
        ]
        return PerformanceIssue(
            metric_value=record.throughput,
            baseline_value=baseline,
            threshold_value=threshold,
            degradation_ratio=ratio,
            sample_count=len(self._degraded_records),
            first_timestamp=self._first_degraded_timestamp or record.timestamp,
            last_timestamp=record.timestamp,
            first_iteration=(
                self._degraded_records[0].iteration
                if self._degraded_records
                else record.iteration
            ),
            last_iteration=record.iteration,
            evidence=evidence,
        )

    def _evaluate_record(
        self, record: PerformanceRecord
    ) -> Optional[PerformanceIssue]:
        throughput = record.throughput
        if throughput is None:
            return None

        baseline = (
            self._healthy_baseline()
            if len(self._healthy_records)
            >= max(1, self._config.minimum_baseline_records)
            else None
        )
        threshold = self._threshold(baseline)
        is_degraded = threshold is not None and throughput < threshold

        if is_degraded:
            self._recovery_count = 0
            if self._first_degraded_timestamp is None:
                self._first_degraded_timestamp = record.timestamp
                self._degraded_records = []
            self._degraded_records.append(record)

            duration = max(
                0, record.timestamp - self._first_degraded_timestamp
            )
            if (
                not self._issue_reported
                and len(self._degraded_records)
                >= max(1, self._config.consecutive_degraded_records)
                and duration
                >= self._config.minimum_degradation_duration_seconds
            ):
                self._issue_reported = True
                return self._make_issue(record)
            return None

        self._add_healthy_record(record)
        if self._issue_reported:
            self._recovery_count += 1
            if self._recovery_count >= max(1, self._config.recovery_records):
                self._issue_reported = False
                self._recovery_count = 0
                self._degraded_records = []
                self._first_degraded_timestamp = None
        elif self._degraded_records:
            self._degraded_records = []
            self._first_degraded_timestamp = None
        return None

    def observe(self, **kwargs) -> Optional[DiagnosisObservation]:
        training_log = self._collect_training_log(**kwargs)
        if training_log is None or not training_log.logs:
            return None

        issue = None
        for log_line in training_log.logs:
            record = self.parse_log_line(log_line)
            if record is None or not self._history.add(record):
                continue
            candidate = self._evaluate_record(record)
            if issue is None and candidate is not None:
                issue = candidate

        if issue is None:
            return None
        return DiagnosisObservation(
            observation=issue.issue_type,
            extra_infos={"performance_issue": issue.to_json()},
        )

    def resolve(
        self, problem: DiagnosisObservation, **kwargs
    ) -> List[DiagnosisAction]:
        if (
            problem is None
            or problem.observation
            != DiagnosisErrorConstant.TRAINING_SPEED_DEGRADATION
        ):
            return [NoAction()]

        issue_json = problem.extra_infos.get("performance_issue")
        if not issue_json:
            return [NoAction()]
        try:
            issue = PerformanceIssue.from_json(issue_json)
        except (TypeError, ValueError, json.JSONDecodeError):
            return [NoAction()]

        data = PerformanceDiagnosisData(
            issue_type=issue.issue_type,
            metric_name=issue.metric_name,
            metric_value=issue.metric_value,
            baseline_value=issue.baseline_value,
            threshold_value=issue.threshold_value,
            degradation_ratio=issue.degradation_ratio,
            sample_count=issue.sample_count,
            first_timestamp=issue.first_timestamp,
            last_timestamp=issue.last_timestamp,
            first_iteration=issue.first_iteration,
            last_iteration=issue.last_iteration,
            evidence=issue.evidence,
            node_rank=kwargs.get("node_rank", env_utils.get_node_rank()),
        )
        return [
            ReportDiagnosisDataAction(
                data_cls=data.__class__.__name__,
                data_content=data.to_json(),
                node_rank=data.node_rank,
                reason=issue.issue_type,
            )
        ]

