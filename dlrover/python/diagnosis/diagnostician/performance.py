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
import math
import re
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from statistics import median
from typing import Any, Deque, Dict, Iterable, List, Optional, Set, Tuple

from dlrover.python.common import env_utils
from dlrover.python.common.log import default_logger as logger
from dlrover.python.diagnosis.common.constants import (
    DiagnosisErrorConstant,
    EnvConfigKey,
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
_ITERATION_PATTERN = re.compile(r"\biteration\s*:?\s*(\d+)", re.IGNORECASE)
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


def _get_int_env(env_key: str, default: int, minimum: int) -> int:
    raw_value = env_utils.get_env(env_key)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        value = minimum - 1
    if value < minimum:
        logger.warning(
            f"Invalid {env_key}={raw_value}; use default {default}."
        )
        return default
    return value


def _get_float_env(
    env_key: str,
    default: float,
    minimum: float,
    maximum: Optional[float] = None,
) -> float:
    raw_value = env_utils.get_env(env_key)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        value = float("nan")
    if (
        not math.isfinite(value)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        logger.warning(
            f"Invalid {env_key}={raw_value}; use default {default}."
        )
        return default
    return value


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
    def from_dict(cls, value: Dict[str, Any]) -> "PerformanceRecord":
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
        values: List[float] = []
        for record in self._records:
            throughput = record.throughput
            if throughput is not None:
                values.append(throughput)
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
    def from_dict(cls, value: Dict[str, Any]) -> "PerformanceIssue":
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

    # p in the performance rule. If an absolute threshold is not configured,
    # the degradation threshold is healthy_baseline * degradation_ratio.
    throughput_threshold: Optional[float] = None
    degradation_ratio: float = 0.7

    # m and n in the performance rule. A tumbling window closes after both
    # conditions are met: at least m samples and at least n seconds.
    minimum_window_records: int = 10
    degradation_window_seconds: int = 60

    # Fraction of samples in a window that must be below the degradation
    # threshold for the window to be abnormal.
    degradation_fraction_threshold: float = 0.70

    # k in both performance rules. Only consecutive, non-overlapping
    # abnormal windows are counted.
    consecutive_abnormal_windows: int = 1

    # q in the severe rule. The severe threshold is
    # healthy_baseline * severe_degradation_ratio unless an absolute
    # threshold is configured. Severe windows reuse m, n, and k.
    severe_throughput_threshold: Optional[float] = None
    severe_degradation_ratio: float = 0.20
    severe_fraction_threshold: float = 0.10

    # Recovery is also window based to avoid clearing an issue after
    # only a few lucky healthy samples.
    recovery_window_seconds: int = 60
    recovery_healthy_fraction: float = 0.90
    minimum_recovery_records: int = 10

    maximum_evidence_records: int = 5

    @classmethod
    def from_env(cls) -> "PerformanceDiagnosticianConfig":
        """Build a config from PERFORMANCE_M/N/K/P/Q.

        N is expressed in minutes. Missing or invalid values use the
        configuration defaults.
        """
        config = cls()
        config.minimum_window_records = _get_int_env(
            EnvConfigKey.PERFORMANCE_M,
            config.minimum_window_records,
            1,
        )
        default_minutes = config.degradation_window_seconds / 60.0
        window_minutes = _get_float_env(
            EnvConfigKey.PERFORMANCE_N,
            default_minutes,
            0.0,
        )
        config.degradation_window_seconds = int(round(window_minutes * 60))
        config.consecutive_abnormal_windows = _get_int_env(
            EnvConfigKey.PERFORMANCE_K,
            config.consecutive_abnormal_windows,
            1,
        )
        config.degradation_ratio = _get_float_env(
            EnvConfigKey.PERFORMANCE_P,
            config.degradation_ratio,
            0.0,
            1.0,
        )
        config.severe_degradation_ratio = _get_float_env(
            EnvConfigKey.PERFORMANCE_Q,
            config.severe_degradation_ratio,
            0.0,
            1.0,
        )
        return config


class PerformanceDiagnostician(Diagnostician):
    """Diagnose training-speed degradation on one worker.

    Detection uses two independent rules:

    1. sustained degradation: a high fraction of samples in tumbling windows
       of at least m samples and n seconds are below p * baseline;
    2. severe intermittent degradation: a smaller fraction of samples in the
       same size tumbling windows are below q * baseline.

    Both rules confirm an issue after k consecutive abnormal windows.

    The existing TrainingLogCollector, PerformanceHistory, report path, and
    PerformanceIssue/PerformanceDiagnosisData structures are reused.
    """

    def __init__(
        self, config: Optional[PerformanceDiagnosticianConfig] = None
    ):
        super().__init__()
        self._config = config or PerformanceDiagnosticianConfig()
        self._history = PerformanceHistory(
            self._config.history_size,
            self._config.history_retention_seconds,
        )
        self._healthy_records: Deque[PerformanceRecord] = deque(
            maxlen=max(1, self._config.baseline_window_size)
        )

        self._issue_reported = False
        self._sustained_window_started_at: Optional[int] = None
        self._sustained_window_sample_count = 0
        self._sustained_window_abnormal_count = 0
        self._sustained_window_threshold: Optional[float] = None
        self._sustained_window_evidence: Deque[PerformanceRecord] = deque(
            maxlen=max(1, self._config.maximum_evidence_records)
        )
        self._consecutive_abnormal_windows = 0
        self._severe_window_started_at: Optional[int] = None
        self._severe_window_sample_count = 0
        self._severe_window_abnormal_count = 0
        self._severe_window_threshold: Optional[float] = None
        self._severe_window_evidence: Deque[PerformanceRecord] = deque(
            maxlen=max(1, self._config.maximum_evidence_records)
        )
        self._consecutive_severe_windows = 0

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
            match = re.search(rf"([^:|]+):\s*({_NUMBER})\s*$", segment.strip())
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
            if isinstance(logs, (str, bytes)) or not isinstance(
                logs, Iterable
            ):
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

    # A separate threshold models catastrophic slowdown instead of
    # treating a 69%-of-baseline sample and a 5%-of-baseline sample equally.
    def _severe_threshold(self, baseline: Optional[float]) -> Optional[float]:
        if self._config.severe_throughput_threshold is not None:
            return self._config.severe_throughput_threshold
        if baseline is None:
            return None
        return baseline * self._config.severe_degradation_ratio

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

    # Obtain throughput samples from the existing PerformanceHistory
    # instead of maintaining a separate consecutive-degradation list.
    def _throughput_records_in_window(
        self, now: int, window_seconds: int
    ) -> List[PerformanceRecord]:
        start = now - max(0, window_seconds)
        return [
            item
            for item in self._history.records
            if item.throughput is not None and start <= item.timestamp <= now
        ]

    def _reset_sustained_window(self):
        self._sustained_window_started_at = None
        self._sustained_window_sample_count = 0
        self._sustained_window_abnormal_count = 0
        self._sustained_window_threshold = None
        self._sustained_window_evidence.clear()

    def _reset_sustained_detection(self):
        self._reset_sustained_window()
        self._consecutive_abnormal_windows = 0

    def _reset_severe_window(self):
        self._severe_window_started_at = None
        self._severe_window_sample_count = 0
        self._severe_window_abnormal_count = 0
        self._severe_window_threshold = None
        self._severe_window_evidence.clear()

    def _reset_severe_detection(self):
        self._reset_severe_window()
        self._consecutive_severe_windows = 0

    def _reset_performance_windows(self):
        self._reset_sustained_detection()
        self._reset_severe_detection()

    # Rule A -- sustained degradation in non-overlapping, tumbling windows.
    def _detect_sustained_degradation(
        self,
        baseline: Optional[float],
        now: int,
        record: Optional[PerformanceRecord] = None,
    ) -> Tuple[
        bool,
        Optional[float],
        float,
        List[PerformanceRecord],
        int,
    ]:
        if record is None:
            record = self._history.latest
        if record is None:
            return False, None, 0.0, [], 0

        throughput = record.throughput
        if throughput is None or baseline is None:
            return False, None, 0.0, [], 0

        window_started_at = self._sustained_window_started_at
        if window_started_at is None:
            window_threshold = self._threshold(baseline)
            if window_threshold is None:
                return False, None, 0.0, [], 0
            window_started_at = record.timestamp
            self._sustained_window_started_at = window_started_at
            self._sustained_window_threshold = window_threshold

        threshold = self._sustained_window_threshold
        if threshold is None:
            return False, None, 0.0, [], 0
        self._sustained_window_sample_count += 1
        if throughput < threshold:
            self._sustained_window_abnormal_count += 1
            self._sustained_window_evidence.append(record)

        fraction = (
            self._sustained_window_abnormal_count
            / self._sustained_window_sample_count
        )
        minimum_steps = max(1, self._config.minimum_window_records)
        minimum_duration = max(0, self._config.degradation_window_seconds)
        elapsed = now - window_started_at
        if (
            self._sustained_window_sample_count < minimum_steps
            or elapsed < minimum_duration
        ):
            return (
                False,
                threshold,
                fraction,
                [],
                0,
            )

        abnormal_records = list(self._sustained_window_evidence)
        abnormal_count = self._sustained_window_abnormal_count
        window_is_abnormal = (
            fraction >= self._config.degradation_fraction_threshold
        )
        if window_is_abnormal:
            self._consecutive_abnormal_windows += 1
        else:
            self._consecutive_abnormal_windows = 0

        detected = self._consecutive_abnormal_windows >= max(
            1, self._config.consecutive_abnormal_windows
        )
        self._reset_sustained_window()
        if detected:
            self._consecutive_abnormal_windows = 0

        return detected, threshold, fraction, abnormal_records, abnormal_count

    # Rule B -- sparse/extreme degradation in the same tumbling windows.
    def _detect_severe_degradation(
        self,
        baseline: Optional[float],
        now: int,
        record: Optional[PerformanceRecord] = None,
    ) -> Tuple[
        bool,
        Optional[float],
        float,
        List[PerformanceRecord],
        int,
    ]:
        if record is None:
            record = self._history.latest
        if record is None:
            return False, None, 0.0, [], 0

        throughput = record.throughput
        if throughput is None or baseline is None:
            return False, None, 0.0, [], 0

        window_started_at = self._severe_window_started_at
        if window_started_at is None:
            window_threshold = self._severe_threshold(baseline)
            if window_threshold is None:
                return False, None, 0.0, [], 0
            window_started_at = record.timestamp
            self._severe_window_started_at = window_started_at
            self._severe_window_threshold = window_threshold

        threshold = self._severe_window_threshold
        if threshold is None:
            return False, None, 0.0, [], 0
        self._severe_window_sample_count += 1
        if throughput < threshold:
            self._severe_window_abnormal_count += 1
            self._severe_window_evidence.append(record)

        fraction = (
            self._severe_window_abnormal_count
            / self._severe_window_sample_count
        )
        minimum_steps = max(1, self._config.minimum_window_records)
        minimum_duration = max(0, self._config.degradation_window_seconds)
        elapsed = now - window_started_at
        if (
            self._severe_window_sample_count < minimum_steps
            or elapsed < minimum_duration
        ):
            return False, threshold, fraction, [], 0

        abnormal_records = list(self._severe_window_evidence)
        abnormal_count = self._severe_window_abnormal_count
        window_is_abnormal = fraction >= self._config.severe_fraction_threshold
        if window_is_abnormal:
            self._consecutive_severe_windows += 1
        else:
            self._consecutive_severe_windows = 0

        detected = self._consecutive_severe_windows >= max(
            1, self._config.consecutive_abnormal_windows
        )
        self._reset_severe_window()
        if detected:
            self._consecutive_severe_windows = 0

        return detected, threshold, fraction, abnormal_records, abnormal_count

    # Recovery uses a recent window instead of N consecutive samples.
    def _is_recovered(
        self,
        baseline: Optional[float],
        now: int,
    ) -> bool:
        records = self._throughput_records_in_window(
            now, self._config.recovery_window_seconds
        )
        if len(records) < max(1, self._config.minimum_recovery_records):
            return False

        threshold = self._threshold(baseline)
        if threshold is None:
            return False

        healthy_count = sum(
            1
            for item in records
            if item.throughput is not None and item.throughput >= threshold
        )
        healthy_fraction = healthy_count / len(records)
        return healthy_fraction >= self._config.recovery_healthy_fraction

    # Build evidence from the records that actually triggered the
    # selected rule. The metric value is the worst observed throughput rather
    # than necessarily the most recent sample.
    def _make_issue(
        self,
        record: PerformanceRecord,
        baseline: Optional[float],
        threshold: Optional[float],
        abnormal_records: List[PerformanceRecord],
        sample_count: Optional[int] = None,
    ) -> PerformanceIssue:
        evidence_records = abnormal_records or [record]
        evidence_records = sorted(
            evidence_records, key=lambda item: item.timestamp
        )

        representative = min(
            evidence_records,
            key=lambda item: (
                item.throughput
                if item.throughput is not None
                else float("inf")
            ),
        )
        throughput = representative.throughput
        ratio = (
            throughput / baseline
            if throughput is not None
            and baseline is not None
            and baseline != 0
            else None
        )
        evidence = [
            item.to_dict()
            for item in evidence_records[
                -self._config.maximum_evidence_records :
            ]
        ]

        return PerformanceIssue(
            metric_value=representative.throughput,
            baseline_value=baseline,
            threshold_value=threshold,
            degradation_ratio=ratio,
            sample_count=(
                len(abnormal_records) if sample_count is None else sample_count
            ),
            first_timestamp=evidence_records[0].timestamp,
            last_timestamp=evidence_records[-1].timestamp,
            first_iteration=evidence_records[0].iteration,
            last_iteration=evidence_records[-1].iteration,
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
        normal_threshold = self._threshold(baseline)

        # Once an issue has been reported, do not generate another
        # issue until a window-based recovery condition is satisfied.
        if self._issue_reported:
            if self._is_recovered(baseline, record.timestamp):
                self._issue_reported = False
                self._reset_performance_windows()
                # Only feed a recovered/healthy value back into the healthy
                # baseline. This prevents an active incident from pulling the
                # baseline downward.
                if normal_threshold is None or throughput >= normal_threshold:
                    self._add_healthy_record(record)
            return None

        # Evaluate both independent detectors. Severe degradation is
        # checked first so evidence/threshold reflect the stronger condition
        # when both rules are true at the same time.
        (
            severe_detected,
            severe_threshold,
            _,
            severe_records,
            severe_count,
        ) = self._detect_severe_degradation(
            baseline,
            record.timestamp,
            record,
        )
        (
            sustained_detected,
            sustained_threshold,
            _,
            degraded_records,
            degraded_count,
        ) = self._detect_sustained_degradation(
            baseline,
            record.timestamp,
            record,
        )

        if severe_detected:
            self._issue_reported = True
            self._reset_performance_windows()
            return self._make_issue(
                record,
                baseline,
                severe_threshold,
                severe_records,
                severe_count,
            )

        if sustained_detected:
            self._issue_reported = True
            self._reset_performance_windows()
            return self._make_issue(
                record,
                baseline,
                sustained_threshold,
                degraded_records,
                degraded_count,
            )

        # Evaluate the normal threshold and add the record to the healthy baseline
        # if it meets the criteria.
        if normal_threshold is None or throughput >= normal_threshold:
            self._add_healthy_record(record)

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

        job_uid = str(kwargs.get("job_uid", env_utils.get_job_uid()) or "")
        run_id = str(
            kwargs.get("run_id")
            or env_utils.get_env("TORCHELASTIC_RUN_ID")
            or ""
        )
        rdzv_epoch = int(kwargs.get("rdzv_epoch", 0) or 0)
        config_fingerprint = str(kwargs.get("config_fingerprint", "") or "")
        node_id = int(kwargs.get("node_id", env_utils.get_node_id()))
        node_rank = int(kwargs.get("node_rank", env_utils.get_node_rank()))
        report_key = ":".join(
            [
                job_uid,
                run_id,
                str(rdzv_epoch),
                config_fingerprint,
                str(node_id),
                str(node_rank),
                issue.issue_type,
                str(issue.first_timestamp),
                str(issue.last_iteration),
            ]
        )
        report_id = str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"dlrover-performance:{report_key}")
        )

        data = PerformanceDiagnosisData(
            timestamp=issue.last_timestamp or int(time.time()),
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
            node_id=node_id,
            node_rank=node_rank,
            report_id=report_id,
            job_uid=job_uid,
            run_id=run_id,
            rdzv_epoch=rdzv_epoch,
            config_fingerprint=config_fingerprint,
        )
        return [
            ReportDiagnosisDataAction(
                data_cls=data.__class__.__name__,
                data_content=data.to_json(),
                node_rank=data.node_rank,
                reason=issue.issue_type,
                report_id=data.report_id,
                data_type=data.data_type,
                schema_version=data.schema_version,
            )
        ]
