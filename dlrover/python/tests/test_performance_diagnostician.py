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

import os
import unittest
from unittest.mock import patch

from dlrover.python.diagnosis.common.diagnosis_action import (
    NoAction,
    ReportDiagnosisDataAction,
)
from dlrover.python.diagnosis.common.diagnosis_data import (
    PerformanceDiagnosisData,
    TrainingLog,
)
from dlrover.python.diagnosis.common.diagnostician import (
    DiagnosisObservation,
)
from dlrover.python.diagnosis.diagnostician.performance import (
    ELAPSED_TIME_METRIC,
    THROUGHPUT_METRIC,
    PerformanceHistory,
    PerformanceIssue,
    PerformanceDiagnostician,
    PerformanceDiagnosticianConfig,
    PerformanceRecord,
)


class TestPerformanceDiagnostician(unittest.TestCase):
    @staticmethod
    def _log_at(timestamp, iteration, throughput):
        return (
            f"[{timestamp}] "
            "iteration {iteration}/300000 "
            "elapsed time per iteration (ms): 100 | "
            "throughput per GPU (TFLOP/s/GPU): {throughput} | "
            "tokens(B/day/GPU): 1.0"
        ).format(
            iteration=iteration,
            throughput=throughput,
        )

    @classmethod
    def _log(cls, second, iteration, throughput):
        return cls._log_at(
            f"2026-10-15 12:21:{second:02d}",
            iteration,
            throughput,
        )

    def _diagnostician(self):
        return PerformanceDiagnostician(
            PerformanceDiagnosticianConfig(
                minimum_baseline_records=3,
                degradation_window_seconds=2,
                degradation_fraction_threshold=0.3,
                minimum_window_records=3,
                severe_fraction_threshold=1.0,
                recovery_window_seconds=1,
                recovery_healthy_fraction=1.0,
                minimum_recovery_records=2,
            )
        )

    def test_parse_training_log(self):
        record = PerformanceDiagnostician.parse_log_line(
            self._log(13, 144377, 0.6)
        )
        self.assertIsNotNone(record)
        self.assertEqual(record.iteration, 144377)
        self.assertEqual(record.throughput, 0.6)
        self.assertEqual(record.metrics["tokens_b_per_day_per_gpu"], 1.0)

    def test_requires_sustained_degradation_and_deduplicates_tail(self):
        diagnostician = self._diagnostician()
        healthy = [self._log(i, i, 1.0) for i in range(1, 4)]
        degraded = [
            self._log(4, 4, 0.5),
            self._log(5, 5, 0.5),
            self._log(6, 6, 0.5),
        ]

        self.assertIsNone(diagnostician.observe(logs=healthy))
        observation = diagnostician.observe(logs=degraded)
        self.assertIsNotNone(observation)
        self.assertEqual(observation.observation, "training_speed_degradation")

        actions = diagnostician.resolve(
            observation,
            node_id=3,
            node_rank=7,
            job_uid="job-1",
            run_id="run-1",
            rdzv_epoch=2,
            config_fingerprint="config-1",
        )
        self.assertIsInstance(actions[0], ReportDiagnosisDataAction)
        self.assertEqual(actions[0].node_rank, 7)
        self.assertEqual(actions[0].data_type, "PERFORMANCE_ISSUE")
        self.assertEqual(actions[0].schema_version, 1)
        self.assertTrue(actions[0].report_id)

        data = PerformanceDiagnosisData.from_json(actions[0].data_content)
        self.assertEqual(data.report_id, actions[0].report_id)
        self.assertEqual(data.job_uid, "job-1")
        self.assertEqual(data.run_id, "run-1")
        self.assertEqual(data.rdzv_epoch, 2)
        self.assertEqual(data.node_id, 3)

        serialized_action = ReportDiagnosisDataAction.from_json(
            actions[0].to_json()
        )
        self.assertEqual(serialized_action.report_id, actions[0].report_id)
        self.assertEqual(serialized_action.data_type, actions[0].data_type)

        repeated_actions = diagnostician.resolve(
            observation,
            node_id=3,
            node_rank=7,
            job_uid="job-1",
            run_id="run-1",
            rdzv_epoch=2,
            config_fingerprint="config-1",
        )
        self.assertEqual(repeated_actions[0].report_id, actions[0].report_id)
        self.assertEqual(
            repeated_actions[0].data_content, actions[0].data_content
        )
        self.assertIsNone(diagnostician.observe(logs=healthy + degraded))

    def test_confirmed_issue_logs_details(self):
        diagnostician = self._diagnostician()
        self.assertIsNone(
            diagnostician.observe(
                logs=[
                    self._log(1, 1, 1.0),
                    self._log(2, 2, 1.0),
                    self._log(3, 3, 1.0),
                ]
            )
        )

        with self.assertLogs("dlrover.logger", level="INFO") as captured:
            observation = diagnostician.observe(
                logs=[
                    self._log(4, 4, 0.5),
                    self._log(5, 5, 0.5),
                    self._log(6, 6, 0.5),
                ]
            )

        self.assertIsNotNone(observation)
        output = "\n".join(captured.output)
        self.assertIn("Confirmed training speed degradation", output)
        self.assertIn("rule=sustained", output)
        self.assertIn("trigger_throughput=0.5", output)
        self.assertIn("worst_throughput=0.5", output)
        self.assertIn("baseline=1.0", output)
        self.assertIn("threshold=0.7", output)
        self.assertIn("abnormal_fraction=1.000", output)
        self.assertIn("abnormal_samples=3", output)
        self.assertIn("iterations=4..6", output)

    def test_recovery_allows_a_new_issue(self):
        diagnostician = self._diagnostician()
        self.assertIsNone(
            diagnostician.observe(
                logs=[
                    self._log(1, 1, 1.0),
                    self._log(2, 2, 1.0),
                    self._log(3, 3, 1.0),
                ]
            )
        )
        self.assertIsNotNone(
            diagnostician.observe(
                logs=[
                    self._log(4, 4, 0.5),
                    self._log(5, 5, 0.5),
                    self._log(6, 6, 0.5),
                ]
            )
        )
        self.assertIsNone(
            diagnostician.observe(
                logs=[self._log(7, 7, 1.0), self._log(8, 8, 1.0)]
            )
        )
        self.assertIsNotNone(
            diagnostician.observe(
                logs=[
                    self._log(9, 9, 0.5),
                    self._log(10, 10, 0.5),
                    self._log(11, 11, 0.5),
                ]
            )
        )

    def test_sustained_window_uses_steps_and_duration(self):
        diagnostician = PerformanceDiagnostician(
            PerformanceDiagnosticianConfig(
                minimum_baseline_records=1,
                minimum_window_records=3,
                degradation_window_seconds=60,
                degradation_fraction_threshold=1.0,
            )
        )
        self.assertIsNone(
            diagnostician.observe(
                logs=[
                    self._log_at(
                        "2026-10-15 12:00:00",
                        1,
                        1.0,
                    )
                ]
            )
        )
        self.assertIsNone(
            diagnostician.observe(
                logs=[
                    self._log_at(
                        "2026-10-15 12:01:00",
                        2,
                        0.5,
                    ),
                    self._log_at(
                        "2026-10-15 12:03:00",
                        3,
                        0.5,
                    ),
                ]
            )
        )
        self.assertIsNotNone(
            diagnostician.observe(
                logs=[
                    self._log_at(
                        "2026-10-15 12:05:00",
                        4,
                        0.5,
                    )
                ]
            )
        )

    def test_sustained_window_waits_for_duration(self):
        diagnostician = PerformanceDiagnostician(
            PerformanceDiagnosticianConfig(
                minimum_baseline_records=1,
                minimum_window_records=2,
                degradation_window_seconds=120,
                degradation_fraction_threshold=1.0,
            )
        )
        self.assertIsNone(
            diagnostician.observe(
                logs=[self._log_at("2026-10-15 12:00:00", 1, 1.0)]
            )
        )
        self.assertIsNone(
            diagnostician.observe(
                logs=[
                    self._log_at("2026-10-15 12:00:30", 2, 0.5),
                    self._log_at("2026-10-15 12:01:00", 3, 0.5),
                ]
            )
        )
        self.assertIsNotNone(
            diagnostician.observe(
                logs=[self._log_at("2026-10-15 12:02:30", 4, 0.5)]
            )
        )

    def test_consecutive_abnormal_windows_are_required(self):
        diagnostician = PerformanceDiagnostician(
            PerformanceDiagnosticianConfig(
                minimum_baseline_records=1,
                minimum_window_records=2,
                degradation_window_seconds=0,
                degradation_fraction_threshold=1.0,
                consecutive_abnormal_windows=2,
            )
        )
        self.assertIsNone(diagnostician.observe(logs=[self._log(1, 1, 1.0)]))
        self.assertIsNone(
            diagnostician.observe(
                logs=[self._log(2, 2, 0.5), self._log(3, 3, 0.5)]
            )
        )
        self.assertIsNone(
            diagnostician.observe(
                logs=[self._log(4, 4, 1.0), self._log(5, 5, 1.0)]
            )
        )
        self.assertIsNone(
            diagnostician.observe(
                logs=[self._log(6, 6, 0.5), self._log(7, 7, 0.5)]
            )
        )
        self.assertIsNotNone(
            diagnostician.observe(
                logs=[self._log(8, 8, 0.5), self._log(9, 9, 0.5)]
            )
        )

    def test_degradation_ratio_is_configurable(self):
        diagnostician = PerformanceDiagnostician(
            PerformanceDiagnosticianConfig(
                minimum_baseline_records=1,
                minimum_window_records=1,
                degradation_window_seconds=0,
                degradation_fraction_threshold=1.0,
                degradation_ratio=0.8,
            )
        )
        self.assertIsNone(diagnostician.observe(logs=[self._log(1, 1, 1.0)]))
        self.assertIsNotNone(
            diagnostician.observe(logs=[self._log(2, 2, 0.75)])
        )

    def test_sustained_degradation_defaults(self):
        config = PerformanceDiagnosticianConfig()
        self.assertEqual(config.minimum_window_records, 10)
        self.assertEqual(config.degradation_window_seconds, 60)
        self.assertEqual(config.degradation_ratio, 0.7)
        self.assertEqual(config.severe_degradation_ratio, 0.2)
        self.assertEqual(config.consecutive_abnormal_windows, 1)

    def test_config_from_environment(self):
        with patch.dict(
            os.environ,
            {
                "PERFORMANCE_M": "20",
                "PERFORMANCE_N": "1.5",
                "PERFORMANCE_K": "2",
                "PERFORMANCE_P": "0.8",
                "PERFORMANCE_Q": "0.1",
            },
            clear=True,
        ):
            config = PerformanceDiagnosticianConfig.from_env()

        self.assertEqual(config.minimum_window_records, 20)
        self.assertEqual(config.degradation_window_seconds, 90)
        self.assertEqual(config.consecutive_abnormal_windows, 2)
        self.assertEqual(config.degradation_ratio, 0.8)
        self.assertEqual(config.severe_degradation_ratio, 0.1)

    def test_invalid_environment_values_use_defaults(self):
        with patch.dict(
            os.environ,
            {
                "PERFORMANCE_M": "invalid",
                "PERFORMANCE_N": "-1",
                "PERFORMANCE_K": "0",
                "PERFORMANCE_P": "1.1",
                "PERFORMANCE_Q": "invalid",
            },
            clear=True,
        ):
            config = PerformanceDiagnosticianConfig.from_env()

        self.assertEqual(config.minimum_window_records, 10)
        self.assertEqual(config.degradation_window_seconds, 60)
        self.assertEqual(config.consecutive_abnormal_windows, 1)
        self.assertEqual(config.degradation_ratio, 0.7)
        self.assertEqual(config.severe_degradation_ratio, 0.2)

    def test_config_from_empty_environment_uses_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            config = PerformanceDiagnosticianConfig.from_env()

        self.assertEqual(config, PerformanceDiagnosticianConfig())

    def test_performance_record_and_history(self):
        record = PerformanceRecord(
            timestamp=10,
            iteration=3,
            metrics={THROUGHPUT_METRIC: 2.0},
        )
        self.assertEqual(
            PerformanceRecord.from_json(record.to_json()),
            record,
        )

        history = PerformanceHistory(max_records=2, retention_seconds=10)
        self.assertTrue(history.add(record))
        self.assertFalse(history.add(record))
        metadata_only = PerformanceRecord(timestamp=11, metrics={"x": 1.0})
        self.assertTrue(history.add(metadata_only))
        self.assertEqual(history.throughput_baseline, 2.0)
        self.assertTrue(
            history.add(
                PerformanceRecord(
                    timestamp=12,
                    metrics={THROUGHPUT_METRIC: 1.0},
                )
            )
        )
        self.assertEqual(len(history), 2)
        latest = history.latest
        self.assertIsNotNone(latest)
        assert latest is not None
        self.assertEqual(latest.timestamp, 12)
        self.assertTrue(
            history.add(
                PerformanceRecord(
                    timestamp=30,
                    metrics={THROUGHPUT_METRIC: 0.5},
                )
            )
        )
        self.assertEqual(len(history), 1)
        history.clear()
        self.assertEqual(len(history), 0)
        self.assertIsNone(history.latest)
        self.assertIsNone(history.throughput_baseline)

    def test_parse_invalid_and_fallback_records(self):
        self.assertIsNone(
            PerformanceDiagnostician.parse_log_line(
                None  # type: ignore[arg-type]
            )
        )
        self.assertIsNone(
            PerformanceDiagnostician.parse_log_line("no metrics here")
        )
        elapsed_only = PerformanceDiagnostician.parse_log_line(
            "elapsed time per iteration (ms): 12.5"
        )
        self.assertIsNotNone(elapsed_only)
        assert elapsed_only is not None
        self.assertEqual(elapsed_only.metrics[ELAPSED_TIME_METRIC], 12.5)
        self.assertIsNone(elapsed_only.iteration)

        with patch(
            "dlrover.python.diagnosis.diagnostician.performance.time.time",
            return_value=123,
        ):
            invalid_time = PerformanceDiagnostician.parse_log_line(
                "[2026-99-99 99:99:99] throughput per GPU (TFLOP/s/GPU): 1.0"
            )
        self.assertIsNotNone(invalid_time)
        assert invalid_time is not None
        self.assertEqual(invalid_time.timestamp, 123)

    def test_collect_training_log_variants(self):
        diagnostician = self._diagnostician()
        training_log = TrainingLog(logs=["line"])
        self.assertIs(
            diagnostician._collect_training_log(training_log=training_log),
            training_log,
        )
        self.assertIsNone(diagnostician._collect_training_log(logs="line"))
        collected_logs = diagnostician._collect_training_log(
            logs=["line-1", "line-2"]
        )
        self.assertIsNotNone(collected_logs)
        assert collected_logs is not None
        self.assertEqual(collected_logs.logs, ["line-1", "line-2"])
        self.assertIsNone(diagnostician._collect_training_log(log_file=1))
        self.assertIsNone(diagnostician._collect_training_log())

        collector = patch(
            "dlrover.python.diagnosis.diagnostician.performance."
            "TrainingLogCollector"
        )
        with collector as collector_cls:
            collector_cls.return_value.collect_data.return_value = training_log
            self.assertIs(
                diagnostician._collect_training_log(log_file="train.log"),
                training_log,
            )

    def test_severe_degradation_issue(self):
        diagnostician = PerformanceDiagnostician(
            PerformanceDiagnosticianConfig(
                minimum_baseline_records=1,
                minimum_window_records=2,
                degradation_window_seconds=0,
                severe_fraction_threshold=0.3,
                severe_degradation_ratio=0.2,
            )
        )
        self.assertIsNone(diagnostician.observe(logs=[self._log(1, 1, 1.0)]))
        observation = diagnostician.observe(
            logs=[self._log(2, 2, 0.1), self._log(3, 3, 1.0)]
        )
        self.assertIsNotNone(observation)
        assert observation is not None
        issue = PerformanceIssue.from_json(
            observation.extra_infos["performance_issue"]
        )
        self.assertEqual(issue.metric_value, 0.1)
        self.assertAlmostEqual(issue.threshold_value, 0.2)
        self.assertAlmostEqual(issue.degradation_ratio, 0.1)

    def test_severe_window_reuses_duration_and_ratio(self):
        diagnostician = PerformanceDiagnostician(
            PerformanceDiagnosticianConfig(
                minimum_baseline_records=1,
                minimum_window_records=2,
                degradation_window_seconds=120,
                severe_fraction_threshold=1.0,
                severe_degradation_ratio=0.3,
            )
        )
        self.assertIsNone(
            diagnostician.observe(
                logs=[self._log_at("2026-10-15 12:00:00", 1, 1.0)]
            )
        )
        self.assertIsNone(
            diagnostician.observe(
                logs=[
                    self._log_at("2026-10-15 12:00:30", 2, 0.25),
                    self._log_at("2026-10-15 12:01:00", 3, 0.25),
                ]
            )
        )
        observation = diagnostician.observe(
            logs=[self._log_at("2026-10-15 12:02:30", 4, 0.25)]
        )
        self.assertIsNotNone(observation)
        assert observation is not None
        issue = PerformanceIssue.from_json(
            observation.extra_infos["performance_issue"]
        )
        self.assertAlmostEqual(issue.threshold_value, 0.3)

    def test_consecutive_severe_windows_are_required(self):
        diagnostician = PerformanceDiagnostician(
            PerformanceDiagnosticianConfig(
                minimum_baseline_records=1,
                minimum_window_records=2,
                degradation_window_seconds=0,
                degradation_fraction_threshold=1.0,
                consecutive_abnormal_windows=2,
                severe_fraction_threshold=0.5,
            )
        )
        self.assertIsNone(diagnostician.observe(logs=[self._log(1, 1, 1.0)]))
        self.assertIsNone(
            diagnostician.observe(
                logs=[self._log(2, 2, 0.1), self._log(3, 3, 1.0)]
            )
        )
        self.assertIsNotNone(
            diagnostician.observe(
                logs=[self._log(4, 4, 0.1), self._log(5, 5, 1.0)]
            )
        )

    def test_absolute_thresholds(self):
        diagnostician = PerformanceDiagnostician(
            PerformanceDiagnosticianConfig(
                throughput_threshold=2.0,
                severe_throughput_threshold=0.1,
            )
        )
        self.assertEqual(diagnostician._threshold(10.0), 2.0)
        self.assertEqual(diagnostician._severe_threshold(10.0), 0.1)

    def test_empty_windows_and_metadata_only_records(self):
        diagnostician = PerformanceDiagnostician(
            PerformanceDiagnosticianConfig(
                minimum_window_records=1,
                minimum_recovery_records=1,
            )
        )
        self.assertIs(diagnostician.config, diagnostician._config)
        self.assertIs(diagnostician.history, diagnostician._history)
        record = PerformanceRecord(
            timestamp=10,
            metrics={"other": 1.0},
        )
        diagnostician.history.add(
            PerformanceRecord(
                timestamp=10,
                metrics={THROUGHPUT_METRIC: 1.0},
            )
        )
        self.assertIsNone(diagnostician._evaluate_record(record))
        self.assertFalse(
            diagnostician._detect_sustained_degradation(None, 10)[0]
        )
        self.assertFalse(diagnostician._detect_severe_degradation(None, 10)[0])
        self.assertFalse(diagnostician._is_recovered(None, 10))
        self.assertIsNone(diagnostician.observe(logs=[]))

    def test_resolve_rejects_invalid_observations(self):
        diagnostician = self._diagnostician()
        self.assertIsInstance(
            diagnostician.resolve(None)[0],  # type: ignore[arg-type]
            NoAction,
        )
        self.assertIsInstance(
            diagnostician.resolve(DiagnosisObservation(observation="other"))[
                0
            ],
            NoAction,
        )
        missing = DiagnosisObservation("training_speed_degradation")
        self.assertIsInstance(diagnostician.resolve(missing)[0], NoAction)
        invalid = DiagnosisObservation(
            "training_speed_degradation",
            {"performance_issue": "{"},
        )
        self.assertIsInstance(diagnostician.resolve(invalid)[0], NoAction)


if __name__ == "__main__":
    unittest.main()
