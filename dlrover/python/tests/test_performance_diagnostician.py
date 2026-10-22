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

import unittest

from dlrover.python.diagnosis.common.diagnosis_action import (
    ReportDiagnosisDataAction,
)
from dlrover.python.diagnosis.diagnostician.performance import (
    PerformanceDiagnostician,
    PerformanceDiagnosticianConfig,
)


class TestPerformanceDiagnostician(unittest.TestCase):
    @staticmethod
    def _log(second, iteration, throughput):
        return (
            "[dlrover-trainer7]:[2026-10-15 12:21:{second:02d}] "
            "iteration {iteration}/300000 "
            "elapsed time per iteration (ms): 100 | "
            "throughput per GPU (TFLOP/s/GPU): {throughput} | "
            "tokens(B/day/GPU): 1.0"
        ).format(
            second=second,
            iteration=iteration,
            throughput=throughput,
        )

    def _diagnostician(self):
        return PerformanceDiagnostician(
            PerformanceDiagnosticianConfig(
                minimum_baseline_records=3,
                consecutive_degraded_records=2,
                minimum_degradation_duration_seconds=0,
                recovery_records=2,
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
        degraded = [self._log(4, 4, 0.5), self._log(5, 5, 0.5)]

        self.assertIsNone(diagnostician.observe(logs=healthy))
        observation = diagnostician.observe(logs=degraded)
        self.assertIsNotNone(observation)
        self.assertEqual(observation.observation, "training_speed_degradation")

        actions = diagnostician.resolve(observation, node_rank=7)
        self.assertIsInstance(actions[0], ReportDiagnosisDataAction)
        self.assertEqual(actions[0].node_rank, 7)
        self.assertIsNone(diagnostician.observe(logs=healthy + degraded))

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
                logs=[self._log(4, 4, 0.5), self._log(5, 5, 0.5)]
            )
        )
        self.assertIsNone(
            diagnostician.observe(
                logs=[self._log(6, 6, 1.0), self._log(7, 7, 1.0)]
            )
        )
        self.assertIsNotNone(
            diagnostician.observe(
                logs=[self._log(8, 8, 0.5), self._log(9, 9, 0.5)]
            )
        )


if __name__ == "__main__":
    unittest.main()
