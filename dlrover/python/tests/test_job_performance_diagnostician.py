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

from dlrover.python.common.constants import NodeStatus, NodeType
from dlrover.python.common.node import Node
from dlrover.python.diagnosis.common.constants import (
    DiagnosisActionType,
    DiagnosisState,
)
from dlrover.python.diagnosis.common.diagnosis_action import (
    NodeAction,
    NoAction,
)
from dlrover.python.diagnosis.common.diagnosis_data import (
    PerformanceDiagnosisData,
)
from dlrover.python.master.diagnosis.diagnosis_data_manager import (
    DiagnosisDataManager,
)
from dlrover.python.master.diagnosis.diagnostician.performance import (
    JobPerformanceDiagnostician,
)


class JobPerformanceDiagnosticianTest(unittest.TestCase):
    def setUp(self):
        self.data_manager = DiagnosisDataManager(expire_time_period=1800)
        self.job_nodes = {
            NodeType.WORKER: {
                0: Node(
                    NodeType.WORKER,
                    0,
                    status=NodeStatus.RUNNING,
                    rank_index=0,
                ),
                1: Node(
                    NodeType.WORKER,
                    1,
                    status=NodeStatus.RUNNING,
                    rank_index=1,
                ),
                2: Node(
                    NodeType.WORKER,
                    2,
                    status=NodeStatus.RUNNING,
                    rank_index=2,
                ),
                3: Node(
                    NodeType.WORKER,
                    3,
                    status=NodeStatus.FAILED,
                    rank_index=3,
                ),
            }
        }
        self.diagnostician = JobPerformanceDiagnostician(self.data_manager)

    def _report(
        self,
        report_id,
        node_id=0,
        timestamp=1000,
        epoch=2,
    ):
        report = PerformanceDiagnosisData(
            timestamp=timestamp,
            issue_type="training_speed_degradation",
            metric_name="throughput_per_gpu_tflops",
            metric_value=0.5,
            baseline_value=1.0,
            threshold_value=0.7,
            degradation_ratio=0.5,
            sample_count=4,
            first_timestamp=max(0, timestamp - 10),
            last_timestamp=timestamp,
            first_iteration=10,
            last_iteration=13,
            report_id=report_id,
            job_uid="job-1",
            run_id="run-1",
            rdzv_epoch=epoch,
            config_fingerprint="config-1",
        )
        self.data_manager.store_data(
            report,
            node_id=node_id,
            node_type=NodeType.WORKER,
            node_rank=node_id,
        )
        return report

    def _context(self, epoch=2):
        return {
            "job_nodes": self.job_nodes,
            "job_uid": "job-1",
            "rdzv_epoch": epoch,
            "rdzv_node_ids": [0, 1, 2],
            "observing_paused": False,
        }

    def test_create_profiling_actions_for_active_workers(self):
        self._report("report-1")
        actions = self.diagnostician.diagnose(**self._context())

        self.assertEqual(len(actions), 3)
        self.assertTrue(all(isinstance(a, NodeAction) for a in actions))
        self.assertEqual(
            {action.instance for action in actions}, {0, 1, 2}
        )
        self.assertTrue(
            all(
                action.action_type
                == DiagnosisActionType.ENABLE_PROFILING
                for action in actions
            )
        )
        self.assertEqual(
            len({action.action_id for action in actions}), 3
        )
        self.assertEqual(
            len({action.action_group_id for action in actions}), 1
        )
        self.assertTrue(
            all(action.rdzv_epoch == 2 for action in actions)
        )
        self.assertTrue(
            all(
                action.parameters["duration_seconds"] == "180"
                for action in actions
            )
        )

        rounds = list(self.diagnostician.rounds.values())
        self.assertEqual(len(rounds), 1)
        self.assertEqual(rounds[0].state, DiagnosisState.CLOSED)
        self.assertEqual(
            rounds[0].planned_action_ids,
            [action.action_id for action in actions],
        )
        self.assertIsInstance(
            self.diagnostician.diagnose(**self._context())[0],
            NoAction,
        )

    def test_correlate_reports_and_reject_stale_epoch(self):
        self._report("report-1", node_id=0)
        self._report("report-2", node_id=1)
        actions = self.diagnostician.diagnose(**self._context())
        self.assertEqual(len(actions), 3)
        rounds = list(self.diagnostician.rounds.values())
        self.assertEqual(
            rounds[0].source_report_ids, ["report-1", "report-2"]
        )

        stale_data_manager = DiagnosisDataManager(expire_time_period=1800)
        another = JobPerformanceDiagnostician(stale_data_manager)
        stale_report = PerformanceDiagnosisData(
            timestamp=2000,
            issue_type="training_speed_degradation",
            metric_name="throughput_per_gpu_tflops",
            metric_value=0.5,
            baseline_value=1.0,
            threshold_value=0.7,
            degradation_ratio=0.5,
            sample_count=4,
            first_timestamp=1990,
            last_timestamp=2000,
            first_iteration=10,
            last_iteration=13,
            report_id="stale-report",
            job_uid="job-1",
            run_id="run-1",
            rdzv_epoch=1,
            config_fingerprint="config-1",
        )
        stale_data_manager.store_data(
            stale_report,
            node_id=0,
            node_type=NodeType.WORKER,
            node_rank=0,
        )
        result = another.diagnose(**self._context(epoch=2))
        self.assertIsInstance(result[0], NoAction)
        stale_round = list(another.rounds.values())[0]
        self.assertEqual(stale_round.state, DiagnosisState.CLOSED)
        self.assertEqual(
            stale_round.explanation,
            "report rendezvous epoch is stale",
        )

    def test_pause_does_not_consume_reports(self):
        self._report("report-1")
        context = self._context()
        context["observing_paused"] = True
        result = self.diagnostician.diagnose(**context)
        self.assertIsInstance(result[0], NoAction)
        self.assertEqual(
            self.diagnostician.last_processed_sequence_id, 0
        )

        context["observing_paused"] = False
        actions = self.diagnostician.diagnose(**context)
        self.assertEqual(len(actions), 3)

    def test_recurrence_creates_a_new_round(self):
        self.diagnostician = JobPerformanceDiagnostician(
            self.data_manager,
            correlation_window_seconds=30,
        )
        self._report("report-1", timestamp=1000)
        first_actions = self.diagnostician.diagnose(**self._context())
        self._report("report-2", timestamp=1100)
        second_actions = self.diagnostician.diagnose(**self._context())

        self.assertEqual(len(first_actions), 3)
        self.assertEqual(len(second_actions), 3)
        self.assertEqual(len(self.diagnostician.rounds), 2)
        self.assertNotEqual(
            first_actions[0].action_group_id,
            second_actions[0].action_group_id,
        )

    def test_node_action_round_trip_preserves_metadata(self):
        self._report("report-1")
        action = self.diagnostician.diagnose(**self._context())[0]
        restored = NodeAction.from_json(action.to_json())
        self.assertEqual(restored.action_id, action.action_id)
        self.assertEqual(restored.action_group_id, action.action_group_id)
        self.assertEqual(restored.diagnosis_id, action.diagnosis_id)
        self.assertEqual(restored.rdzv_epoch, action.rdzv_epoch)
        self.assertEqual(restored.job_uid, action.job_uid)


if __name__ == "__main__":
    unittest.main()
