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
import time
import unittest
from unittest.mock import MagicMock, patch

from dlrover.python.diagnosis.common.constants import DiagnosisActionType
from dlrover.python.diagnosis.common.diagnosis_action import NodeAction
from dlrover.python.elastic_agent.profiler_controller import (
    ProfilerController,
)


class ProfilerControllerTest(unittest.TestCase):
    def setUp(self):
        env_patcher = patch.dict(os.environ, {"JOB_UID": "job-1"})
        env_patcher.start()
        self.addCleanup(env_patcher.stop)
        self.response = MagicMock()
        self.post = MagicMock(return_value=self.response)

    def _controller(self):
        return ProfilerController(
            node_id=0,
            control_url=(
                "http://127.0.0.1:18888/"
                "HostingService/UpdateCuptiState"
            ),
            request_retries=1,
            stop_retry_interval_seconds=0,
            post_func=self.post,
        )

    def _action(
        self,
        action_id="action-1",
        node_id=0,
        duration="10",
        epoch=2,
    ):
        return NodeAction(
            node_id=node_id,
            node_type="worker",
            instance=node_id,
            action_type=DiagnosisActionType.ENABLE_PROFILING,
            action_id=action_id,
            action_group_id="performance-1",
            diagnosis_id="performance-1",
            job_uid="job-1",
            run_id="run-1",
            rdzv_epoch=epoch,
            parameters={"duration_seconds": duration},
        )

    def test_enable_duplicate_and_explicit_disable(self):
        controller = self._controller()
        action = self._action()
        self.assertTrue(
            controller.enable(
                action,
                current_run_id="run-1",
                current_rdzv_epoch=2,
            )
        )
        self.assertTrue(
            controller.enable(
                action,
                current_run_id="run-1",
                current_rdzv_epoch=2,
            )
        )
        self.assertEqual(
            self.post.call_args_list[0].kwargs["json"],
            {"enabled": True},
        )
        self.assertEqual(
            self.post.call_args_list[0].args[0],
            "http://127.0.0.1:18888/"
            "HostingService/UpdateCuptiState",
        )
        self.assertEqual(self.post.call_count, 1)

        self.assertTrue(controller.disable())
        self.assertFalse(controller.is_active)
        self.assertEqual(
            self.post.call_args_list[-1].kwargs["json"],
            {"enabled": False},
        )
        self.assertEqual(self.post.call_count, 2)

    def test_duration_watchdog_disables_profiling(self):
        controller = self._controller()
        self.assertTrue(
            controller.enable(
                self._action(duration="1"),
                current_run_id="run-1",
                current_rdzv_epoch=2,
            )
        )

        deadline = time.time() + 3
        while self.post.call_count < 2 and time.time() < deadline:
            time.sleep(0.05)

        self.assertEqual(self.post.call_count, 2)
        self.assertEqual(
            self.post.call_args_list[-1].kwargs["json"],
            {"enabled": False},
        )
        self.assertFalse(controller.is_active)

    def test_reject_stale_or_wrong_target_action(self):
        controller = self._controller()
        stale_action = self._action()
        stale_action._timestamp -= stale_action.expired_time_period + 1
        self.assertFalse(controller.enable(stale_action))
        self.assertFalse(controller.enable(self._action(node_id=1)))
        self.assertFalse(
            controller.enable(
                self._action(),
                current_rdzv_epoch=3,
            )
        )
        self.post.assert_not_called()

    def test_start_failure_leaves_controller_idle(self):
        self.post.side_effect = RuntimeError("connection refused")
        controller = self._controller()
        self.assertFalse(controller.enable(self._action()))
        self.assertFalse(controller.is_active)
        self.assertEqual(self.post.call_count, 2)


if __name__ == "__main__":
    unittest.main()
