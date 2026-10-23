# Copyright 2024 The DLRover Authors. All rights reserved.
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

import time
import unittest
from typing import List
from unittest.mock import MagicMock, patch

from dlrover.python.common.constants import (
    Accelerators,
    GpuMetricEnum,
    NpuMetricEnum,
    PlatformType,
    PreCheckStatus,
)
from dlrover.python.common.global_context import Context
from dlrover.python.common.metric.context import JobMetricContext
from dlrover.python.common.metric.monitor import (
    GpuMetricMonitor,
    NpuMetricMonitor,
)
from dlrover.python.common.node import Node
from dlrover.python.diagnosis.common.constants import (
    DiagnosisDataType,
)
from dlrover.python.diagnosis.common.diagnosis_action import (
    DiagnosisAction,
    NoAction,
)
from dlrover.python.diagnosis.common.diagnosis_data import (
    PerformanceDiagnosisData,
    TrainingLog,
)
from dlrover.python.master.diagnosis.diagnosis_data_manager import (
    DiagnosisDataManager,
    STORE_ACCEPTED,
    STORE_CONFLICT,
    STORE_DUPLICATE,
    STORE_REJECTED,
)
from dlrover.python.master.diagnosis.diagnosis_master import DiagnosisMaster
from dlrover.python.master.diagnosis.precheck_operator import (
    PreCheckOperator,
    PreCheckResult,
)
from dlrover.python.master.node.job_context import get_job_context
from dlrover.python.scheduler.kubernetes import K8sJobArgs
from dlrover.python.util.function_util import TimeoutException

_metric_context = JobMetricContext.singleton_instance()
_dlrover_context = Context.singleton_instance()


class DiagnosisMasterTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_data_manager(self):
        mgr = DiagnosisDataManager(1)
        log1 = TrainingLog(0)
        mgr.store_data(log1)
        time.sleep(0.01)
        log2 = TrainingLog(0)
        mgr.store_data(log2)

        logs = mgr.get_data(DiagnosisDataType.TRAINING_LOG)
        self.assertEqual(len(logs), 2)

        time.sleep(1.5)
        log3 = TrainingLog(0)
        mgr.store_data(log3)
        logs = mgr.get_data(DiagnosisDataType.TRAINING_LOG)
        self.assertEqual(len(logs), 1)

    def test_performance_report_storage(self):
        mgr = DiagnosisDataManager(
            expire_time_period=1800,
            max_records_per_type=2,
        )

        def new_report(report_id, throughput, schema_version=1):
            return PerformanceDiagnosisData(
                timestamp=123,
                issue_type="training_speed_degradation",
                metric_name="throughput_per_gpu_tflops",
                metric_value=throughput,
                baseline_value=1.0,
                threshold_value=0.7,
                sample_count=2,
                first_timestamp=100,
                last_timestamp=101,
                first_iteration=10,
                last_iteration=11,
                report_id=report_id,
                job_uid="job-1",
                run_id="run-1",
                rdzv_epoch=2,
                config_fingerprint="config-1",
                schema_version=schema_version,
            )

        self.assertEqual(
            mgr.store_data(new_report("missing-source", 0.5)),
            STORE_REJECTED,
        )
        self.assertEqual(
            mgr.store_data(
                new_report("unsupported-schema", 0.5, schema_version=2),
                node_id=3,
                node_type="worker",
                node_rank=1,
            ),
            STORE_REJECTED,
        )

        report = new_report("report-1", 0.5)
        self.assertEqual(
            mgr.store_data(
                report,
                node_id=3,
                node_type="worker",
                node_rank=1,
            ),
            STORE_ACCEPTED,
        )
        self.assertEqual(report.node_id, 3)
        self.assertEqual(report.node_rank, 1)
        self.assertGreater(report.received_at, 0)
        self.assertEqual(report.sequence_id, 1)

        duplicate = new_report("report-1", 0.5)
        self.assertEqual(
            mgr.store_data(
                duplicate,
                node_id=3,
                node_type="worker",
                node_rank=1,
            ),
            STORE_DUPLICATE,
        )
        self.assertEqual(len(mgr.get_data("PERFORMANCE_ISSUE")), 1)

        conflict = new_report("report-1", 0.6)
        self.assertEqual(
            mgr.store_data(
                conflict,
                node_id=3,
                node_type="worker",
                node_rank=1,
            ),
            STORE_CONFLICT,
        )

        second = new_report("report-2", 0.4)
        mgr.store_data(second, node_id=4, node_type="worker", node_rank=2)
        third = new_report("report-3", 0.3)
        mgr.store_data(third, node_id=5, node_type="worker", node_rank=3)

        reports = mgr.get_data(
            "PERFORMANCE_ISSUE",
            after_sequence_id=second.sequence_id,
        )
        self.assertEqual([data.report_id for data in reports], ["report-3"])
        self.assertEqual(
            mgr.get_latest_sequence_id("PERFORMANCE_ISSUE"),
            third.sequence_id,
        )

    def test_diagnosis_master_api(self):
        args = K8sJobArgs(PlatformType.KUBERNETES, "default", "test")
        args.xpu_type = Accelerators.NVIDIA_GPU
        monitor = GpuMetricMonitor(
            job_name=args.job_name,
            metrics=[
                GpuMetricEnum.GPU_UTIL,
                GpuMetricEnum.GPU_TENSOR_UTIL,
            ],
        )
        mgr = DiagnosisMaster(job_args=args)
        mgr.new_metric_monitor(monitor)
        mgr.start_metric_collect()
        mgr.stop_metric_collect()

        args.xpu_type = Accelerators.ASCEND_NPU
        monitor = NpuMetricMonitor(
            job_name=args.job_name,
            metrics=[
                NpuMetricEnum.NPU_UTIL,
            ],
        )
        mgr = DiagnosisMaster(job_args=args)
        mgr.new_metric_monitor(monitor)
        mgr.start_metric_collect()
        mgr.stop_metric_collect()

        mgr = DiagnosisMaster(job_args=args)
        mgr.pre_check()
        mgr.start_observing()
        mgr.stop_observing()

    @patch(
        "dlrover.python.master.diagnosis.diagnosis_master.get_pre_check_timeout"
    )
    def test_pre_check(self, mock_get_pre_check_timeout):
        mock_get_pre_check_timeout.return_value = 2
        job_context = get_job_context()
        mgr = DiagnosisMaster()
        original_get_pre_check_status = mgr._job_context.get_pre_check_status

        mgr.pre_check()
        self.assertEqual(job_context._action_queue.len(), 0)

        _dlrover_context = Context.singleton_instance()
        _dlrover_context.pre_check_operators = [
            ("dlrover.python.tests.test_diagnosis_master", "TestOperator")
        ]
        mgr._job_context.get_pre_check_status = MagicMock(
            return_value=PreCheckStatus.CHECKING
        )
        mgr.get_pre_check_operators = MagicMock(return_value=[TestOperator()])
        try:
            mgr.pre_check()
            self.fail()
        except TimeoutException:
            pass
        mgr._job_context.get_pre_check_status = original_get_pre_check_status

        _dlrover_context.pre_check_operators = [
            (
                "dlrover.python.tests.test_diagnosis_master",
                "TestOperator",
                True,
            )
        ]
        mgr.pre_check()
        self.assertEqual(
            job_context.get_pre_check_status(), PreCheckStatus.PASS
        )

        _dlrover_context.pre_check_operators = []
        mgr.pre_check()
        self.assertEqual(
            job_context.get_pre_check_status(), PreCheckStatus.DISABLED
        )

        mgr._job_context.get_pre_check_status = MagicMock(
            return_value=PreCheckStatus.PASS
        )
        mgr.pre_check()
        mgr._job_context.get_pre_check_status = original_get_pre_check_status


class TestOperator(PreCheckOperator):
    @classmethod
    def get_retry_interval_secs(cls) -> int:
        raise TimeoutException()

    def check(self, *args, **kwargs) -> PreCheckResult:
        return PreCheckResult(1, "test", [Node("worker", 0)])

    def failed_actions(self, *args, **kwargs) -> List[DiagnosisAction]:
        return [NoAction()]
