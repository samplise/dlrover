# Copyright 2022 The EasyDL Authors. All rights reserved.
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

from dlrover.python.common.constants import (
    DistributionStrategy,
    NodeEventType,
    NodeExitReason,
    NodeStatus,
    NodeType,
)
from dlrover.python.master.node_manager.event_callback import (
    TaskRescheduleCallback,
)
from dlrover.python.master.node_manager.job_config import (
    JobResourceConfig,
    get_critical_worker_index,
    set_critical_node,
)
from dlrover.python.master.node_manager.node_manager import create_node_manager
from dlrover.python.master.node_manager.status_flow import (
    NODE_STATE_FLOWS,
    NodeStateFlow,
    get_node_state_flow,
)
from dlrover.python.master.node_watcher.base_watcher import Node, NodeEvent
from dlrover.python.tests.test_utils import (
    create_task_manager,
    mock_list_job_pods,
)

_MOCK_JOB_UUID = "11111"


def get_service_fn(*args):
    return "test:2222"


def get_job_uuid():
    return "11111"


class MockArgs(object):
    def __init__(self):
        self.job_name = "test"
        self.namespace = "test"
        self.ps_is_critical = True
        self.ps_relaunch_max_num = 1
        self.use_ddp = False
        self.critical_worker_index = "0:3"
        self.distribution_strategy = DistributionStrategy.PARAMETER_SERVER
        self.relaunch_on_worker_failure = 1
        self.num_workers = 3
        self.worker_resource_request = "cpu=1,memory=4096Mi"
        self.worker_pod_priority = ""

        self.num_ps_pods = 3
        self.ps_resource_request = "cpu=1,memory=4096Mi"
        self.ps_pod_priority = ""

        self.num_evaluators = 1
        self.evaluator_resource_request = "cpu=1,memory=4096Mi"
        self.evaluator_pod_priority = ""

        self.num_tf_master = 3
        self.tf_master_resource_request = "cpu=1,memory=4096Mi"
        self.tf_master_pod_priority = ""


class NodeStatusFlowTest(unittest.TestCase):
    def test_get_node_state_flow(self):
        flow: NodeStateFlow = get_node_state_flow(
            NodeStatus.PENDING, NodeEventType.MODIFIED, NodeStatus.RUNNING
        )
        self.assertEqual(flow, NODE_STATE_FLOWS[2])

        flow = get_node_state_flow(
            NodeStatus.RUNNING, NodeEventType.MODIFIED, NodeStatus.SUCCEEDED
        )
        self.assertEqual(flow, NODE_STATE_FLOWS[5])

        flow = get_node_state_flow(
            NodeStatus.RUNNING, NodeEventType.DELETED, NodeStatus.DELETED
        )
        self.assertEqual(flow, NODE_STATE_FLOWS[8])
        self.assertTrue(flow.should_relaunch)

        flow = get_node_state_flow(
            NodeStatus.SUCCEEDED, NodeEventType.DELETED, NodeStatus.DELETED
        )
        self.assertEqual(flow, NODE_STATE_FLOWS[-2])
        self.assertFalse(flow.should_relaunch)


class JobConfigTest(unittest.TestCase):
    def test_job_resource(self):
        job = JobResourceConfig()
        job.add_node_group_resource(NodeType.PS, 3, "cpu=1,memory=4096Mi", "")
        job.add_node_group_resource(
            NodeType.WORKER, 5, "cpu=1,memory=4096Mi", ""
        )
        group_resource = job.get_node_group_resource(NodeType.WORKER)
        self.assertEqual(group_resource.count, 5)
        self.assertEqual(group_resource.node_resource.cpu, 1)
        self.assertEqual(group_resource.node_resource.memory, 4096)
        group_resource = job.get_node_group_resource(NodeType.PS)
        self.assertEqual(group_resource.count, 3)
        self.assertEqual(group_resource.node_resource.cpu, 1)
        self.assertEqual(group_resource.node_resource.memory, 4096)
        node_types = job.get_node_types()
        self.assertListEqual(node_types, [NodeType.PS, NodeType.WORKER])
        self.assertEqual(job.worker_num, 5)
        self.assertEqual(job.ps_num, 3)

        nodes = job.init_job_node_meta(1, get_service_fn)
        self.assertEqual(len(nodes[NodeType.WORKER]), 5)
        self.assertEqual(len(nodes[NodeType.PS]), 3)
        self.assertEqual(nodes[NodeType.PS][0].id, 0)
        self.assertEqual(nodes[NodeType.PS][0].type, NodeType.PS)
        self.assertEqual(nodes[NodeType.WORKER][2].id, 2)
        self.assertEqual(nodes[NodeType.WORKER][0].type, NodeType.WORKER)
        self.assertEqual(nodes[NodeType.WORKER][0].used_resource.cpu, 1)

    def test_set_critical_node(self):
        job = JobResourceConfig()
        job.add_node_group_resource(NodeType.PS, 3, "cpu=1,memory=4096Mi", "")
        job.add_node_group_resource(
            NodeType.WORKER, 5, "cpu=1,memory=4096Mi", ""
        )

        nodes = job.init_job_node_meta(1, get_service_fn)
        set_critical_node(
            nodes, critical_worker_index={0: 3}, ps_relaunch_max_num=2
        )
        self.assertTrue(nodes[NodeType.PS][0].critical)
        self.assertEqual(nodes[NodeType.PS][0].max_relaunch_count, 2)
        self.assertTrue(nodes[NodeType.WORKER][0].critical)
        self.assertEqual(nodes[NodeType.WORKER][0].max_relaunch_count, 3)
        self.assertTrue(nodes[NodeType.WORKER][0].critical)
        self.assertFalse(nodes[NodeType.WORKER][1].critical)

    def test_get_critical_worker_index(self):
        args = MockArgs()
        critical_worker = get_critical_worker_index(args)
        self.assertDictEqual(critical_worker, {0: 3})
        args.critical_worker_index = "default"
        critical_worker = get_critical_worker_index(args)
        self.assertDictEqual(critical_worker, {0: 1})
        args.critical_worker_index = "all"
        critical_worker = get_critical_worker_index(args)
        self.assertDictEqual(critical_worker, {0: 1, 1: 1, 2: 1})

    def test_create_node_manager(self):
        args = MockArgs()
        manager = create_node_manager(args)
        self.assertEqual(manager._ps_relaunch_max_num, 1)
        manager._k8s_client.get_job_uuid = get_job_uuid
        manager._node_watcher._list_job_pods = mock_list_job_pods
        manager.start()
        self.assertEqual(manager._job_uuid, _MOCK_JOB_UUID)
        self.assertEqual(len(manager._job_nodes), 4)
        self.assertTrue(manager._job_nodes[NodeType.PS][0].critical)

        node = Node(
            node_type=NodeType.WORKER,
            node_id=1,
            status=NodeStatus.RUNNING,
        )

        node_event: NodeEvent = NodeEvent(NodeEventType.MODIFIED, node)
        manager._process_event(node_event)
        self.assertEqual(
            manager._job_nodes[NodeType.WORKER][1].status, NodeStatus.RUNNING
        )

        should_relaunch = manager._should_relaunch(node, NODE_STATE_FLOWS[5])
        self.assertFalse(should_relaunch)

        should_relaunch = manager._should_relaunch(node, NODE_STATE_FLOWS[6])
        self.assertTrue(should_relaunch)

        node.relaunch_count = node.max_relaunch_count + 1
        should_relaunch = manager._should_relaunch(node, NODE_STATE_FLOWS[6])
        self.assertFalse(should_relaunch)

        node.exit_reason = NodeExitReason.FATAL_ERROR
        should_relaunch = manager._should_relaunch(node, NODE_STATE_FLOWS[6])
        self.assertFalse(should_relaunch)

    def test_recover_tasks_for_failed_workers(self):
        dataset_name = "test"
        task_manager = create_task_manager()
        task_callback = TaskRescheduleCallback(task_manager)
        args = MockArgs()
        manager = create_node_manager(args)
        manager._init_attr_for_typed_pods()
        manager.add_pod_event_callback(task_callback)

        dataset = task_manager.get_dataset(dataset_name)
        task_manager.get_dataset_task(0, dataset_name)
        node = Node(
            node_type=NodeType.WORKER,
            node_id=0,
            status=NodeStatus.RUNNING,
        )
        manager._process_node_events(NODE_STATE_FLOWS[7], node)
        self.assertEqual(len(dataset.doing), 0)