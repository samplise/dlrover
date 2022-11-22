# Copyright 2022 The DLRover Authors. All rights reserved.
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

import threading
import time
from typing import Dict, List

from dlrover.python.common.constants import (
    DistributionStrategy,
    NodeEventType,
    NodeExitReason,
    NodeResourceBoundary,
    NodeStatus,
    NodeType,
)
from dlrover.python.common.log_utils import default_logger as logger
from dlrover.python.master.node_manager.event_callback import (
    ClusterContext,
    NodeEventCallback,
)
from dlrover.python.master.node_manager.job_config import (
    JobResourceConfig,
    get_critical_worker_index,
    set_critical_node,
)
from dlrover.python.master.node_manager.status_flow import (
    NodeStateFlow,
    get_node_state_flow,
)
from dlrover.python.master.node_watcher.base_watcher import Node, NodeEvent
from dlrover.python.master.node_watcher.pod_watcher import PodWatcher
from dlrover.python.scheduler.kubernetes import k8sClient

_MAX_POD_RELAUNCH_COUNT = 5


def create_node_manager(args):
    # relaunch on worker failure for PS or custom strategy
    if (
        args.distribution_strategy != DistributionStrategy.PARAMETER_SERVER
        and args.distribution_strategy != DistributionStrategy.CUSTOM
    ):
        args.relaunch_on_worker_failure = 0

    job_resource = JobResourceConfig()
    job_resource.add_node_group_resource(
        NodeType.WORKER,
        args.num_workers,
        args.worker_resource_request,
        args.worker_pod_priority,
    )

    job_resource.add_node_group_resource(
        NodeType.PS,
        args.num_ps_pods,
        args.ps_resource_request,
        args.ps_pod_priority,
    )

    # Keep the same as the worker.
    evaluator_pod_priority = (
        args.evaluator_pod_priority
        if args.evaluator_pod_priority == "low"
        else "high"
    )

    job_resource.add_node_group_resource(
        NodeType.EVALUATOR,
        args.num_evaluators,
        args.evaluator_resource_request,
        evaluator_pod_priority,
    )

    job_resource.add_node_group_resource(
        NodeType.TF_MASTER,
        args.num_tf_master,
        args.tf_master_resource_request,
        args.tf_master_pod_priority,
    )

    critical_worker_index = get_critical_worker_index(args)

    # Custom distribution strategy does not exit if there are pending pods
    wait_pending_relaunch = (
        args.distribution_strategy == DistributionStrategy.CUSTOM
    )

    return NodeManager(
        job_resource=job_resource,
        job_name=args.job_name,
        namespace=args.namespace,
        relaunch_on_worker_failure=args.relaunch_on_worker_failure,
        ps_is_critical=args.ps_is_critical,
        critical_worker_index=critical_worker_index,
        wait_pending_relaunch=wait_pending_relaunch,
        ps_relaunch_max_num=args.ps_relaunch_max_num,
        use_ddp=args.use_ddp,
    )


class NodeManager(object):
    def __init__(
        self,
        job_resource,
        job_name,
        namespace,
        relaunch_on_worker_failure=0,
        ps_is_critical=True,
        critical_worker_index={},
        wait_pending_relaunch=False,
        ps_relaunch_max_num=1,
        use_ddp=False,
    ):
        self._job_resource = job_resource
        self._relaunch_on_worker_failure = min(
            relaunch_on_worker_failure, _MAX_POD_RELAUNCH_COUNT
        )
        self._wait_pending_relaunch = wait_pending_relaunch
        self._start_launch_waiting_workers_time = time.time()
        self._stop_launch_worker_for_ps = False
        self._ps_is_critical = ps_is_critical
        self._critical_worker_index = critical_worker_index
        self._ps_relaunch_max_num = min(
            ps_relaunch_max_num, _MAX_POD_RELAUNCH_COUNT
        )
        self._use_ddp = use_ddp
        self._pod_event_callbacks: List[NodeEventCallback] = []
        self._chief_worker_started = False
        self._stop_process_event = False
        self._last_pod_stats = None
        self._training_dataset = None

        # Protects followed variables, which are accessed from event_cb.
        self._lock = threading.Lock()

        # TODO @workingloong bstract the k8sClient.
        self._k8s_client = k8sClient(namespace, job_name)
        self._job_nodes: Dict[str, Dict[int, Node]] = {}
        self._node_watcher = PodWatcher(job_name, namespace)

    def start(self):
        self._job_uuid = self._k8s_client.get_job_uuid()
        self._init_attr_for_typed_pods()
        threading.Thread(
            target=self._monitor_nodes, name="node_monitor", daemon=True
        ).start()

    def add_pod_event_callback(self, pod_event_callback):
        self._pod_event_callbacks.append(pod_event_callback)

    def set_training_dataset(self, dataset):
        if not self._training_dataset:
            self._training_dataset = dataset

    def _init_attr_for_typed_pods(self):
        self._job_nodes = self._job_resource.init_job_node_meta(
            self._relaunch_on_worker_failure,
            self._k8s_client.get_service_address,
        )

        # worker and eval ids for pods that should be created
        # after all ps are running.
        self._workers_waiting_ps_running = []

        # ps pod ids that are deleted and waiting for relaunch
        self._deleted_ps_pod_ids = []

        self._relaunch_pod = True
        self._pending_relaunch_count = 0

        set_critical_node(
            self._job_nodes,
            self._ps_is_critical,
            self._critical_worker_index,
            self._ps_relaunch_max_num,
        )

    def _monitor_nodes(self):
        while True:
            nodes = self._node_watcher.list()
            self._process_list_nodes(nodes)
            try:
                if self._stop_process_event:
                    logger.info("Stop processing node events")
                for event in self._node_watcher.watch():
                    try:
                        self._process_event(event)
                    except Exception as e:
                        logger.warning(e)
            except Exception as e:
                logger.warning(e)
                time.sleep(30)

    def _process_list_nodes(self, nodes: List[Node]):
        """Callback with pod list by the list api of k8s."""
        exist_pods: Dict[str, List[int]] = {}
        for node_type in self._job_nodes.keys():
            exist_pods[node_type] = []
        for node in nodes:
            exist_pods[node.type].append(node.id)
            if node.status == NodeStatus.DELETED:
                type = NodeEventType.DELETED
            else:
                type = NodeEventType.MODIFIED
            # Mock event to avoid missing events
            event = NodeEvent(type, node)
            self._process_event(event)

        for node_type in self._job_nodes.keys():
            for node_id, node in self._job_nodes[node_type].items():
                if (
                    node.status != NodeStatus.INITIAL
                    and not node.is_released
                    and node_id not in exist_pods[node_type]
                ):
                    logger.info(
                        "Node %s %s is deleted without the event",
                        node_type,
                        node_id,
                    )
                    node.is_released = True

    def _process_event(self, event: NodeEvent):
        node_type = event.node.type
        node_id = event.node.id
        cur_node = self._job_nodes[node_type][node_id]
        cur_node.update_info(
            name=event.node.name,
            start_time=event.node.start_time,
            create_time=event.node.create_time,
        )

        # For the given node id, check whether it meets
        # the state change condition
        new_status = event.node.status
        with self._lock:
            old_status = cur_node.status
            status_change_flow: NodeStateFlow = get_node_state_flow(
                old_status, event.event_type, new_status
            )
            cur_node.update_status(new_status)
            # If there is no matched state change, return directly
            # If the pod has been succeed, return directly
            if (
                status_change_flow is None
                or status_change_flow.from_status == NodeStatus.SUCCEEDED
            ):
                return

            # Update the pod status for pod_info
            new_status = status_change_flow.to_status
            cur_node.set_exit_reason(event.node.exit_reason)
            self._process_node_events(status_change_flow, cur_node)

            should_relaunch = self._should_relaunch(
                cur_node, status_change_flow
            )
            if should_relaunch and self._wait_pending_relaunch:
                self._pending_relaunch_count += 1

        logger.info(
            "%s status change: %s to %s, by evt_type %s, phase %s",
            cur_node.name,
            old_status,
            new_status,
            event.event_type,
            new_status,
        )

        if should_relaunch:
            self._relaunch_typed_pod(cur_node)

    def _process_node_events(
        self, status_change_flow: NodeStateFlow, node: Node
    ):
        cluster_context = ClusterContext(node_manager=self)
        if status_change_flow.to_status == NodeStatus.RUNNING:
            [
                callback.on_node_started(node, cluster_context)
                for callback in self._pod_event_callbacks
            ]
        elif status_change_flow.to_status == NodeStatus.SUCCEEDED:
            [
                callback.on_node_succeeded(node, cluster_context)
                for callback in self._pod_event_callbacks
            ]
        elif status_change_flow.to_status == NodeStatus.FAILED:
            [
                callback.on_node_failed(node, cluster_context)
                for callback in self._pod_event_callbacks
            ]
        elif (
            status_change_flow.from_status != NodeStatus.FAILED
            and status_change_flow.from_status != NodeStatus.SUCCEEDED
            and status_change_flow.to_status == NodeStatus.DELETED
        ):
            [
                callback.on_node_deleted(node, cluster_context)
                for callback in self._pod_event_callbacks
            ]

    def _should_relaunch(self, node: Node, status_change_flow: NodeStateFlow):
        should_relaunch = (
            status_change_flow.should_relaunch
            and self._relaunch_pod
            and node.relaunchable
        )
        if should_relaunch:
            if node.exit_reason == NodeExitReason.FATAL_ERROR:
                should_relaunch = False
            elif node.exit_reason == NodeExitReason.OOM:
                mem = node.used_resource.memory
                if mem > NodeResourceBoundary.MAX_MEMORY:
                    should_relaunch = False
                    logger.warn(
                        "The memory of worker %s is beyond the limit %s MB.",
                        mem,
                        NodeResourceBoundary.MAX_MEMORY,
                    )
                elif node.relaunch_count >= node.max_relaunch_count:
                    should_relaunch = False
                    logger.warn(
                        "The relaunched count %s is beyond the maximum %s.",
                        node.relaunch_count,
                        node.max_relaunch_count,
                    )
                else:
                    node.is_recovered_oom = True
            elif node.exit_reason != NodeExitReason.KILLED:
                if node.relaunch_count > node.max_relaunch_count:
                    logger.warning(
                        "The relaunch count for Error has been exhausted."
                    )
                    should_relaunch = False
        if should_relaunch:
            node.inc_relaunch_count()

        return should_relaunch

    def _relaunch_typed_pod(self, node: Node):
        logger.info("Relaunch the pod: {}".format(node.name))