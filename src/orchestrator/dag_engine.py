"""
Async DAG Orchestrator
AI Search Framework
Dependency-aware task scheduling, confidence-based replication dispatch, and execution tracing.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional
from ..models.base import BaseModelRunner, ModelResponse
from ..router.capability_router import CapabilityRouter
from ..instrumentation.logger import ExperimentLogger

class DAGOrchestrator:
    def __init__(
        self,
        pool_runners: Dict[str, BaseModelRunner],
        router: CapabilityRouter,
        logger: Optional[ExperimentLogger] = None,
        parallelism_mode: str = "simulated"
    ):
        self.pool_runners = pool_runners
        self.router = router
        self.logger = logger
        self.parallelism_mode = parallelism_mode

    async def execute_dag(
        self,
        dag: Dict[str, Any],
        run_record: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Executes the DAG respecting dependencies, routing to specialized SLMs,
        and collecting intermediate outputs with full dispatch tracing.
        """
        subtasks = dag.get("subtasks", [])
        results: Dict[str, str] = {} # node_id -> output text
        node_map = {t["id"]: t for t in subtasks}
        completed_nodes = set()
        dispatch_trace = []

        total_simulated_parallel_ms = 0.0

        # Topological layer-by-layer async resolution
        while len(completed_nodes) < len(subtasks):
            # Find all nodes whose dependencies are completed
            ready_nodes = [
                t for t in subtasks
                if t["id"] not in completed_nodes and all(dep in completed_nodes for dep in t["dependencies"])
            ]

            if not ready_nodes:
                # Deadlock or unresolvable cycle safeguard: release first remaining node
                remaining = [t for t in subtasks if t["id"] not in completed_nodes]
                ready_nodes = [remaining[0]]

            # Schedule ready nodes concurrently
            tasks = []
            for node in ready_nodes:
                tasks.append(self._execute_single_subtask(node, results, run_record))

            layer_start = time.perf_counter()
            layer_outputs = await asyncio.gather(*tasks)
            layer_end = time.perf_counter()
            layer_wall_clock_ms = (layer_end - layer_start) * 1000.0

            # Calculate theoretical simulated parallel latency (max latency of layer nodes)
            max_simulated_node_latency = max((out["simulated_latency_ms"] for out in layer_outputs), default=0.0)
            total_simulated_parallel_ms += max_simulated_node_latency

            for out in layer_outputs:
                nid = out["node_id"]
                results[nid] = out["text"]
                completed_nodes.add(nid)
                dispatch_trace.append(out["trace_entry"])

        run_record["dispatch_trace"] = dispatch_trace
        run_record["simulated_parallel_latency_ms"] = total_simulated_parallel_ms

        return {
            "subtask_results": results,
            "dispatch_trace": dispatch_trace,
            "simulated_parallel_latency_ms": total_simulated_parallel_ms
        }

    async def _execute_single_subtask(
        self,
        node: Dict[str, Any],
        completed_results: Dict[str, str],
        run_record: Dict[str, Any]
    ) -> Dict[str, Any]:
        node_id = node["id"]
        node_text = node["text"]
        explicit_cap = node.get("capability", "general")

        # 1. Route subtask
        route_info = self.router.route(node_text, explicit_cap)
        primary_cap = route_info["primary_capability"]
        confidence = route_info["confidence"]
        replicate = route_info["replicate"]
        sec_cap = route_info.get("secondary_capability")

        # 2. Build contextual prompt incorporating dependency results
        context_blocks = []
        for dep_id in node.get("dependencies", []):
            if dep_id in completed_results:
                context_blocks.append(f"--- Context from prerequisite subtask ({dep_id}) ---\n{completed_results[dep_id]}")
        
        context_str = "\n\n".join(context_blocks)
        full_subtask_prompt = f"{context_str}\n\nTask: {node_text}" if context_str else f"Task: {node_text}"

        # 3. Select runner
        primary_runner = self.pool_runners.get(primary_cap, self.pool_runners.get("general", list(self.pool_runners.values())[0]))

        # 4. Dispatch primary execution
        start_t = time.perf_counter()
        resp: ModelResponse = await primary_runner.generate(
            prompt=full_subtask_prompt,
            system_prompt=f"You are a specialized {primary_cap} SLM. Solve the given subtask rigorously."
        )
        end_t = time.perf_counter()

        # Log primary stage
        if self.logger:
            self.logger.record_stage(
                record=run_record,
                stage_name=f"pool_{primary_cap}_{node_id}",
                model_name=primary_runner.model_name,
                model_revision=primary_runner.revision,
                start_time_s=start_t,
                end_time_s=end_t,
                prompt_tokens=resp.prompt_tokens,
                completion_tokens=resp.completion_tokens,
                input_data=full_subtask_prompt,
                output_data=resp.text,
                confidence_score=confidence,
                extra_metadata={"node_id": node_id, "capability": primary_cap, "replicated": False}
            )

        final_node_text = resp.text
        simulated_lat_ms = resp.metadata.get("simulated_ideal_latency_ms", resp.latency_ms)

        # 5. Handle replication if triggered
        sec_resp = None
        if replicate and sec_cap and sec_cap in self.pool_runners:
            sec_runner = self.pool_runners[sec_cap]
            s_start = time.perf_counter()
            sec_resp = await sec_runner.generate(
                prompt=full_subtask_prompt,
                system_prompt=f"You are a specialized {sec_cap} SLM providing verified secondary execution."
            )
            s_end = time.perf_counter()

            if self.logger:
                self.logger.record_stage(
                    record=run_record,
                    stage_name=f"pool_replication_{sec_cap}_{node_id}",
                    model_name=sec_runner.model_name,
                    model_revision=sec_runner.revision,
                    start_time_s=s_start,
                    end_time_s=s_end,
                    prompt_tokens=sec_resp.prompt_tokens,
                    completion_tokens=sec_resp.completion_tokens,
                    input_data=full_subtask_prompt,
                    output_data=sec_resp.text,
                    confidence_score=route_info.get("secondary_confidence"),
                    extra_metadata={"node_id": node_id, "capability": sec_cap, "replicated": True}
                )
            
            # Combine primary and replicated output for maximum coverage
            final_node_text = f"{resp.text}\n\n[Replication Synthesis ({sec_cap})]:\n{sec_resp.text}"
            simulated_lat_ms = max(simulated_lat_ms, sec_resp.metadata.get("simulated_ideal_latency_ms", sec_resp.latency_ms))

        trace_entry = {
            "node_id": node_id,
            "capability": primary_cap,
            "confidence": confidence,
            "replicated": replicate,
            "secondary_capability": sec_cap if replicate else None,
            "dependencies": node.get("dependencies", []),
            "prompt_tokens": resp.prompt_tokens + (sec_resp.prompt_tokens if sec_resp else 0),
            "completion_tokens": resp.completion_tokens + (sec_resp.completion_tokens if sec_resp else 0),
            "latency_ms": resp.latency_ms
        }

        return {
            "node_id": node_id,
            "text": final_node_text,
            "simulated_latency_ms": simulated_lat_ms,
            "trace_entry": trace_entry
        }

