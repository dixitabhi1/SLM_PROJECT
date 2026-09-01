"""
Scheduling SLM & Execution Engine (v2 Architecture)
Builds topological execution graphs, budgets concurrency, and manages multi-agent collaboration.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional
from ...models.base import BaseModelRunner, ModelResponse
from ...instrumentation.logger import ExperimentLogger

class SchedulingSLM:
    def __init__(
        self,
        pool_runners: Dict[str, BaseModelRunner],
        logger: Optional[ExperimentLogger] = None,
        max_concurrent_slms: int = 4
    ):
        self.pool_runners = pool_runners
        self.logger = logger
        self.max_concurrent_slms = max_concurrent_slms

    def build_execution_schedule(self, matched_tasks: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Builds topological parallel execution batches respecting dependencies and concurrency cap.
        """
        task_map = {t["task"]["id"]: t for t in matched_tasks}
        completed_ids = set()
        layers = []

        all_ids = set(task_map.keys())

        while len(completed_ids) < len(all_ids):
            ready = []
            for tid, entry in task_map.items():
                if tid in completed_ids:
                    continue
                deps = entry["task"].get("dependencies", [])
                # Ready if all dependencies are already completed or not in the graph
                if all(d in completed_ids or d not in all_ids for d in deps):
                    ready.append(entry)

            if not ready:
                # Cycle fallback: pick first uncompleted
                remaining = [entry for tid, entry in task_map.items() if tid not in completed_ids]
                ready = [remaining[0]]

            # Split ready into chunks up to max_concurrent_slms
            for i in range(0, len(ready), self.max_concurrent_slms):
                chunk = ready[i:i + self.max_concurrent_slms]
                layers.append(chunk)
                for item in chunk:
                    completed_ids.add(item["task"]["id"])

        return layers

    async def execute_schedule(
        self,
        schedule_layers: List[List[Dict[str, Any]]],
        run_record: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Executes layers sequentially, nodes within layers concurrently.
        Supports single-agent dispatch and multi-agent collaboration with local synthesis.
        """
        completed_results: Dict[str, str] = {} # node_id -> text
        execution_trace = []
        total_simulated_parallel_ms = 0.0

        for layer_idx, layer in enumerate(schedule_layers):
            layer_tasks = []
            for item in layer:
                layer_tasks.append(self._execute_node(item, completed_results, run_record))

            layer_start = time.perf_counter()
            layer_outputs = await asyncio.gather(*layer_tasks)
            layer_end = time.perf_counter()

            max_sim_lat = max((out["simulated_latency_ms"] for out in layer_outputs), default=0.0)
            total_simulated_parallel_ms += max_sim_lat

            for out in layer_outputs:
                nid = out["node_id"]
                completed_results[nid] = out["text"]
                execution_trace.append(out["trace_entry"])

        run_record["dispatch_trace"] = execution_trace
        run_record["simulated_parallel_latency_ms"] = total_simulated_parallel_ms

        return {
            "subtask_results": completed_results,
            "execution_trace": execution_trace,
            "simulated_parallel_latency_ms": total_simulated_parallel_ms
        }

    async def _execute_node(
        self,
        item: Dict[str, Any],
        completed_results: Dict[str, str],
        run_record: Dict[str, Any]
    ) -> Dict[str, Any]:
        task = item["task"]
        match_info = item["match_info"]
        node_id = task["id"]
        task_text = task["text"]
        is_collab = match_info.get("collaboration_mode", False)
        assigned_team = match_info.get("assigned_team", ["general"])

        # Build context from dependencies
        context_parts = []
        for dep in task.get("dependencies", []):
            if dep in completed_results:
                context_parts.append(f"[Prerequisite {dep}]:\n{completed_results[dep]}")
        context_str = "\n\n".join(context_parts)
        node_prompt = f"{context_str}\n\nTask: {task_text}" if context_str else f"Task: {task_text}"

        if not is_collab:
            # Single Agent Execution
            domain = match_info.get("assigned_agent", "general")
            runner = self.pool_runners.get(domain, self.pool_runners.get("general", list(self.pool_runners.values())[0]))
            
            s_start = time.perf_counter()
            resp: ModelResponse = await runner.generate(
                prompt=node_prompt,
                system_prompt=f"You are a specialized {domain} SLM. Solve the subtask completely."
            )
            s_end = time.perf_counter()

            if self.logger:
                self.logger.record_stage(
                    record=run_record,
                    stage_name=f"pool_{domain}_{node_id}",
                    model_name=runner.model_name,
                    model_revision=runner.revision,
                    start_time_s=s_start,
                    end_time_s=s_end,
                    prompt_tokens=resp.prompt_tokens,
                    completion_tokens=resp.completion_tokens,
                    input_data=node_prompt,
                    output_data=resp.text,
                    extra_metadata={"node_id": node_id, "mode": "single_agent", "depth": task.get("depth", 0)}
                )

            final_text = resp.text
            sim_lat = resp.metadata.get("simulated_ideal_latency_ms", resp.latency_ms)

        else:
            # Multi-Agent Collaboration Team (Depth Limit Reached on Multi-Color Task)
            collab_tasks = []
            team_runners = []
            for domain in assigned_team:
                r = self.pool_runners.get(domain, self.pool_runners.get("general"))
                if r:
                    team_runners.append((domain, r))
                    collab_tasks.append(r.generate(
                        prompt=node_prompt,
                        system_prompt=f"You are the {domain} expert in a multi-agent team collaborating on a complex subtask."
                    ))

            c_start = time.perf_counter()
            collab_resps = await asyncio.gather(*collab_tasks)
            c_end = time.perf_counter()

            # Stage 1: Local Collaboration Synthesis Pass
            local_syntheses = []
            for (dom, r), c_resp in zip(team_runners, collab_resps):
                local_syntheses.append(f"[{dom.upper()} Specialist Analysis]:\n{c_resp.text}")
                if self.logger:
                    self.logger.record_stage(
                        record=run_record,
                        stage_name=f"pool_collab_{dom}_{node_id}",
                        model_name=r.model_name,
                        model_revision=r.revision,
                        start_time_s=c_start,
                        end_time_s=c_end,
                        prompt_tokens=c_resp.prompt_tokens,
                        completion_tokens=c_resp.completion_tokens,
                        input_data=node_prompt,
                        output_data=c_resp.text,
                        extra_metadata={"node_id": node_id, "mode": "multi_agent_collab", "depth": task.get("depth", 0)}
                    )

            final_text = f"=== Multi-Agent Collaboration ({', '.join(assigned_team)}) ===\n" + "\n\n".join(local_syntheses)
            sim_lat = max((r.metadata.get("simulated_ideal_latency_ms", r.latency_ms) for r in collab_resps), default=0.0)

        trace_entry = {
            "node_id": node_id,
            "depth": task.get("depth", 0),
            "is_collaboration": is_collab,
            "team": assigned_team,
            "dependencies": task.get("dependencies", []),
            "latency_ms": (time.perf_counter() - s_start) * 1000.0 if not is_collab else (c_end - c_start) * 1000.0
        }

        return {
            "node_id": node_id,
            "text": final_text,
            "simulated_latency_ms": sim_lat,
            "trace_entry": trace_entry
        }

