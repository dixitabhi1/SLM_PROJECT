"""
All-SLM Pipeline v5 Runner (LLM Council Architecture)
AI Search Framework - Version 5
Implements:
1. Multi-Domain Deterministic Reference Grounding for Math, Science, Structured Data, and Retrieval (Solution 2)
2. Grounded & Mechanically Verified Coding Specialist with Sandboxed AST Execution (Solution 3)
3. Zero-Drift Deterministic Template Aggregator with SLM Executive Overview (Solution 1)
4. Calibrated Multi-Node Decomposer with Standing Hard Rule 15 Non-Collapse Assertion
5. Zero LLMs in the architecture (all models <=5B)
"""

import time
from typing import Dict, List, Any, Optional
from ..models.base import BaseModelRunner
from ..instrumentation.logger import ExperimentLogger
from ..v3.decomposer.decomposer import DecomposerSLM_v3
from ..v2.analyser.agent_analyser import AgentAnalyserSLM
from ..v2.colorer.agent_colorer import AgentColorerSLM
from ..v2.scheduling.scheduling_slm import SchedulingSLM
from .aggregator.template_aggregator import DeterministicTemplateAggregator
from ..v3.analyser.task_analyser import TaskAnalyserSLM_v3
from ..v3.colorer.task_colorer import TaskColorerSLM_v3
from ..v3.matching.matching_slm import MatchingSLM_v3
from .tools.engineering_retrieval_tool import EngineeringRetrievalTool
from .tools.coding_retrieval_tool import CodingRetrievalTool
from .tools.science_tech_retrieval_tool import ScienceTechRetrievalTool
from .specialists.grounded_engineering_runner import GroundedEngineeringModelRunner
from .specialists.grounded_science_runner import GroundedScienceTechModelRunner
from .specialists.grounded_verified_coding_runner import GroundedVerifiedCodingModelRunner
from ..v4.tools.code_verifier import MechanicalCodeVerifier

class SLMPipeline_v5:
    def __init__(
        self,
        decomposer_runner: BaseModelRunner,
        base_pool_runners: Dict[str, BaseModelRunner],
        aggregator_runner: BaseModelRunner,
        retrieval_tool: Optional[EngineeringRetrievalTool] = None,
        coding_retrieval_tool: Optional[CodingRetrievalTool] = None,
        science_retrieval_tool: Optional[ScienceTechRetrievalTool] = None,
        code_verifier: Optional[MechanicalCodeVerifier] = None,
        coding_max_retries: int = 3,
        logger: Optional[ExperimentLogger] = None,
        max_depth: int = 3,
        max_concurrent_slms: int = 4
    ):
        self.retrieval_tool = retrieval_tool or EngineeringRetrievalTool()
        self.coding_retrieval_tool = coding_retrieval_tool or CodingRetrievalTool()
        self.science_retrieval_tool = science_retrieval_tool or ScienceTechRetrievalTool()
        self.code_verifier = code_verifier or MechanicalCodeVerifier()
        self.coding_max_retries = coding_max_retries
        self.decomposer_runner = decomposer_runner
        self.aggregator_runner = aggregator_runner

        # Build grounded and verified specialist pool
        self.pool_runners = {}
        for domain, runner in base_pool_runners.items():
            if domain == "science_tech":
                self.pool_runners[domain] = GroundedScienceTechModelRunner(
                    base_runner=runner,
                    retrieval_tool=self.science_retrieval_tool
                )
            elif domain == "coding":
                self.pool_runners[domain] = GroundedVerifiedCodingModelRunner(
                    base_runner=runner,
                    retrieval_tool=self.coding_retrieval_tool,
                    verifier=self.code_verifier,
                    max_retries=self.coding_max_retries
                )
            elif domain in ["mathematics", "structured_data", "retrieval_qa"]:
                self.pool_runners[domain] = GroundedEngineeringModelRunner(
                    base_runner=runner,
                    retrieval_tool=self.retrieval_tool
                )
            else:
                self.pool_runners[domain] = runner

        self.decomposer = DecomposerSLM_v3(decomposer_runner)
        self.task_analyser = TaskAnalyserSLM_v3()
        self.agent_analyser = AgentAnalyserSLM()
        self.task_colorer = TaskColorerSLM_v3()
        self.agent_colorer = AgentColorerSLM()
        self.matching = MatchingSLM_v3(max_depth=max_depth)
        self.scheduling = SchedulingSLM(pool_runners=self.pool_runners, logger=logger, max_concurrent_slms=max_concurrent_slms)
        self.aggregator = DeterministicTemplateAggregator(aggregator_runner, logger=logger)
        self.logger = logger
        self.max_depth = max_depth

    async def execute_query(
        self,
        query_id: str,
        query_text: str,
        complexity_tier: str,
        seed: int,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Executes full v5 pipeline across 8 specialist domains with grounded tools and template assembly.
        """
        run_id = f"slm_pipe_v5_{query_id}_{seed}"
        
        record = self.logger.create_run_record(
            run_id=run_id,
            system_type="slm_pipeline_v5",
            query_id=query_id,
            query_text=query_text,
            complexity_tier=complexity_tier,
            seed=seed,
            config=config
        ) if self.logger else {}

        record["loop_events"] = []
        record["v3_task_skill_vectors"] = {}
        record["v3_task_colors"] = {}

        start_total = time.perf_counter()

        # --- Stage 1: Decomposer Execution ---
        d_start = time.perf_counter()
        init_res = await self.decomposer.decompose_initial(query_text)
        d_end = time.perf_counter()
        
        active_tasks = init_res["subtasks"]
        d_resp = init_res["model_response"]

        # --- Hard Rule 15 Standing Automated Pre-Flight Assertion ---
        if complexity_tier in ["three_plus_domain", "compound_dag", "compound"]:
            assert len(active_tasks) >= 2, (
                f"[HARD RULE 15 VIOLATION] Known compound query '{query_id}' collapsed into {len(active_tasks)} node(s). "
                f"Decomposer must emit >= 2 distinct subtask nodes for compound queries! Subtasks: {active_tasks}"
            )

        if self.logger:
            self.logger.record_stage(
                record=record,
                stage_name="decomposition_initial",
                model_name=d_resp.model_name,
                model_revision=d_resp.model_revision,
                start_time_s=d_start,
                end_time_s=d_end,
                prompt_tokens=d_resp.prompt_tokens,
                completion_tokens=d_resp.completion_tokens,
                input_data=query_text,
                output_data={"initial_tasks": active_tasks},
                extra_metadata={"depth": 0}
            )

        # --- Convergence with Bounded Feedback Loop ---
        final_matched_tasks = []
        pending_queue = list(active_tasks)
        agent_profiles = self.agent_analyser.get_all_agent_profiles()

        loop_iterations = 0

        while pending_queue:
            task = pending_queue.pop(0)
            task_id = task["id"]
            current_depth = task.get("depth", 0)

            # SLM-2: 8D Skill Vector
            skill_vector = self.task_analyser.analyse_skill_vector(task["text"], task.get("capability", ""))
            record["v3_task_skill_vectors"][task_id] = skill_vector

            # SLM-3: 8-Color Task Colorer
            color_info = self.task_colorer.color_task(skill_vector)
            record["v3_task_colors"][task_id] = color_info

            # Matching SLM
            match_res = self.matching.match_task_and_evaluate_loop(task, color_info, agent_profiles)

            if match_res["action"] == "LOOP_BACK_TO_DECOMPOSER":
                loop_iterations += 1
                loop_event = {
                    "event_index": loop_iterations,
                    "parent_task_id": task_id,
                    "recursion_depth": current_depth,
                    "active_colors": color_info["active_colors"],
                    "skill_vector": skill_vector,
                    "reason": "multi_color_subtask_trigger"
                }
                record["loop_events"].append(loop_event)

                # Re-decompose task into child subtasks
                rd_start = time.perf_counter()
                re_decomp = await self.decomposer.re_decompose_task(task, next_depth=current_depth + 1)
                rd_end = time.perf_counter()
                
                rd_resp = re_decomp["model_response"]
                if self.logger:
                    self.logger.record_stage(
                        record=record,
                        stage_name=f"decomposition_loop_{loop_iterations}",
                        model_name=rd_resp.model_name,
                        model_revision=rd_resp.model_revision,
                        start_time_s=rd_start,
                        end_time_s=rd_end,
                        prompt_tokens=rd_resp.prompt_tokens,
                        completion_tokens=rd_resp.completion_tokens,
                        input_data=task,
                        output_data={"child_subtasks": re_decomp["child_subtasks"]},
                        extra_metadata={"depth": current_depth + 1, "parent_task_id": task_id}
                    )

                for child in re_decomp["child_subtasks"]:
                    pending_queue.append(child)
            else:
                final_matched_tasks.append({
                    "task": task,
                    "match_info": match_res,
                    "color_info": color_info,
                    "skill_vector": skill_vector
                })

        # --- Stage 2 & 3: Scheduling & Specialist Execution ---
        schedule_layers = self.scheduling.build_execution_schedule(final_matched_tasks)
        exec_res = await self.scheduling.execute_schedule(schedule_layers, record)
        subtask_results = exec_res["subtask_results"]

        # --- Stage 4: Deterministic Template Aggregation ---
        final_answer = await self.aggregator.aggregate_global(
            original_query=query_text,
            subtask_results=subtask_results,
            run_record=record
        )

        end_total = time.perf_counter()

        log_path = ""
        if self.logger:
            log_path = self.logger.finalize_run(record, start_total, end_total, final_answer)

        return {
            "query_id": query_id,
            "response": final_answer,
            "matched_tasks_count": len(final_matched_tasks),
            "feedback_loop_fired": len(record["loop_events"]) > 0,
            "loop_events_count": len(record["loop_events"]),
            "wall_clock_latency_ms": (end_total - start_total) * 1000.0,
            "log_path": log_path,
            "record": record
        }

