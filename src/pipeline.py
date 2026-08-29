"""
All-SLM Pipeline Runner
AI Search Framework
Orchestrates: Decomposer (<=3B) -> Router (Rule/Embedding) -> Pool SLMs (<=8B) -> Aggregator (<=8B)
"""

import time
from typing import Dict, Any, Optional
from .models.base import BaseModelRunner
from .decomposer.decomposer import TaskGraphDecomposer
from .router.capability_router import CapabilityRouter
from .orchestrator.dag_engine import DAGOrchestrator
from .aggregator.aggregator import ResponseAggregator
from .instrumentation.logger import ExperimentLogger

class SLMPipeline:
    def __init__(
        self,
        decomposer_runner: BaseModelRunner,
        pool_runners: Dict[str, BaseModelRunner],
        aggregator_runner: BaseModelRunner,
        router: CapabilityRouter,
        logger: Optional[ExperimentLogger] = None,
        parallelism_mode: str = "simulated"
    ):
        self.decomposer = TaskGraphDecomposer(decomposer_runner)
        self.router = router
        self.aggregator = ResponseAggregator(aggregator_runner, logger=logger)
        self.orchestrator = DAGOrchestrator(
            pool_runners=pool_runners,
            router=router,
            logger=logger,
            parallelism_mode=parallelism_mode
        )
        self.logger = logger

    async def execute_query(
        self,
        query_id: str,
        query_text: str,
        complexity_tier: str,
        seed: int,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Executes end-to-end all-SLM decomposed search pipeline with full stage logging.
        """
        run_id = f"slm_pipe_{query_id}_{seed}"
        
        record = self.logger.create_run_record(
            run_id=run_id,
            system_type="slm_pipeline",
            query_id=query_id,
            query_text=query_text,
            complexity_tier=complexity_tier,
            seed=seed,
            config=config
        ) if self.logger else {}

        start_total = time.perf_counter()

        # --- Stage 1: Decomposition ---
        d_start = time.perf_counter()
        decomp_result = await self.decomposer.decompose(query_text)
        d_end = time.perf_counter()

        dag = decomp_result["dag"]
        d_model_resp = decomp_result["model_response"]

        if self.logger:
            self.logger.record_stage(
                record=record,
                stage_name="decomposition_slm",
                model_name=d_model_resp.model_name,
                model_revision=d_model_resp.model_revision,
                start_time_s=d_start,
                end_time_s=d_end,
                prompt_tokens=d_model_resp.prompt_tokens,
                completion_tokens=d_model_resp.completion_tokens,
                input_data=query_text,
                output_data=dag,
                extra_metadata={"is_schema_valid": decomp_result["is_schema_valid"], "parse_error": decomp_result["parse_error"]}
            )

        # --- Stage 2 & 3: Routing & Orchestrated Pool Execution ---
        orch_result = await self.orchestrator.execute_dag(dag, record)
        subtask_results = orch_result["subtask_results"]

        # --- Stage 4: Aggregation & Contradiction Resolution ---
        final_answer = await self.aggregator.aggregate(
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
            "dag": dag,
            "wall_clock_latency_ms": (end_total - start_total) * 1000.0,
            "simulated_parallel_latency_ms": orch_result["simulated_parallel_latency_ms"] + d_model_resp.latency_ms,
            "total_tokens": record.get("total_tokens", 0) if self.logger else 0,
            "cost_usd": record.get("total_cost_usd", 0.0) if self.logger else 0.0,
            "log_path": log_path,
            "record": record
        }

