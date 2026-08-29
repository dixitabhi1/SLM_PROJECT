"""
LLM Baseline Runner
AI Search Framework
Executes direct monolithic LLM queries (no decomposition) with paired instrumentation logging.
"""

import time
from typing import Dict, Any, Optional
from ..models.base import BaseModelRunner, ModelResponse
from ..instrumentation.logger import ExperimentLogger

BASELINE_SYSTEM_PROMPT = """You are an advanced frontier-class AI assistant.
Answer the user's query comprehensively, with deep domain accuracy, mathematical rigor, fully functional and well-commented code, and clear structured reasoning. Address every constraint and facet of the query thoroughly.
"""

class BaselineRunner:
    def __init__(self, baseline_model_runner: BaseModelRunner, logger: Optional[ExperimentLogger] = None):
        self.runner = baseline_model_runner
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
        Executes monolithic baseline query and logs structured run record.
        """
        run_id = f"baseline_{query_id}_{seed}"
        
        # Initialize run record
        record = self.logger.create_run_record(
            run_id=run_id,
            system_type="llm_baseline",
            query_id=query_id,
            query_text=query_text,
            complexity_tier=complexity_tier,
            seed=seed,
            config=config
        ) if self.logger else {}

        start_total = time.perf_counter()
        
        # Dispatch baseline inference
        start_stage = time.perf_counter()
        resp: ModelResponse = await self.runner.generate(
            prompt=query_text,
            system_prompt=BASELINE_SYSTEM_PROMPT,
            temperature=0.0
        )
        end_stage = time.perf_counter()

        if self.logger:
            self.logger.record_stage(
                record=record,
                stage_name="baseline_monolithic_generation",
                model_name=self.runner.model_name,
                model_revision=self.runner.revision,
                start_time_s=start_stage,
                end_time_s=end_stage,
                prompt_tokens=resp.prompt_tokens,
                completion_tokens=resp.completion_tokens,
                input_data=query_text,
                output_data=resp.text,
                extra_metadata={"system": "monolithic_baseline"}
            )

        end_total = time.perf_counter()
        
        log_path = ""
        if self.logger:
            log_path = self.logger.finalize_run(record, start_total, end_total, resp.text)

        return {
            "query_id": query_id,
            "response": resp.text,
            "wall_clock_latency_ms": (end_total - start_total) * 1000.0,
            "prompt_tokens": resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens,
            "cost_usd": record.get("total_cost_usd", 0.0) if self.logger else 0.0,
            "log_path": log_path,
            "record": record
        }

