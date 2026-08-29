"""
End-to-End Pipeline and Baseline Integration Tests
"""

import asyncio
import os
import json
from src.models.mock_runner import MockModelRunner
from src.router.capability_router import CapabilityRouter
from src.instrumentation.logger import ExperimentLogger
from src.pipeline import SLMPipeline
from src.baseline.runner import BaselineRunner

def test_slm_pipeline_e2e_run():
    async def _test():
        logger = ExperimentLogger(pricing_table_path="config/pricing_table.json", log_dir="logs/test_runs")
        router = CapabilityRouter()
        
        decomposer_runner = MockModelRunner("meta-llama/Llama-3.2-3B-Instruct")
        pool_runners = {
            "coding": MockModelRunner("Qwen/Qwen2.5-Coder-7B-Instruct"),
            "math": MockModelRunner("Qwen/Qwen2.5-Math-7B-Instruct"),
            "reasoning": MockModelRunner("microsoft/Phi-3.5-mini-instruct"),
            "retrieval": MockModelRunner("meta-llama/Llama-3.2-3B-Instruct"),
            "general": MockModelRunner("meta-llama/Llama-3.1-8B-Instruct")
        }
        aggregator_runner = MockModelRunner("meta-llama/Llama-3.1-8B-Instruct")

        pipeline = SLMPipeline(
            decomposer_runner=decomposer_runner,
            pool_runners=pool_runners,
            aggregator_runner=aggregator_runner,
            router=router,
            logger=logger,
            parallelism_mode="simulated"
        )

        query_id = "TD_CM_01"
        query_text = "Derive the Kalman Filter state update equations and implement a vectorized Python class."
        
        res = await pipeline.execute_query(
            query_id=query_id,
            query_text=query_text,
            complexity_tier="two_domain",
            seed=42,
            config={"test": True}
        )

        assert res["query_id"] == query_id
        assert len(res["response"]) > 0
        assert res["wall_clock_latency_ms"] > 0
        assert res["cost_usd"] > 0
        assert os.path.exists(res["log_path"])

        with open(res["log_path"], "r") as f:
            log_data = json.load(f)
        assert log_data["system_type"] == "slm_pipeline"
        assert "decomposition_slm" in log_data["stages"]
        assert "aggregator_synthesis" in log_data["stages"]

    asyncio.run(_test())

def test_baseline_runner_e2e_run():
    async def _test():
        logger = ExperimentLogger(pricing_table_path="config/pricing_table.json", log_dir="logs/test_runs")
        baseline_runner = MockModelRunner("meta-llama/Llama-3.1-70B-Instruct")
        runner = BaselineRunner(baseline_runner, logger=logger)

        query_id = "TD_CM_01"
        query_text = "Derive the Kalman Filter state update equations and implement a vectorized Python class."

        res = await runner.execute_query(
            query_id=query_id,
            query_text=query_text,
            complexity_tier="two_domain",
            seed=42,
            config={"test": True}
        )

        assert res["query_id"] == query_id
        assert len(res["response"]) > 0
        assert res["cost_usd"] > 0
        assert os.path.exists(res["log_path"])

        with open(res["log_path"], "r") as f:
            log_data = json.load(f)
        assert log_data["system_type"] == "llm_baseline"
        assert "baseline_monolithic_generation" in log_data["stages"]

    asyncio.run(_test())

