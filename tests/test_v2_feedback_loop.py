"""
Unit Tests for v2 Feedback Loop and Architecture
Tests:
1. Single-color direct forward
2. Multi-color loop-back trigger
3. Multi-color at depth limit (depth=3) multi-agent collaboration
4. End-to-end v2 pipeline run with loop event tracing
"""

import pytest
import asyncio
from src.models.mock_runner import MockModelRunner
from src.instrumentation.logger import ExperimentLogger
from src.v2.analyser.task_analyser import TaskAnalyserSLM
from src.v2.colorer.task_colorer import TaskColorerSLM
from src.v2.matching.matching_slm import MatchingSLM
from src.v2.pipeline import SLMPipeline_v2

def test_task_analyser_and_colorer():
    analyser = TaskAnalyserSLM()
    colorer = TaskColorerSLM()

    # Pure coding task
    s_code = analyser.analyse_skill_vector("Write an asynchronous Python function implementing LRU cache.")
    assert s_code["coding"] > 0.40
    c_code = colorer.color_task(s_code)
    assert c_code["dominant_color"] == "blue"
    assert c_code["spans_multiple_colors"] is False

    # Compound Code + Math task
    s_compound = analyser.analyse_skill_vector("Derive the Kalman filter state update equations and implement a vectorized Python simulation.")
    assert s_compound["math"] > 0.20
    assert s_compound["coding"] > 0.20
    c_compound = colorer.color_task(s_compound)
    assert c_compound["spans_multiple_colors"] is True

def test_matching_loop_decision_logic():
    matching = MatchingSLM(max_depth=3)
    agent_profiles = {}

    # Case 1: Single-color task -> Forward
    single_color_info = {"spans_multiple_colors": False, "dominant_color": "blue", "dominant_domain": "coding", "active_domains": ["coding"], "active_colors": ["blue"]}
    res1 = matching.match_task_and_evaluate_loop({"id": "node_1", "depth": 0, "dependencies": []}, single_color_info, agent_profiles)
    assert res1["action"] == "FORWARD_TO_SCHEDULING"
    assert res1["collaboration_mode"] is False
    assert res1["assigned_agent"] == "coding"

    # Case 2: Multi-color task at Depth 0 -> Loop-back to Decomposer
    multi_color_info = {"spans_multiple_colors": True, "dominant_color": "green", "dominant_domain": "math", "active_domains": ["math", "coding"], "active_colors": ["green", "blue"]}
    res2 = matching.match_task_and_evaluate_loop({"id": "node_1", "depth": 0, "dependencies": []}, multi_color_info, agent_profiles)
    assert res2["action"] == "LOOP_BACK_TO_DECOMPOSER"
    assert res2["depth_limit_reached"] is False

    # Case 3: Multi-color task at Depth 3 (Limit Reached) -> Forward with Multi-Agent Collaboration
    res3 = matching.match_task_and_evaluate_loop({"id": "node_1.1.1.1", "depth": 3, "dependencies": []}, multi_color_info, agent_profiles)
    assert res3["action"] == "FORWARD_TO_SCHEDULING"
    assert res3["depth_limit_reached"] is True
    assert res3["collaboration_mode"] is True
    assert set(res3["assigned_team"]) == {"math", "coding"}

def test_v2_pipeline_e2e_run():
    async def _test():
        logger = ExperimentLogger(pricing_table_path="config/pricing_table.json", log_dir="logs/test_runs")
        decomposer_runner = MockModelRunner("meta-llama/Llama-3.2-3B-Instruct")
        pool_runners = {
            "coding": MockModelRunner("Qwen/Qwen2.5-Coder-7B-Instruct"),
            "math": MockModelRunner("Qwen/Qwen2.5-Math-7B-Instruct"),
            "reasoning": MockModelRunner("microsoft/Phi-3.5-mini-instruct"),
            "retrieval": MockModelRunner("meta-llama/Llama-3.2-3B-Instruct"),
            "general": MockModelRunner("meta-llama/Llama-3.1-8B-Instruct")
        }
        aggregator_runner = MockModelRunner("meta-llama/Llama-3.1-8B-Instruct")

        pipe_v2 = SLMPipeline_v2(
            decomposer_runner=decomposer_runner,
            pool_runners=pool_runners,
            aggregator_runner=aggregator_runner,
            logger=logger,
            max_depth=3,
            max_concurrent_slms=4
        )

        res = await pipe_v2.execute_query(
            query_id="TD_CM_01",
            query_text="Derive the Kalman filter mathematical equations and implement a vectorized Python simulation.",
            complexity_tier="two_domain",
            seed=42,
            config={"test_v2": True}
        )

        assert res["query_id"] == "TD_CM_01"
        assert len(res["response"]) > 0
        assert res["matched_tasks_count"] >= 1
        assert "record" in res
        assert "v2_task_skill_vectors" in res["record"]

    asyncio.run(_test())

