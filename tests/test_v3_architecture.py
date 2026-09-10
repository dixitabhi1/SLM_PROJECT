"""
Unit Tests for v3 Architecture (8 Domains, <=5B Parameter Cap, Fixes 1-3 Active)
Tests:
1. 8D Task Analyser skill vector computation
2. 8-Color Task Colorer mapping & Fix 3-narrow synthesis exclusion
3. Matching SLM loop decision logic across 8 domains
4. End-to-end SLMPipeline_v3 execution with mock runners
"""

import pytest
import asyncio
from src.models.mock_runner import MockModelRunner
from src.instrumentation.logger import ExperimentLogger
from src.v3.analyser.task_analyser import TaskAnalyserSLM_v3, SKILL_CATEGORIES_V3
from src.v3.colorer.task_colorer import TaskColorerSLM_v3, COLOR_TAXONOMY_V3
from src.v3.matching.matching_slm import MatchingSLM_v3
from src.v3.pipeline import SLMPipeline_v3

def test_task_analyser_v3_8d_vector():
    analyser = TaskAnalyserSLM_v3()
    
    # 1. Coding query
    s_code = analyser.analyse_skill_vector("Implement an asynchronous LRU cache with full type annotations and unit tests in Python.")
    assert s_code["coding"] > 0.40
    assert abs(sum(s_code.values()) - 1.0) < 1e-3

    # 2. Science / Tech query
    s_sci = analyser.analyse_skill_vector("Explain quantum thermodynamics and semiconductor bandgap energy transitions.")
    assert s_sci["science_tech"] > 0.40

    # 3. Structured Data query
    s_data = analyser.analyse_skill_vector("Write an optimized SQL query with foreign key joins to aggregate relational customer schema.")
    assert s_data["structured_data"] > 0.40

    # 4. Systems / Ops query
    s_ops = analyser.analyse_skill_vector("Configure a bash shell script with Docker and Kubernetes socket networking.")
    assert s_ops["systems_ops"] > 0.40

def test_task_colorer_v3_8_colors_and_narrow_exclusion():
    colorer = TaskColorerSLM_v3(multi_color_threshold=0.20)
    
    # Verify taxonomy has exactly 8 domains
    assert len(COLOR_TAXONOMY_V3) == 8

    # Case 1: Single specialist domain (Coding -> Blue)
    s_code = {d: 0.05 for d in SKILL_CATEGORIES_V3}
    s_code["coding"] = 0.65
    c_code = colorer.color_task(s_code)
    assert c_code["dominant_color"] == "blue"
    assert c_code["dominant_domain"] == "coding"
    assert c_code["spans_multiple_colors"] is False

    # Case 2: Multi-specialist compound task (Coding + Mathematics -> Blue + Green)
    s_compound = {d: 0.05 for d in SKILL_CATEGORIES_V3}
    s_compound["coding"] = 0.45
    s_compound["mathematics"] = 0.40
    c_comp = colorer.color_task(s_compound)
    assert "blue" in c_comp["active_colors"]
    assert "green" in c_comp["active_colors"]
    assert c_comp["spans_multiple_colors"] is True

    # Case 3: Fix 3-narrow: Coding + Creative Synthesis bleed (Rose)
    # Coding = 0.60, Creative Synthesis = 0.30 (both > 0.20 threshold)
    s_bleed = {d: 0.02 for d in SKILL_CATEGORIES_V3}
    s_bleed["coding"] = 0.60
    s_bleed["creative_synthesis"] = 0.30
    c_bleed = colorer.color_task(s_bleed)
    assert "blue" in c_bleed["active_colors"]
    assert "rose" in c_bleed["active_colors"]
    # Rose must NOT trigger multi-color decomposition
    assert c_bleed["spans_multiple_colors"] is False

def test_matching_slm_v3_decisions():
    matching = MatchingSLM_v3(max_depth=3)
    agent_profiles = {}

    # Case 1: Single domain -> Forward to scheduling
    single_color = {
        "spans_multiple_colors": False,
        "dominant_color": "teal",
        "dominant_domain": "science_tech",
        "active_domains": ["science_tech"],
        "active_colors": ["teal"]
    }
    res1 = matching.match_task_and_evaluate_loop(
        {"id": "node_1", "depth": 0, "dependencies": []},
        single_color,
        agent_profiles
    )
    assert res1["action"] == "FORWARD_TO_SCHEDULING"
    assert res1["assigned_agent"] == "science_tech"
    assert res1["collaboration_mode"] is False

    # Case 2: Multi-color task at Depth 0 -> Loop-back to Decomposer
    multi_color = {
        "spans_multiple_colors": True,
        "dominant_color": "blue",
        "dominant_domain": "coding",
        "active_domains": ["coding", "mathematics"],
        "active_colors": ["blue", "green"]
    }
    res2 = matching.match_task_and_evaluate_loop(
        {"id": "node_1", "depth": 0, "dependencies": []},
        multi_color,
        agent_profiles
    )
    assert res2["action"] == "LOOP_BACK_TO_DECOMPOSER"

    # Case 3: Multi-color task at Depth 3 (Terminal limit) -> Multi-Agent Collaboration
    res3 = matching.match_task_and_evaluate_loop(
        {"id": "node_1.1.1.1", "depth": 3, "dependencies": []},
        multi_color,
        agent_profiles
    )
    assert res3["action"] == "FORWARD_TO_SCHEDULING"
    assert res3["collaboration_mode"] is True
    assert set(res3["assigned_team"]) == {"coding", "mathematics"}

def test_slm_pipeline_v3_e2e_run():
    async def _test():
        logger = ExperimentLogger(pricing_table_path="config/pricing_table.json", log_dir="logs/test_runs")
        decomposer_runner = MockModelRunner("meta-llama/Llama-3.2-3B-Instruct")
        
        # 8 Pool runners (all <=5B)
        pool_runners = {
            "coding": MockModelRunner("Qwen/Qwen2.5-Coder-3B-Instruct"),
            "mathematics": MockModelRunner("Qwen/Qwen2.5-Math-1.5B-Instruct"),
            "formal_reasoning": MockModelRunner("microsoft/Phi-3.5-mini-instruct"),
            "retrieval_qa": MockModelRunner("meta-llama/Llama-3.2-3B-Instruct"),
            "science_tech": MockModelRunner("Qwen/Qwen2.5-3B-Instruct"),
            "structured_data": MockModelRunner("Qwen/Qwen2.5-Coder-1.5B-Instruct"),
            "creative_synthesis": MockModelRunner("meta-llama/Llama-3.2-3B-Instruct"),
            "systems_ops": MockModelRunner("Qwen/Qwen2.5-Coder-1.5B-Instruct")
        }
        aggregator_runner = MockModelRunner("meta-llama/Llama-3.2-3B-Instruct")

        pipeline_v3 = SLMPipeline_v3(
            decomposer_runner=decomposer_runner,
            pool_runners=pool_runners,
            aggregator_runner=aggregator_runner,
            logger=logger,
            max_depth=3,
            max_concurrent_slms=4
        )

        res = await pipeline_v3.execute_query(
            query_id="V3_TEST_01",
            query_text="Derive the Kalman filter state update equations and implement a vectorized Python simulation with edge-case unit tests.",
            complexity_tier="two_domain",
            seed=42,
            config={"version": "3.0.0"}
        )

        assert res["query_id"] == "V3_TEST_01"
        assert len(res["response"]) > 0
        assert res["matched_tasks_count"] >= 1
        assert "record" in res
        assert "v3_task_skill_vectors" in res["record"]
        assert "v3_task_colors" in res["record"]

    asyncio.run(_test())

