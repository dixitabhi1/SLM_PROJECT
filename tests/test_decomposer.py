"""
Unit Tests for Task Graph Decomposer
"""

import asyncio
from src.models.mock_runner import MockModelRunner
from src.decomposer.decomposer import TaskGraphDecomposer

def test_decomposer_valid_dag():
    async def _test():
        runner = MockModelRunner(model_name="meta-llama/Llama-3.2-3B-Instruct")
        decomposer = TaskGraphDecomposer(runner)
        
        res = await decomposer.decompose("Derive the Kalman filter equations and implement in Python.")
        assert "dag" in res
        assert res["is_schema_valid"] is True
        subtasks = res["dag"]["subtasks"]
        assert len(subtasks) >= 1
        assert "node_1" in [s["id"] for s in subtasks]
    asyncio.run(_test())

def test_decomposer_fallback_handling():
    runner = MockModelRunner(model_name="meta-llama/Llama-3.2-3B-Instruct")
    decomposer = TaskGraphDecomposer(runner)
    
    dag, valid, err = decomposer._parse_and_validate_dag("Invalid non-json response text", "What is 2+2?")
    assert valid is False
    assert len(dag["subtasks"]) == 1
    assert dag["subtasks"][0]["text"] == "What is 2+2?"

