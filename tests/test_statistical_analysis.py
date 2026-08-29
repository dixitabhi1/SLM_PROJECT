"""
Unit Tests for Statistical Analysis and GED Engine
"""

import pytest
from src.analysis.metrics import StatisticalAnalyzer

def test_paired_ratio_stats():
    analyzer = StatisticalAnalyzer()
    slm_latencies = [120.0, 150.0, 110.0, 130.0, 140.0]
    baseline_latencies = [450.0, 500.0, 480.0, 520.0, 490.0]
    
    stats = analyzer.compute_paired_ratio_stats(slm_latencies, baseline_latencies)
    assert stats["n"] == 5
    assert stats["mean_ratio"] < 0.35
    assert stats["ci_95_lower"] < stats["ci_95_upper"]

def test_non_inferiority_test():
    analyzer = StatisticalAnalyzer(non_inferiority_margin=0.20)
    slm_scores = [4.5, 4.2, 4.8, 4.0, 4.6]
    baseline_scores = [4.6, 4.3, 4.7, 4.1, 4.5]
    
    res = analyzer.compute_non_inferiority_test(slm_scores, baseline_scores)
    assert "p_value" in res
    assert "non_inferiority_demonstrated" in res

def test_graph_edit_distance():
    analyzer = StatisticalAnalyzer()
    
    gen_dag = {
        "subtasks": [
            {"id": "node_1", "capability": "math", "dependencies": []},
            {"id": "node_2", "capability": "coding", "dependencies": ["node_1"]}
        ]
    }
    gold_dag = {
        "subtasks": [
            {"id": "node_1", "capability": "math", "dependencies": []},
            {"id": "node_2", "capability": "coding", "dependencies": ["node_1"]}
        ]
    }
    
    res = analyzer.compute_graph_edit_distance(gen_dag, gold_dag)
    assert res["exact_match"] is True
    assert res["ged"] == 0
    assert res["structural_similarity"] == 1.0

