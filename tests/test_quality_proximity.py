"""
Unit Tests for Three-Dimensional Evaluation: Quality Proximity (P) & Signed Delta (ΔQ)
Validates mathematical formulation against user specification and boundary conditions.
"""

import pytest
from src.analysis.metrics import StatisticalAnalyzer

def test_user_pdf_exact_examples():
    """
    Tests the exact numerical examples from Page 1 of the user's PDF:
    SLM CQS | LLM CQS | Proximity
    4.5     | 4.5     | 1.00
    4.0     | 4.5     | 0.875
    3.5     | 4.5     | 0.75
    2.0     | 4.5     | 0.375
    1.0     | 5.0     | 0.00
    """
    analyzer = StatisticalAnalyzer()
    slm = [4.5, 4.0, 3.5, 2.0, 1.0]
    llm = [4.5, 4.5, 4.5, 4.5, 5.0]
    expected_proximities = [1.00, 0.875, 0.75, 0.375, 0.00]
    expected_deltas = [0.0, -0.5, -1.0, -2.5, -4.0]

    res = analyzer.compute_quality_proximity(slm, llm)

    assert "error" not in res
    assert res["n"] == 5
    assert res["per_query_proximity"] == expected_proximities
    assert res["per_query_delta"] == expected_deltas

    # Mean Proximity: (1.00 + 0.875 + 0.75 + 0.375 + 0.00) / 5 = 3.0 / 5 = 0.60 -> 60.0%
    assert res["mean_quality_proximity_pct"] == 60.0
    # Mean Delta: (0.0 - 0.5 - 1.0 - 2.5 - 4.0) / 5 = -8.0 / 5 = -1.60
    assert res["mean_quality_delta"] == -1.60

def test_user_pdf_page2_delta_and_proximity_example():
    """
    Tests the scenario from Page 2 of the user's PDF:
    Mean SLM CQS = 3.82, Mean LLM CQS = 4.21, Mean ΔQ = -0.39, Quality Proximity = 90.25%
    """
    analyzer = StatisticalAnalyzer()
    # Single paired comparison with exact difference: 3.82 vs 4.21
    res = analyzer.compute_quality_proximity([3.82], [4.21])

    assert res["mean_slm_cqs"] == 3.82
    assert res["mean_baseline_cqs"] == 4.21
    assert round(res["mean_quality_delta"], 2) == -0.39
    assert res["mean_quality_proximity_pct"] == 90.25

def test_symmetry_and_reverse_lead():
    """
    Tests that if SLM beats the LLM, Proximity remains positive and bounded,
    while Signed Delta reflects a positive gain.
    """
    analyzer = StatisticalAnalyzer()
    slm = [4.5, 5.0]
    llm = [4.0, 4.0]

    res = analyzer.compute_quality_proximity(slm, llm)
    # Differences: |4.5 - 4.0| = 0.5 -> P = 0.875; |5.0 - 4.0| = 1.0 -> P = 0.750
    # Mean P = (0.875 + 0.750) / 2 = 0.8125 -> 81.25%
    assert res["mean_quality_proximity_pct"] == 81.25
    # Deltas: +0.5, +1.0 -> Mean ΔQ = +0.75
    assert res["mean_quality_delta"] == 0.75
    assert res["slm_wins"] == 2
    assert res["slm_win_rate_pct"] == 100.0

def test_confidence_interval_bounds():
    """
    Verifies that 95% confidence intervals are mathematically valid and bounded.
    """
    analyzer = StatisticalAnalyzer()
    slm = [3.0, 3.5, 4.0, 3.2, 3.8, 4.2, 3.5, 3.9, 4.1, 3.6]
    llm = [4.0, 4.2, 4.5, 3.8, 4.0, 4.5, 4.1, 4.3, 4.4, 4.0]

    res = analyzer.compute_quality_proximity(slm, llm)
    assert res["n"] == 10
    assert 0.0 <= res["proximity_ci_95_lower_pct"] <= res["mean_quality_proximity_pct"]
    assert res["mean_quality_proximity_pct"] <= res["proximity_ci_95_upper_pct"] <= 100.0
    assert res["delta_ci_95_lower"] <= res["mean_quality_delta"] <= res["delta_ci_95_upper"]

def test_invalid_inputs():
    analyzer = StatisticalAnalyzer()
    assert "error" in analyzer.compute_quality_proximity([], [])
    assert "error" in analyzer.compute_quality_proximity([3.0], [3.0, 4.0])
    assert "error" in analyzer.compute_quality_proximity([3.0], [3.0], scale_min=5.0, scale_max=1.0)

