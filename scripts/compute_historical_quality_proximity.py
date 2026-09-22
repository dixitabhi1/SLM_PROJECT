"""
Master Quality Proximity & Signed Delta Ingestion Engine
AI Search Framework - Three-Dimensional Evaluation Framework

Ingests all historical judge logs and cryptographic keys across:
1. Phase v2 Pilot (Single Domain vs Llama-8B, Qwen-32B, Llama-70B, Qwen-72B, Gemini-1.5-Pro)
2. Phase v3 Pilot (Compound DAGs vs 120B Baseline)
3. Phase v5.1 Rescue Ladder (~20B, ~32B, 120B)
4. Step B Clean Run (V3_CD_01 & V3_CD_41)
5. Phase F Parameter Adaptation (Retrieval Specialist, Coding Specialist, and Pipeline Integration)

Computes for every cohort:
- Mean SLM CQS
- Mean Baseline CQS
- Mean Quality Proximity (P_mean) with 95% CI
- Mean Signed Delta (ΔQ) with 95% CI
- Win Rate (%)
- Concordance & Truncation-Free status

Saves authoritative ledger to results/quality_proximity_master_ledger.json.
"""

import os
import sys
import glob
import json
import math
import statistics
from typing import Dict, Any, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.analysis.metrics import StatisticalAnalyzer

RESULTS_DIR = "results"
OUT_FILE = os.path.join(RESULTS_DIR, "quality_proximity_master_ledger.json")

analyzer = StatisticalAnalyzer()

def parse_key_and_public_logs(key_pattern: str, slm_identifiers: List[str]) -> List[Dict[str, Any]]:
    """
    Parses matched key and public judge logs, extracting candidate CQS scores.
    """
    key_files = sorted(glob.glob(key_pattern))
    parsed_trials = []

    for kf in key_files:
        try:
            with open(kf, "r", encoding="utf-8") as f:
                k = json.load(f)
        except Exception:
            continue

        pub_file = k.get("public_log_file")
        if not pub_file or not os.path.exists(pub_file):
            continue

        try:
            with open(pub_file, "r", encoding="utf-8") as f:
                p = json.load(f)
        except Exception:
            continue

        scores = p.get("criteria_scores", {})
        if "Candidate A" not in scores or "Candidate B" not in scores:
            continue

        sc_a = scores["Candidate A"]
        sc_b = scores["Candidate B"]

        cqs_a = (sc_a.get("correctness", 0) + sc_a.get("completeness", 0) + sc_a.get("coherence", 0)) / 3.0
        cqs_b = (sc_b.get("correctness", 0) + sc_b.get("completeness", 0) + sc_b.get("coherence", 0)) / 3.0

        cand_a_sys = k.get("candidate_a_system", "")
        cand_b_sys = k.get("candidate_b_system", "")

        cand_a_is_slm = any(ident in cand_a_sys.lower() for ident in slm_identifiers)
        cand_b_is_slm = any(ident in cand_b_sys.lower() for ident in slm_identifiers)

        if cand_a_is_slm and not cand_b_is_slm:
            slm_cqs = cqs_a
            base_cqs = cqs_b
            base_sys = cand_b_sys
        elif cand_b_is_slm and not cand_a_is_slm:
            slm_cqs = cqs_b
            base_cqs = cqs_a
            base_sys = cand_a_sys
        else:
            # Symmetrical specialist comparisons (e.g. FT vs Base)
            slm_cqs = cqs_a
            base_cqs = cqs_b
            base_sys = cand_b_sys

        sel = p.get("selected_candidate", "")
        reasoning = p.get("reasoning", "")
        trunc = any(w in reasoning.lower() for w in ["truncat", "cut off", "abrupt"])

        parsed_trials.append({
            "query_id": k.get("query_id", ""),
            "order_tag": k.get("order_tag", ""),
            "baseline_system": base_sys,
            "slm_cqs": slm_cqs,
            "baseline_cqs": base_cqs,
            "unblinded_winner": k.get("unblinded_winner", ""),
            "truncation_flagged": trunc,
            "key_file": kf,
            "public_file": pub_file
        })

    return parsed_trials

def analyze_cohort(trials: List[Dict[str, Any]], cohort_name: str) -> Dict[str, Any]:
    if not trials:
        return {"cohort": cohort_name, "error": "No trials found"}

    slm_scores = [t["slm_cqs"] for t in trials]
    base_scores = [t["baseline_cqs"] for t in trials]

    stats = analyzer.compute_quality_proximity(slm_scores, base_scores)

    # Positional agreement calculation
    by_query = {}
    for t in trials:
        by_query.setdefault(t["query_id"], []).append(t["unblinded_winner"])

    agreements = sum(1 for qid, w_list in by_query.items() if len(w_list) == 2 and w_list[0] == w_list[1])
    total_pairs = sum(1 for qid, w_list in by_query.items() if len(w_list) == 2)
    swap_consistency = (agreements / total_pairs * 100.0) if total_pairs > 0 else 0.0

    trunc_count = sum(1 for t in trials if t["truncation_flagged"])

    return {
        "cohort_name": cohort_name,
        "total_trials": len(trials),
        "total_queries": len(by_query),
        "slm_wins": stats["slm_wins"],
        "baseline_wins": stats["baseline_wins"],
        "ties": stats["ties"],
        "slm_win_rate_pct": stats["slm_win_rate_pct"],
        "mean_slm_cqs": stats["mean_slm_cqs"],
        "mean_baseline_cqs": stats["mean_baseline_cqs"],
        "mean_quality_proximity_pct": stats["mean_quality_proximity_pct"],
        "proximity_ci_95": [stats["proximity_ci_95_lower_pct"], stats["proximity_ci_95_upper_pct"]],
        "mean_quality_delta": stats["mean_quality_delta"],
        "delta_ci_95": [stats["delta_ci_95_lower"], stats["delta_ci_95_upper"]],
        "swap_consistency_pct": round(swap_consistency, 2),
        "truncation_flagged_trials": trunc_count,
        "truncation_free_rate_pct": round((1.0 - trunc_count / len(trials)) * 100.0, 2)
    }

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 80)
    print("COMPUTING MASTER HISTORICAL QUALITY PROXIMITY LEDGER")
    print("=" * 80)

    cohorts = {}

    # 1. Phase v2 Pilot (Single Domain vs Diverse Baselines)
    v2_trials = parse_key_and_public_logs("logs/judge_keys/key_*.json", ["slm_pipeline", "slm_council", "candidate_a"])
    if v2_trials:
        cohorts["v2_pilot_single_domain_all_baselines"] = analyze_cohort(v2_trials, "Phase v2 Pilot (Single Domain, vs 8B-70B-Gemini)")
        # Split by baseline
        by_b = {}
        for t in v2_trials:
            by_b.setdefault(t["baseline_system"], []).append(t)
        for b_name, b_trials in by_b.items():
            clean_b = b_name.replace("/", "_")
            cohorts[f"v2_pilot_vs_{clean_b}"] = analyze_cohort(b_trials, f"Phase v2 Pilot vs {b_name}")

    # 2. Phase v3 Pilot (Compound DAGs vs 120B Baseline)
    v3_trials = parse_key_and_public_logs("logs/v3_judge_keys/key_*.json", ["slm_pipeline", "slm_council", "candidate_a"])
    if v3_trials:
        cohorts["v3_pilot_compound_vs_120b"] = analyze_cohort(v3_trials, "Phase v3 Pilot (Compound DAGs vs 120B)")

    # 3. Phase v5.1 Multi-Scale Ladder
    for scale_id, label in [("20b", "20B Baseline"), ("32b", "32B Baseline"), ("120b", "120B Baseline")]:
        v5_scale_trials = parse_key_and_public_logs(f"logs/v5_1_judge_keys_{scale_id}/key_*.json", ["slm_council", "slm_v5"])
        if v5_scale_trials:
            cohorts[f"v5_1_ladder_vs_{scale_id}"] = analyze_cohort(v5_scale_trials, f"Phase v5.1 Council vs {label}")

    # 4. Phase Step B (Audited Coding Loop)
    step_b_trials = parse_key_and_public_logs("logs/step_b_judge_keys/key_*.json", ["slm_pipeline", "slm_council", "slm_v5"])
    if step_b_trials:
        # Separate un-truncated V3_CD_01 from truncated V3_CD_41
        cd01_trials = [t for t in step_b_trials if "V3_CD_01" in t["query_id"]]
        cd41_trials = [t for t in step_b_trials if "V3_CD_41" in t["query_id"]]
        if cd01_trials:
            cohorts["step_b_audited_clean_V3_CD_01"] = analyze_cohort(cd01_trials, "Step B Audited Clean Cohort (V3_CD_01)")
        if cd41_trials:
            cohorts["step_b_truncated_V3_CD_41"] = analyze_cohort(cd41_trials, "Step B Truncated Cohort (V3_CD_41 - Voided)")

    # 5. Phase F Parameter Adaptation Cohorts
    # 5a. Retrieval Specialist Held-Out (FT vs Base)
    ret_ft_trials = parse_key_and_public_logs("logs/phase_f_judge_keys/key_RET_QA_*.json", ["finetuned", "ft"])
    if ret_ft_trials:
        cohorts["phase_f_retrieval_specialist_ft_vs_base"] = analyze_cohort(ret_ft_trials, "Phase F Retrieval Specialist (FT vs Base)")

    # 5b. Coding Specialist Held-Out (FT vs Base)
    code_ft_trials = parse_key_and_public_logs("logs/phase_f_judge_keys/key_CODE_QA_*.json", ["phi3.5_coder_ft", "coder_ft", "ft"])
    if code_ft_trials:
        cohorts["phase_f_coding_specialist_ft_vs_base"] = analyze_cohort(code_ft_trials, "Phase F Coding Specialist (FT vs Base)")

    # 5c. Pipeline Integration on V3_CD_21
    pipe_f_trials = parse_key_and_public_logs("logs/phase_f_judge_keys/key_V3_CD_21_slm_v5_ft_*.json", ["slm_v5_ft_retrieval"])
    if pipe_f_trials:
        cohorts["phase_f_pipeline_v3_cd_21_all_baselines"] = analyze_cohort(pipe_f_trials, "Phase F Pipeline on V3_CD_21 (vs 20B/32B/120B)")
        # By baseline tier
        for b_name in ["gpt_20b", "gemini_32b", "gpt_120b"]:
            sub = [t for t in pipe_f_trials if b_name in t["baseline_system"]]
            if sub:
                cohorts[f"phase_f_pipeline_v3_cd_21_vs_{b_name}"] = analyze_cohort(sub, f"Phase F Pipeline V3_CD_21 vs {b_name}")

    master_ledger = {
        "timestamp_utc": "2026-09-22T23:25:00Z",
        "evaluation_framework": "Three-Dimensional Evaluation Paradigm (Win Rate, Quality Proximity, Cost/Compute)",
        "formula": {
            "proximity_definition": "P_i = 1 - (|Q_S,i - Q_L,i| / 4.0)",
            "mean_proximity": "P_mean = (1/N) * sum(P_i) * 100%",
            "signed_delta": "ΔQ_i = Q_S,i - Q_L,i",
            "mean_delta": "Mean_ΔQ = (1/N) * sum(ΔQ_i)"
        },
        "cohorts": cohorts
    }

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(master_ledger, f, indent=2)

    print(f"\nSuccessfully generated Master Quality Proximity Ledger at {OUT_FILE}")
    print("\nAuthoritative Summary Across Key Project Phases:")
    print("-" * 115)
    print(f"{'Cohort Name':<42} | {'Win Rate':<10} | {'SLM CQS':<8} | {'Base CQS':<9} | {'Mean DeltaQ':<11} | {'Quality Proximity':<18} | {'N':<4}")
    print("-" * 115)
    for c_id, c in cohorts.items():
        if "error" in c:
            continue
        print(f"{c['cohort_name'][:42]:<42} | {c['slm_win_rate_pct']:>8.1f}% | {c['mean_slm_cqs']:>8.2f} | {c['mean_baseline_cqs']:>9.2f} | {c['mean_quality_delta']:>+11.2f} | {c['mean_quality_proximity_pct']:>8.2f}% ({c['proximity_ci_95'][0]:.1f}-{c['proximity_ci_95'][1]:.1f}%) | {c['total_trials']:>4}")
    print("-" * 115)

if __name__ == "__main__":
    main()
