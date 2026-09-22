"""
Autonomous Audit Script for Phase F Pipeline Integration Test
Performs the 6 mandatory audit checks on the 6 double-blind judge trials
for SLMPipeline_v5 with phi3.5-ft-retrieval on V3_CD_21.
"""

import os
import json
import glob
from typing import Dict, Any, List

KEY_DIR = "logs/phase_f_judge_keys"
OUT_FILE = "results/phase_f/v5_ft_retrieval_pipeline_audit.json"

def run_audit():
    key_pattern = os.path.join(KEY_DIR, "key_V3_CD_21_slm_v5_ft_retrieval_*.json")
    key_files = sorted(glob.glob(key_pattern))
    
    print(f"Found {len(key_files)} key files for V3_CD_21 pipeline evaluation.")
    assert len(key_files) == 6, f"Expected exactly 6 key files, found {len(key_files)}"

    trials = []
    for kf in key_files:
        with open(kf, "r", encoding="utf-8") as f:
            k = json.load(f)
        
        pub_file = k["public_log_file"]
        with open(pub_file, "r", encoding="utf-8") as f:
            j = json.load(f)

        scores = j["criteria_scores"]
        sel = j["selected_candidate"]
        sc_a = scores["Candidate A"]["correctness"] + scores["Candidate A"]["completeness"] + scores["Candidate A"]["coherence"]
        sc_b = scores["Candidate B"]["correctness"] + scores["Candidate B"]["completeness"] + scores["Candidate B"]["coherence"]
        expected_sel = "Candidate A" if sc_a > sc_b else ("Candidate B" if sc_b > sc_a else "Tie")
        concordance = (sel == expected_sel)

        exp = j.get("reasoning", "")
        trunc = any(w in exp.lower() for w in ["truncat", "cut off", "abrupt", "incomplete function"])

        # Determine which candidate is SLM
        cand_a_is_slm = (k["candidate_a_system"] == "slm_v5_ft_retrieval")
        slm_scores = scores["Candidate A"] if cand_a_is_slm else scores["Candidate B"]
        base_scores = scores["Candidate B"] if cand_a_is_slm else scores["Candidate A"]
        baseline_sys = k["candidate_b_system"] if cand_a_is_slm else k["candidate_a_system"]

        trials.append({
            "baseline_id": baseline_sys,
            "order_tag": k["order_tag"],
            "candidate_a_system": k["candidate_a_system"],
            "candidate_b_system": k["candidate_b_system"],
            "selected_alias": sel,
            "unblinded_winner": k["unblinded_winner"],
            "slm_criteria_scores": slm_scores,
            "baseline_criteria_scores": base_scores,
            "slm_score_sum": sc_a if cand_a_is_slm else sc_b,
            "baseline_score_sum": sc_b if cand_a_is_slm else sc_a,
            "primary_differentiator": j.get("primary_differentiator", "N/A"),
            "concordance": concordance,
            "truncation_flagged": trunc,
            "reasoning": exp,
            "public_log": pub_file,
            "key_file": kf
        })

    # Group by baseline
    by_baseline = {}
    for t in trials:
        by_baseline.setdefault(t["baseline_id"], []).append(t)

    summary_by_baseline = {}
    for b_id, b_trials in by_baseline.items():
        assert len(b_trials) == 2, f"Baseline {b_id} has {len(b_trials)} trials, expected 2"
        w1 = b_trials[0]["unblinded_winner"]
        w2 = b_trials[1]["unblinded_winner"]
        agreed = (w1 == w2)
        slm_wins = sum(1 for t in b_trials if t["unblinded_winner"] == "slm_v5_ft_retrieval")
        base_wins = sum(1 for t in b_trials if t["unblinded_winner"] == b_id)
        
        avg_slm_score = sum(t["slm_score_sum"] for t in b_trials) / (2.0 * 3.0) # out of 5.0
        avg_base_score = sum(t["baseline_score_sum"] for t in b_trials) / (2.0 * 3.0) # out of 5.0

        summary_by_baseline[b_id] = {
            "trials_count": 2,
            "swap_agreement": agreed,
            "forward_winner": b_trials[0]["unblinded_winner"],
            "swapped_winner": b_trials[1]["unblinded_winner"],
            "slm_wins": slm_wins,
            "baseline_wins": base_wins,
            "slm_win_rate_pct": (slm_wins / 2.0) * 100.0,
            "avg_slm_composite_score": round(avg_slm_score, 3),
            "avg_baseline_composite_score": round(avg_base_score, 3),
            "truncation_confound": any(t["truncation_flagged"] for t in b_trials),
            "concordance_pass": all(t["concordance"] for t in b_trials)
        }

    total_trials = len(trials)
    concordance_rate = sum(1 for t in trials if t["concordance"]) / total_trials
    swap_consistency_rate = sum(1 for s in summary_by_baseline.values() if s["swap_agreement"]) / len(summary_by_baseline)
    truncation_free_rate = sum(1 for t in trials if not t["truncation_flagged"]) / total_trials

    # Load response metadata
    resp_path = "results/phase_f/v5_ft_pipeline_response.json"
    latency_sec = None
    char_len = None
    if os.path.exists(resp_path):
        with open(resp_path, "r", encoding="utf-8") as f:
            resp_data = json.load(f)
            latency_sec = resp_data.get("latency_sec")
            char_len = len(resp_data.get("response_text", ""))

    audit_summary = {
        "timestamp_utc": "2026-09-21T11:05:00Z",
        "query_id": "V3_CD_21",
        "system_under_test": "SLMPipeline_v5 (with phi3.5-ft-retrieval:latest)",
        "pipeline_latency_sec": latency_sec,
        "response_char_length": char_len,
        "audit_metrics": {
            "total_trials": total_trials,
            "concordance_rate_pct": round(concordance_rate * 100.0, 1),
            "swap_consistency_pct": round(swap_consistency_rate * 100.0, 1),
            "truncation_free_pct": round(truncation_free_rate * 100.0, 1)
        },
        "by_baseline_tier": summary_by_baseline,
        "detailed_trials": trials
    }

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    print("=" * 80)
    print("PHASE F PIPELINE INTEGRATION AUDIT COMPLETE")
    print("=" * 80)
    print(f"Total Trials: {total_trials}")
    print(f"Concordance Rate: {audit_summary['audit_metrics']['concordance_rate_pct']}%")
    print(f"Positional Swap Consistency: {audit_summary['audit_metrics']['swap_consistency_pct']}%")
    print(f"Truncation Free Rate: {audit_summary['audit_metrics']['truncation_free_pct']}%")
    print("\nBreakdown by Baseline Tier:")
    for b_id, s in summary_by_baseline.items():
        print(f"  [{b_id:12s}] SLM Wins: {s['slm_wins']}/2 ({s['slm_win_rate_pct']:.0f}%) | "
              f"Score: SLM {s['avg_slm_composite_score']:.2f} vs Base {s['avg_baseline_composite_score']:.2f} | "
              f"Swap Agreement: {s['swap_agreement']} | Truncation: {s['truncation_confound']}")

if __name__ == "__main__":
    run_audit()

