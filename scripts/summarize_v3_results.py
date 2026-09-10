"""
v3 Results Summary and Analysis
Analyzes 128 pairwise double-blind judge trials and generation metrics for Version 3.
"""

import os
import json
import glob
from collections import defaultdict
from typing import Dict, Any, List

KEY_LOG_DIR = "logs/v3_judge_keys"
PUBLIC_LOG_DIR = "logs/v3_judge_pairwise"
PILOT_DIR = "results/v3_pilot"
SLM_FILE = os.path.join(PILOT_DIR, "slm_pipeline_responses.jsonl")
BASELINE_FILE = os.path.join(PILOT_DIR, "llm_baseline_responses.jsonl")
COMPARISON_FILE = os.path.join(PILOT_DIR, "comparison.jsonl")

BASELINES_V3 = ["qwen_32b", "llama_70b", "qwen_72b", "gemini_frontier"]

def analyze_judge_results():
    if not os.path.exists(KEY_LOG_DIR):
        print("No key logs found.")
        return

    key_files = glob.glob(os.path.join(KEY_LOG_DIR, "key_*.json"))
    if not key_files:
        print("No key log files found.")
        return

    trials = []
    for kf in key_files:
        try:
            with open(kf, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("status") == "SUCCESS":
                    trials.append(data)
        except Exception:
            pass

    print(f"Loaded {len(trials)} successful judge trials.\n")

    # Group by baseline
    baseline_stats = defaultdict(lambda: {"wins": 0, "losses": 0, "ties": 0, "total": 0})
    # Group by tier
    tier_stats = defaultdict(lambda: {"wins": 0, "losses": 0, "ties": 0, "total": 0})
    # Positional agreement: query_id + baseline -> {forward_winner, swapped_winner}
    pair_orders = defaultdict(dict)

    for t in trials:
        qid = t["query_id"]
        tier = "SD" if "SD" in qid else ("TD" if "TD" in qid else "CD")
        unblinded_winner = t["unblinded_winner"]
        order = t["order_tag"]

        # Figure out which baseline was evaluated
        cand_a = t["candidate_a_system"]
        cand_b = t["candidate_b_system"]
        baseline_id = cand_b if cand_a == "slm_pipeline_v3" else cand_a

        pair_orders[(qid, baseline_id)][order] = unblinded_winner

        is_slm_win = (unblinded_winner == "slm_pipeline_v3")
        is_tie = (unblinded_winner == "Tie")
        is_base_win = (unblinded_winner == baseline_id)

        stat = baseline_stats[baseline_id]
        stat["total"] += 1
        tstat = tier_stats[tier]
        tstat["total"] += 1

        if is_slm_win:
            stat["wins"] += 1
            tstat["wins"] += 1
        elif is_tie:
            stat["ties"] += 1
            tstat["ties"] += 1
        elif is_base_win:
            stat["losses"] += 1
            tstat["losses"] += 1

    total_wins = sum(s["wins"] for s in baseline_stats.values())
    total_losses = sum(s["losses"] for s in baseline_stats.values())
    total_ties = sum(s["ties"] for s in baseline_stats.values())
    total_trials = sum(s["total"] for s in baseline_stats.values())

    win_rate = (total_wins / total_trials * 100.0) if total_trials else 0.0
    win_rate_no_tie = (total_wins / (total_wins + total_losses) * 100.0) if (total_wins + total_losses) else 0.0

    print("=" * 70)
    print("AI SEARCH FRAMEWORK v3: DOUBLE-BLIND PAIRWISE BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Total Completed Pairwise Trials: {total_trials}")
    print(f"Overall SLM Win Rate (inc. ties): {win_rate:.1f}% ({total_wins}W / {total_losses}L / {total_ties}T)")
    print(f"Overall SLM Win Rate (decisive):  {win_rate_no_tie:.1f}%\n")

    print("-" * 70)
    print("RESULTS BY BASELINE MODEL (Floor >= 30B):")
    print(f"{'Baseline':<18} {'Trials':<8} {'Wins':<8} {'Losses':<8} {'Ties':<8} {'Win Rate':<10}")
    print("-" * 70)
    for b in BASELINES_V3:
        s = baseline_stats[b]
        wr = (s["wins"] / s["total"] * 100.0) if s["total"] else 0.0
        print(f"{b:<18} {s['total']:<8} {s['wins']:<8} {s['losses']:<8} {s['ties']:<8} {wr:.1f}%")
    print("-" * 70)

    print("\nRESULTS BY QUERY COMPLEXITY TIER:")
    print(f"{'Tier':<18} {'Trials':<8} {'Wins':<8} {'Losses':<8} {'Ties':<8} {'Win Rate':<10}")
    print("-" * 70)
    for tier in ["SD", "TD", "CD"]:
        s = tier_stats[tier]
        wr = (s["wins"] / s["total"] * 100.0) if s["total"] else 0.0
        desc = "Single-Domain" if tier == "SD" else ("Two-Domain" if tier == "TD" else "Multi-Domain")
        print(f"{desc:<18} {s['total']:<8} {s['wins']:<8} {s['losses']:<8} {s['ties']:<8} {wr:.1f}%")
    print("-" * 70)

    # Order Consistency
    consistent = 0
    total_pairs = len(pair_orders)
    for (qid, bid), orders in pair_orders.items():
        fw = orders.get("forward")
        sw = orders.get("swapped")
        if fw and sw:
            if fw == sw:
                consistent += 1

    print(f"\nOrder Consistency (Positional Robustness):")
    print(f"Symmetric pairs completed: {total_pairs}")
    if total_pairs:
        print(f"Order Agreement: {consistent}/{total_pairs} ({consistent/total_pairs*100:.1f}%)")

def analyze_generation_metrics():
    if not os.path.exists(SLM_FILE) or not os.path.exists(BASELINE_FILE):
        return

    slm_data = []
    with open(SLM_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                slm_data.append(json.loads(line))

    baseline_data = defaultdict(list)
    with open(BASELINE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                baseline_data[r["baseline_model_id"]].append(r)

    print("\n" + "=" * 70)
    print("GENERATION PROFILE & EFFICIENCY METRICS")
    print("=" * 70)
    print(f"{'System':<18} {'Queries':<8} {'Avg Latency (s)':<18} {'Avg Out Chars':<15}")
    print("-" * 70)

    avg_slm_lat = sum(x.get("wall_clock_latency_ms", 0.0) / 1000.0 for x in slm_data) / len(slm_data) if slm_data else 0
    avg_slm_chars = sum(len(x.get("response_text", "")) for x in slm_data) / len(slm_data) if slm_data else 0
    print(f"{'SLMPipeline_v3':<18} {len(slm_data):<8} {avg_slm_lat:<18.2f} {avg_slm_chars:<15.0f}")

    for b in BASELINES_V3:
        b_list = baseline_data[b]
        avg_lat = sum(x.get("latency_ms", 0.0) / 1000.0 for x in b_list) / len(b_list) if b_list else 0
        avg_chars = sum(len(x.get("response_text", "")) for x in b_list) / len(b_list) if b_list else 0
        print(f"{b:<18} {len(b_list):<8} {avg_lat:<18.2f} {avg_chars:<15.0f}")
    print("-" * 70)

if __name__ == "__main__":
    analyze_judge_results()
    analyze_generation_metrics()

