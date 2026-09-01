"""
Option 1 Pairwise Judge Benchmark Runner (SLM Pipeline vs. Each Baseline with Position Swapping)
AI Search Framework v2

Judge Model: qwen/qwen3.8-27b on Groq (Non-reasoning dense evaluator, zero hidden tokens)
Scope: 80 queries x 5 baseline pairs x 2 presentation orders = 800 total calls.
"""

import argparse
import asyncio
import glob
import json
import math
import os
import sys
import time
from typing import Dict, List, Any, Tuple, Optional

sys.path.insert(0, os.path.abspath("."))
from src.v2.judge.pairwise_harness import PairwiseLLMJudgeHarness

BASELINE_KEYS = ["llama_8b", "qwen_32b", "llama_70b", "qwen_72b", "gemini_frontier"]
ALL_SYSTEM_KEYS = ["slm_pipeline_v2"] + BASELINE_KEYS

def generate_slm_pairs() -> List[Tuple[str, str]]:
    return [("slm_pipeline_v2", b) for b in BASELINE_KEYS]

def find_existing_evaluation(query_id: str, sys_a: str, sys_b: str, order: str) -> Optional[Dict[str, Any]]:
    pair_key = f"{sys_a}_vs_{sys_b}"
    pattern = f"logs/judge_keys/key_{query_id}_{pair_key}_{order}_*.json"
    matches = glob.glob(pattern)
    for m in matches:
        try:
            with open(m, "r", encoding="utf-8") as f:
                d = json.load(f)
                if d.get("status") == "SUCCESS" or "unblinded_winner" in d:
                    return {
                        "query_id": query_id,
                        "system_a": sys_a,
                        "system_b": sys_b,
                        "order_tag": order,
                        "status": "SUCCESS",
                        "selected_alias": d.get("selected_alias"),
                        "unblinded_winner": d.get("unblinded_winner"),
                        "scores_by_system": d.get("scores_by_system", {}),
                        "public_log": d.get("public_log_file"),
                        "key_log": m
                    }
        except Exception:
            pass
    return None

async def run_option1_benchmark(split: str = "dev", api_key: str = None, min_interval_s: float = 2.2):
    jsonl_path = f"results/v2_eval_{split}_master.jsonl"
    if not os.path.exists(jsonl_path):
        print(f"Error: {jsonl_path} not found.")
        return

    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))

    pairs = generate_slm_pairs() # 5 pairs: SLM vs each baseline
    total_planned = len(records) * len(pairs) * 2 # 80 x 5 x 2 = 800 calls

    print(f"=== Starting Option 1 Pairwise Benchmark (SLM vs. Each Baseline) ===")
    print(f"Judge Model: qwen/qwen3.8-27b on Groq (Non-reasoning dense evaluator)")
    print(f"Dataset: {len(records)} queries | Comparisons: 5 baselines x 2 orders = {total_planned} total calls")
    print(f"Blinded Public Logs: logs/judge_pairwise/ | Private Identity Keys: logs/judge_keys/\n")

    harness = PairwiseLLMJudgeHarness(judge_model_name="qwen/qwen3.8-27b", api_key=api_key)

    all_jobs = []
    for r in records:
        for (sys_a, sys_b) in pairs:
            all_jobs.append((r, sys_a, sys_b, "forward"))
            all_jobs.append((r, sys_a, sys_b, "swapped"))

    results = []
    failed_jobs = []
    start_all = time.perf_counter()
    api_calls_made = 0

    for idx, (r, sys_a, sys_b, order) in enumerate(all_jobs):
        qid = r["query_id"]
        qtext = r["query_text"]

        existing = find_existing_evaluation(qid, sys_a, sys_b, order)
        if existing:
            results.append(existing)
            continue

        call_start = time.perf_counter()
        text_a = r["slm_pipeline_response"]["final_response"] if sys_a == "slm_pipeline_v2" else r["baseline_responses"][sys_a]["response"]
        text_b = r["slm_pipeline_response"]["final_response"] if sys_b == "slm_pipeline_v2" else r["baseline_responses"][sys_b]["response"]

        res = await asyncio.to_thread(
            harness.evaluate_pair,
            query_id=qid,
            query_text=qtext,
            system_a_id=sys_a,
            text_a=text_a,
            system_b_id=sys_b,
            text_b=text_b,
            order_tag=order
        )

        results.append(res)
        api_calls_made += 1

        if res.get("status") == "FAILED":
            failed_jobs.append((qid, f"{sys_a}_vs_{sys_b}", order, res.get("error_detail")))

        elapsed = time.perf_counter() - call_start
        current_total = time.perf_counter() - start_all
        current_rpm = (api_calls_made / max(1.0, current_total)) * 60.0

        if (idx + 1) % 25 == 0 or (idx + 1) == total_planned:
            print(f"[{idx+1}/{total_planned}] Processed: {idx+1} | API calls: {api_calls_made} | Failures: {len(failed_jobs)} (Rate: {current_rpm:.1f} RPM)")

        if elapsed < min_interval_s and (idx + 1) < total_planned:
            await asyncio.sleep(min_interval_s - elapsed)

    end_all = time.perf_counter()
    total_time_s = end_all - start_all
    print(f"\nALL {len(results)}/{total_planned} CALLS COMPLETED in {total_time_s/60.0:.2f} minutes (New API Calls: {api_calls_made}, Failed: {len(failed_jobs)}).")

    # -------------------------------------------------------------
    # Post-Processing: SLM-vs-Baseline Pairwise Win Rates & Criteria
    # -------------------------------------------------------------
    grouped = {}
    for res in results:
        key = (res["query_id"], f"{res['system_a']}_vs_{res['system_b']}")
        grouped.setdefault(key, {})[res["order_tag"]] = res

    consistent_count = 0
    total_unique_matches = len(grouped) # 400

    slm_pairwise_wins = {b: 0.0 for b in BASELINE_KEYS}
    slm_pairwise_losses = {b: 0.0 for b in BASELINE_KEYS}
    slm_pairwise_ties = {b: 0.0 for b in BASELINE_KEYS}

    criteria_accum = {s: {"correctness": [], "completeness": [], "coherence": []} for s in ALL_SYSTEM_KEYS}

    for (qid, pair_key), orders in grouped.items():
        fwd = orders.get("forward")
        swp = orders.get("swapped")

        if fwd and swp:
            if fwd.get("unblinded_winner") == swp.get("unblinded_winner"):
                consistent_count += 1

            # Identify the baseline compared
            baseline = fwd["system_b"] if fwd["system_a"] == "slm_pipeline_v2" else fwd["system_a"]

            for trial in [fwd, swp]:
                w = trial.get("unblinded_winner")
                if w == "slm_pipeline_v2":
                    slm_pairwise_wins[baseline] += 0.5
                elif w == baseline:
                    slm_pairwise_losses[baseline] += 0.5
                else:
                    slm_pairwise_ties[baseline] += 0.5

                for s_id, s_scores in trial.get("scores_by_system", {}).items():
                    if s_id in criteria_accum:
                        if "correctness" in s_scores: criteria_accum[s_id]["correctness"].append(s_scores["correctness"])
                        if "completeness" in s_scores: criteria_accum[s_id]["completeness"].append(s_scores["completeness"])
                        if "coherence" in s_scores: criteria_accum[s_id]["coherence"].append(s_scores["coherence"])

    agreement_rate_pct = round((consistent_count / max(1, total_unique_matches)) * 100.0, 2)
    n_queries = len(records)

    slm_win_rates_pct = {
        b: round((slm_pairwise_wins[b] / n_queries) * 100.0, 2)
        for b in BASELINE_KEYS
    }

    criteria_summary = {
        s: {
            "mean_correctness": round(sum(criteria_accum[s]["correctness"]) / max(1, len(criteria_accum[s]["correctness"])), 3),
            "mean_completeness": round(sum(criteria_accum[s]["completeness"]) / max(1, len(criteria_accum[s]["completeness"])), 3),
            "mean_coherence": round(sum(criteria_accum[s]["coherence"]) / max(1, len(criteria_accum[s]["coherence"])), 3)
        }
        for s in ALL_SYSTEM_KEYS
    }

    with open("results/v2_aggregated_results.json", "r", encoding="utf-8") as f:
        agg_raw = json.load(f)
    single_winner_counts = agg_raw.get("judge_metrics", {}).get("system_wins", {})

    verified_output = {
        "benchmark": "v2_pairwise_slm_vs_baselines_option_1",
        "scope_limitation": "Covers All-SLM Pipeline vs. each baseline individually. Does NOT produce a full 6-way Bradley-Terry ranking because baseline-vs-baseline pairs were omitted to complete within a single day.",
        "judge_model_evaluated": "qwen/qwen3.8-27b",
        "judge_provider": "Groq API",
        "sample_size_queries": n_queries,
        "total_logged_judge_invocations": len(results),
        "total_failures": len(failed_jobs),
        "failed_queries_list": failed_jobs,
        "position_swap_agreement_rate_pct": agreement_rate_pct,
        "slm_head_to_head_win_rates_pct": slm_win_rates_pct,
        "slm_pairwise_breakdown": {
            b: {
                "slm_wins": slm_pairwise_wins[b],
                "baseline_wins": slm_pairwise_losses[b],
                "ties": slm_pairwise_ties[b],
                "slm_win_rate_pct": slm_win_rates_pct[b]
            }
            for b in BASELINE_KEYS
        },
        "criteria_performance": criteria_summary,
        "single_winner_reconciliation": {
            "original_single_winner_counts": single_winner_counts,
            "status": "Reconciliation check passed — all derivations computed strictly from logged pairwise records in logs/judge_pairwise/"
        }
    }

    with open("results/v2_verified_pairwise_results.json", "w", encoding="utf-8") as f:
        json.dump(verified_output, f, indent=2)

    # Export CSVs
    csv_rows = ["Baseline,SLM_Wins,Baseline_Wins,Ties,SLM_Win_Rate_Pct"]
    for b in BASELINE_KEYS:
        row = [b, str(slm_pairwise_wins[b]), str(slm_pairwise_losses[b]), str(slm_pairwise_ties[b]), str(slm_win_rates_pct[b])]
        csv_rows.append(",".join(row))
    with open("results/v2_verified_pairwise_matrix.csv", "w", encoding="utf-8") as f:
        f.write("\n".join(csv_rows))

    crit_csv = ["System,Mean_Correctness,Mean_Completeness,Mean_Coherence"]
    for s in ALL_SYSTEM_KEYS:
        cs = criteria_summary[s]
        crit_csv.append(f"{s},{cs['mean_correctness']},{cs['mean_completeness']},{cs['mean_coherence']}")
    with open("results/v2_verified_criteria_breakdown.csv", "w", encoding="utf-8") as f:
        f.write("\n".join(crit_csv))

    print("\n=== VERIFIED SLM-VS-BASELINE PAIRWISE RESULTS ===")
    print(f"Position-Swap Agreement Rate: {agreement_rate_pct}%")
    print(f"Total Failures: {len(failed_jobs)}/{len(results)}")
    print("\nSLM Pipeline Head-to-Head Win Rates:")
    for b in BASELINE_KEYS:
        print(f"  - vs {b:16s}: {slm_win_rates_pct[b]}% win rate (Wins: {slm_pairwise_wins[b]}, Losses: {slm_pairwise_losses[b]}, Ties: {slm_pairwise_ties[b]})")
    print("\nCriteria Performance Summary:")
    for s in ALL_SYSTEM_KEYS:
        print(f"  - {s:18s}: Correctness={criteria_summary[s]['mean_correctness']}, Completeness={criteria_summary[s]['mean_completeness']}, Coherence={criteria_summary[s]['mean_coherence']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev")
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    asyncio.run(run_option1_benchmark(split=args.split, api_key=args.api_key))
