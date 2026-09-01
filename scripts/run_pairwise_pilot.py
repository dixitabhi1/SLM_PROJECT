"""
5-Query Pairwise LLM Judge Pilot (150 Real Groq Calls)
AI Search Framework v2
"""

import asyncio
import json
import os
import sys
import time
from typing import Dict, List, Any, Tuple

sys.path.insert(0, os.path.abspath("."))
from src.v2.judge.pairwise_harness import PairwiseLLMJudgeHarness

SYSTEM_KEYS = ["slm_pipeline_v2", "llama_8b", "qwen_32b", "llama_70b", "qwen_72b", "gemini_frontier"]

def generate_round_robin_pairs() -> List[Tuple[str, str]]:
    pairs = []
    for i in range(len(SYSTEM_KEYS)):
        for j in range(i + 1, len(SYSTEM_KEYS)):
            pairs.append((SYSTEM_KEYS[i], SYSTEM_KEYS[j]))
    return pairs

async def run_pilot(max_queries: int = 5, concurrency: int = 4):
    jsonl_path = "results/v2_eval_dev_master.jsonl"
    if not os.path.exists(jsonl_path):
        print(f"Error: {jsonl_path} not found.")
        return

    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))

    records = records[:max_queries]
    pairs = generate_round_robin_pairs() # 15 pairs
    total_calls = len(records) * len(pairs) * 2 # 150 calls

    print(f"=== Starting 5-Query Pairwise Pilot on Groq (openai/gpt-oss-120b) ===")
    print(f"Queries: {len(records)} | Pairs/Query: {len(pairs)} | Orders: 2 (A/B + B/A)")
    print(f"Total API Calls to Dispatch: {total_calls}\n")

    harness = PairwiseLLMJudgeHarness(judge_model_name="openai/gpt-oss-120b")
    semaphore = asyncio.Semaphore(concurrency)

    completed_calls = 0

    async def _evaluate_single(r: Dict[str, Any], sys_a: str, sys_b: str, order: str) -> Dict[str, Any]:
        nonlocal completed_calls
        async with semaphore:
            qid = r["query_id"]
            qtext = r["query_text"]

            text_a = r["slm_pipeline_response"]["final_response"] if sys_a == "slm_pipeline_v2" else r["baseline_responses"][sys_a]["response"]
            text_b = r["slm_pipeline_response"]["final_response"] if sys_b == "slm_pipeline_v2" else r["baseline_responses"][sys_b]["response"]

            res = await harness.evaluate_pair(
                query_id=qid,
                query_text=qtext,
                system_a_id=sys_a,
                text_a=text_a,
                system_b_id=sys_b,
                text_b=text_b,
                order_tag=order
            )
            completed_calls += 1
            if completed_calls % 10 == 0 or completed_calls == total_calls:
                print(f"  [{completed_calls}/{total_calls}] Logged pair {sys_a} vs {sys_b} ({order}) -> Winner: {res['unblinded_winner']}")
            return res

    tasks = []
    for r in records:
        for (sys_a, sys_b) in pairs:
            tasks.append(_evaluate_single(r, sys_a, sys_b, "forward"))
            tasks.append(_evaluate_single(r, sys_a, sys_b, "swapped"))

    start_t = time.perf_counter()
    results = await asyncio.gather(*tasks)
    end_t = time.perf_counter()
    elapsed_s = end_t - start_t
    rpm = (len(results) / elapsed_s) * 60.0

    print(f"\nPILOT COMPLETE in {elapsed_s:.2f}s ({len(results)} calls logged, {rpm:.1f} calls/min).")

    # Group results by (query_id, pair_key)
    grouped = {}
    for res in results:
        key = (res["query_id"], f"{res['system_a']}_vs_{res['system_b']}")
        grouped.setdefault(key, {})[res["order_tag"]] = res

    consistent_count = 0
    pairwise_wins = {s1: {s2: 0.0 for s2 in SYSTEM_KEYS} for s1 in SYSTEM_KEYS}
    criteria_accum = {s: {"correctness": [], "completeness": [], "coherence": []} for s in SYSTEM_KEYS}

    for (qid, pair_key), orders in grouped.items():
        fwd = orders.get("forward")
        swp = orders.get("swapped")
        if fwd and swp:
            if fwd["unblinded_winner"] == swp["unblinded_winner"]:
                consistent_count += 1

            for trial in [fwd, swp]:
                w = trial["unblinded_winner"]
                sa = trial["system_a"]
                sb = trial["system_b"]
                if w == sa: pairwise_wins[sa][sb] += 0.5
                elif w == sb: pairwise_wins[sb][sa] += 0.5
                else:
                    pairwise_wins[sa][sb] += 0.25
                    pairwise_wins[sb][sa] += 0.25

                for s_id, s_scores in trial["scores_by_system"].items():
                    if s_id in criteria_accum:
                        if "correctness" in s_scores: criteria_accum[s_id]["correctness"].append(s_scores["correctness"])
                        if "completeness" in s_scores: criteria_accum[s_id]["completeness"].append(s_scores["completeness"])
                        if "coherence" in s_scores: criteria_accum[s_id]["coherence"].append(s_scores["coherence"])

    agreement_rate = round((consistent_count / len(grouped)) * 100.0, 2)
    extrapolated_2400_s = (2400.0 / len(results)) * elapsed_s
    extrapolated_400_s = (400.0 / len(results)) * elapsed_s

    pilot_summary = {
        "pilot_queries": len(records),
        "total_calls": len(results),
        "elapsed_seconds": round(elapsed_s, 2),
        "effective_rpm": round(rpm, 1),
        "position_swap_agreement_rate_pct": agreement_rate,
        "pairwise_wins_sample": pairwise_wins,
        "criteria_summary_sample": {
            s: {
                "correctness": round(sum(criteria_accum[s]["correctness"]) / max(1, len(criteria_accum[s]["correctness"])), 2),
                "completeness": round(sum(criteria_accum[s]["completeness"]) / max(1, len(criteria_accum[s]["completeness"])), 2),
                "coherence": round(sum(criteria_accum[s]["coherence"]) / max(1, len(criteria_accum[s]["coherence"])), 2)
            }
            for s in SYSTEM_KEYS
        },
        "extrapolated_time_full_2400_min": round(extrapolated_2400_s / 60.0, 1),
        "extrapolated_time_option1_400_min": round(extrapolated_400_s / 60.0, 1)
    }

    with open("results/v2_pilot_pairwise_summary.json", "w", encoding="utf-8") as f:
        json.dump(pilot_summary, f, indent=2)

    print("\n=== PILOT SUMMARY METRICS ===")
    print(f"Position-Swap Agreement Rate: {agreement_rate}% ({consistent_count}/{len(grouped)} pairs)")
    print(f"Measured Throughput: {rpm:.1f} RPM")
    print(f"Extrapolated Full 2,400-Call Time: {pilot_summary['extrapolated_time_full_2400_min']} minutes")
    print(f"Extrapolated Option 1 (400-Call) Time: {pilot_summary['extrapolated_time_option1_400_min']} minutes")
    print("Saved pilot summary to results/v2_pilot_pairwise_summary.json")

if __name__ == "__main__":
    asyncio.run(run_pilot(max_queries=5, concurrency=3))

