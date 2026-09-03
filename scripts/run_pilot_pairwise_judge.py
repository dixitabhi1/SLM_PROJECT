"""
Pilot Pairwise Judge Harness (Evaluates Real Generated Responses)
Compares SLM Pipeline vs. Each Baseline with Position Swapping (A/B and B/A).
Maintains strict blind public records (logs/judge_pairwise/) and private identity mappings (logs/judge_keys/).
Populates judge_verdict_ref in results/v2_pilot/comparison.jsonl.
"""

import asyncio
import glob
import json
import os
import sys
import time
from typing import Dict, List, Any, Tuple, Optional

sys.path.insert(0, os.path.abspath("."))
from src.v2.judge.pairwise_harness import PairwiseLLMJudgeHarness

PILOT_DIR = "results/v2_pilot"
SLM_FILE = os.path.join(PILOT_DIR, "slm_pipeline_responses.jsonl")
BASELINE_FILE = os.path.join(PILOT_DIR, "llm_baseline_responses.jsonl")
COMPARISON_FILE = os.path.join(PILOT_DIR, "comparison.jsonl")

BASELINE_KEYS = ["llama_8b", "qwen_32b", "llama_70b", "qwen_72b", "gemini_frontier"]
ALL_SYSTEM_KEYS = ["slm_pipeline_v2"] + BASELINE_KEYS

def load_verified_responses() -> Tuple[Dict[str, str], Dict[str, Dict[str, str]], List[Dict[str, Any]]]:
    slm_responses = {}
    if os.path.exists(SLM_FILE):
        with open(SLM_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    if d.get("status") == "SUCCESS":
                        slm_responses[d["query_id"]] = d["response_text"]

    baseline_responses = {}
    if os.path.exists(BASELINE_FILE):
        with open(BASELINE_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    if d.get("status") == "SUCCESS":
                        baseline_responses.setdefault(d["query_id"], {})[d["baseline_model_id"]] = d["response_text"]

    comparison_records = []
    if os.path.exists(COMPARISON_FILE):
        with open(COMPARISON_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    comparison_records.append(json.loads(line))

    return slm_responses, baseline_responses, comparison_records

def find_logged_trial(qid: str, sys_a: str, sys_b: str, order: str) -> Optional[Dict[str, Any]]:
    pair_key = f"{sys_a}_vs_{sys_b}"
    pattern = f"logs/judge_keys/key_{qid}_{pair_key}_{order}_*.json"
    matches = glob.glob(pattern)
    for m in matches:
        try:
            with open(m, "r", encoding="utf-8") as f:
                d = json.load(f)
                if d.get("status") == "SUCCESS" and "unblinded_winner" in d:
                    return {
                        "query_id": qid,
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

async def run_pilot_judge(api_key: Optional[str] = None):
    slm_resp, base_resp, comp_records = load_verified_responses()

    # Find completed queries having both SLM and all 5 baselines
    target_qids = [
        cr["query_id"] for cr in comp_records
        if cr["query_id"] in slm_resp and len(base_resp.get(cr["query_id"], {})) == 5
    ]

    total_planned = len(target_qids) * 10
    print(f"=== Running Real Pilot Pairwise Judge on Completed Queries ===", flush=True)
    print(f"Eligible fully-generated queries: {len(target_qids)} ({target_qids})", flush=True)
    print(f"Judge Model: qwen/qwen3.8-27b on Groq (Non-reasoning dense evaluator)", flush=True)
    print(f"Comparisons per query: 5 baselines x 2 orders (forward & swapped) = 10 calls/query", flush=True)
    print(f"Total planned judge calls: {total_planned}\n", flush=True)
    sys.stdout.flush()

    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key and os.path.exists(".env"):
            with open(".env", "r") as f:
                for l in f:
                    if l.startswith("GROQ_API_KEY="):
                        api_key = l.strip().split("=", 1)[1]
                        break

    harness = PairwiseLLMJudgeHarness(judge_model_name="qwen/qwen3.8-27b", api_key=api_key)

    all_verdicts_by_query = {}
    calls_made = 0
    start_t = time.perf_counter()

    for qid in target_qids:
        query_text = ""
        with open("results/v2_eval_dev_master.jsonl", "r", encoding="utf-8") as f:
            for l in f:
                d = json.loads(l)
                if d.get("query_id") == qid:
                    query_text = d.get("query_text", "")
                    break

        text_slm = slm_resp[qid]
        all_verdicts_by_query[qid] = {}

        for b_id in BASELINE_KEYS:
            text_baseline = base_resp[qid][b_id]

            # Forward order (Candidate A = SLM, Candidate B = Baseline)
            res_fwd = find_logged_trial(qid, "slm_pipeline_v2", b_id, "forward")
            if not res_fwd:
                res_fwd = await asyncio.to_thread(
                    harness.evaluate_pair,
                    query_id=qid,
                    query_text=query_text,
                    system_a_id="slm_pipeline_v2",
                    text_a=text_slm,
                    system_b_id=b_id,
                    text_b=text_baseline,
                    order_tag="forward"
                )
                await asyncio.sleep(1.8)
            calls_made += 1

            # Swapped order (Candidate A = Baseline, Candidate B = SLM)
            res_swp = find_logged_trial(qid, "slm_pipeline_v2", b_id, "swapped")
            if not res_swp:
                res_swp = await asyncio.to_thread(
                    harness.evaluate_pair,
                    query_id=qid,
                    query_text=query_text,
                    system_a_id="slm_pipeline_v2",
                    text_a=text_slm,
                    system_b_id=b_id,
                    text_b=text_baseline,
                    order_tag="swapped"
                )
                await asyncio.sleep(1.8)
            calls_made += 1

            all_verdicts_by_query[qid][b_id] = {
                "forward": res_fwd,
                "swapped": res_swp
            }
            print(f"[{calls_made}/{total_planned}] Evaluated {qid}: SLM vs {b_id:15s} | Fwd: {res_fwd.get('unblinded_winner')} | Swp: {res_swp.get('unblinded_winner')}", flush=True)
            sys.stdout.flush()

    total_time = time.perf_counter() - start_t
    print(f"\nAll {calls_made} judge calls completed in {total_time:.2f}s ({total_time/60.0:.2f} min).", flush=True)
    sys.stdout.flush()

    # Update comparison.jsonl with judge_verdict_ref
    updated_comp = []
    for cr in comp_records:
        qid = cr["query_id"]
        if qid in all_verdicts_by_query:
            cr["judge_verdict_ref"] = {
                "evaluated_pairs": list(all_verdicts_by_query[qid].keys()),
                "total_trials": len(all_verdicts_by_query[qid]) * 2,
                "summary": {
                    b_id: {
                        "forward_winner": all_verdicts_by_query[qid][b_id]["forward"].get("unblinded_winner"),
                        "swapped_winner": all_verdicts_by_query[qid][b_id]["swapped"].get("unblinded_winner"),
                        "forward_public_log": all_verdicts_by_query[qid][b_id]["forward"].get("public_log"),
                        "swapped_public_log": all_verdicts_by_query[qid][b_id]["swapped"].get("public_log")
                    }
                    for b_id in all_verdicts_by_query[qid]
                }
            }
        updated_comp.append(cr)

    with open(COMPARISON_FILE, "w", encoding="utf-8") as f:
        for cr in updated_comp:
            f.write(json.dumps(cr) + "\n")

    # -------------------------------------------------------------
    # Compute Statistics: Win Rates, Criteria Scores, Agreement Rate
    # -------------------------------------------------------------
    slm_wins = {b: 0.0 for b in BASELINE_KEYS}
    slm_losses = {b: 0.0 for b in BASELINE_KEYS}
    slm_ties = {b: 0.0 for b in BASELINE_KEYS}
    agreement_matches = 0
    total_unique_pairs = 0

    criteria_accum = {s: {"correctness": [], "completeness": [], "coherence": []} for s in ALL_SYSTEM_KEYS}

    for qid, pairs in all_verdicts_by_query.items():
        for b_id, pair_trials in pairs.items():
            fwd = pair_trials["forward"]
            swp = pair_trials["swapped"]
            total_unique_pairs += 1

            if fwd.get("unblinded_winner") == swp.get("unblinded_winner"):
                agreement_matches += 1

            for trial in [fwd, swp]:
                w = trial.get("unblinded_winner")
                if w == "slm_pipeline_v2":
                    slm_wins[b_id] += 0.5
                elif w == b_id:
                    slm_losses[b_id] += 0.5
                else:
                    slm_ties[b_id] += 0.5

                for s_id, scores in trial.get("scores_by_system", {}).items():
                    if s_id in criteria_accum:
                        if "correctness" in scores: criteria_accum[s_id]["correctness"].append(scores["correctness"])
                        if "completeness" in scores: criteria_accum[s_id]["completeness"].append(scores["completeness"])
                        if "coherence" in scores: criteria_accum[s_id]["coherence"].append(scores["coherence"])

    n_q = len(target_qids)
    agreement_pct = round((agreement_matches / max(1, total_unique_pairs)) * 100.0, 2)
    slm_win_rates = {b: round((slm_wins[b] / max(1, n_q)) * 100.0, 2) for b in BASELINE_KEYS}

    criteria_summary = {
        s: {
            "mean_correctness": round(sum(criteria_accum[s]["correctness"]) / max(1, len(criteria_accum[s]["correctness"])), 3),
            "mean_completeness": round(sum(criteria_accum[s]["completeness"]) / max(1, len(criteria_accum[s]["completeness"])), 3),
            "mean_coherence": round(sum(criteria_accum[s]["coherence"]) / max(1, len(criteria_accum[s]["coherence"])), 3)
        }
        for s in ALL_SYSTEM_KEYS
    }

    report_payload = {
        "benchmark": "v2_pilot_real_pairwise_judge_benchmark",
        "scope": f"Verified evaluation over {n_q} fully completed single-domain queries ({target_qids})",
        "judge_model": "qwen/qwen3.8-27b (Groq API)",
        "total_judge_calls": calls_made,
        "position_swap_agreement_rate_pct": agreement_pct,
        "slm_head_to_head_win_rates_pct": slm_win_rates,
        "slm_pairwise_breakdown": {
            b: {
                "slm_wins": slm_wins[b],
                "baseline_wins": slm_losses[b],
                "ties": slm_ties[b],
                "slm_win_rate_pct": slm_win_rates[b]
            }
            for b in BASELINE_KEYS
        },
        "criteria_performance": criteria_summary
    }

    result_out = os.path.join(PILOT_DIR, "pilot_verified_judge_results.json")
    with open(result_out, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    print("\n" + "="*60, flush=True)
    print(f"VERIFIED PILOT RESULTS (N = {n_q} Single-Domain Queries)", flush=True)
    print("="*60, flush=True)
    print(f"Position-Swap Agreement Rate: {agreement_pct}%", flush=True)
    print("\nSLM Pipeline Head-to-Head Win Rates:", flush=True)
    for b in BASELINE_KEYS:
        print(f"  - vs {b:16s}: {slm_win_rates[b]}% (Wins: {slm_wins[b]}, Losses: {slm_losses[b]}, Ties: {slm_ties[b]})", flush=True)
    print("\nCriteria Performance Breakdown (1-5 Scale):", flush=True)
    for s in ALL_SYSTEM_KEYS:
        cs = criteria_summary[s]
        print(f"  - {s:18s}: Correctness={cs['mean_correctness']:.3f}, Completeness={cs['mean_completeness']:.3f}, Coherence={cs['mean_coherence']:.3f}", flush=True)
    print(f"\nArtifacts Saved:", flush=True)
    print(f"  - {result_out}", flush=True)
    print(f"  - Updated {COMPARISON_FILE}", flush=True)
    sys.stdout.flush()

if __name__ == "__main__":
    asyncio.run(run_pilot_judge())

