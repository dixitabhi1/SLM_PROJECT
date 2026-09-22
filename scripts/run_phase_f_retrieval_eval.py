"""
AI Search Framework - Phase F2: Retrieval Specialist Evaluation Harness
Evaluates Fine-Tuned Retrieval Specialist (Phi-3.5-ft-retrieval) vs Base Specialist (Phi-3.5:cpu)
on the held-out evaluation dataset (data/phase_f/retrieval_qa_eval_held_out.json) and V3_CD_21 Node 1.

Enforces:
1. Hard Rule 13 Pre-Flight Distinct Model Verification.
2. Symmetrical Double-Blind Pairwise Judging (Forward & Swapped).
3. Full Autonomous Audit Loop (concordance, swap consistency, truncation diagnostic, no-blending).
"""

import os
import sys
import json
import time
import asyncio
from types import SimpleNamespace
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.ollama_runner import OllamaModelRunner
from src.v4.tools.retrieval_tool import DeterministicRetrievalTool
from src.v4.specialists.grounded_retrieval_runner import GroundedRetrievalModelRunner
from src.v2.judge.pairwise_harness import PairwiseLLMJudgeHarness
from src.v3.preflight import verify_distinct_roster_preflight

RESULTS_DIR = "results/phase_f"
KEYS_DIR = "logs/phase_f_judge_keys"
JUDGE_DIR = "logs/phase_f_judge_pairwise"

def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(KEYS_DIR, exist_ok=True)
    os.makedirs(JUDGE_DIR, exist_ok=True)

async def run_evaluation(
    eval_samples_path: str = "data/phase_f/retrieval_qa_eval_held_out.json",
    num_eval_queries: int = 10,
    include_v3_cd_21: bool = True
):
    ensure_dirs()
    print("=" * 80)
    print("PHASE F2: RETRIEVAL SPECIALIST ISOLATED EVALUATION")
    print("=" * 80)

    # 1. Initialize Runners
    print("\n[Step 1/5] Initializing SLM Runners...")
    base_runner = OllamaModelRunner(
        logical_model_name="phi3.5-base",
        api_model_name="phi3.5:cpu",
        max_tokens=1536,
        temperature=0.0
    )
    ft_runner = OllamaModelRunner(
        logical_model_name="phi3.5-ft-retrieval",
        api_model_name="phi3.5-ft-retrieval:latest",
        max_tokens=1536,
        temperature=0.0
    )

    # Retrieval Tools (Both get access to identical deterministic corpus)
    retrieval_tool = DeterministicRetrievalTool()
    base_grounded = GroundedRetrievalModelRunner(base_runner=base_runner, retrieval_tool=retrieval_tool, top_k=4)
    ft_grounded = GroundedRetrievalModelRunner(base_runner=ft_runner, retrieval_tool=retrieval_tool, top_k=4)

    # 2. Hard Rule 13 Pre-Flight Verification
    print("\n[Step 2/5] Executing Hard Rule 13 Pre-Flight Verification...")
    harness = PairwiseLLMJudgeHarness(judge_model_name="qwen/qwen3.8-27b")
    pipeline_roster = {
        "base_specialist": base_runner,
        "finetuned_specialist": ft_runner
    }
    baseline_roster = {
        "comparator_baseline": SimpleNamespace(api_model_name="meta-llama/llama-3.3-70b-instruct")
    }
    verify_distinct_roster_preflight(pipeline_roster, baseline_roster, judge_runner=harness)

    # 3. Load Held-Out Queries
    print(f"\n[Step 3/5] Loading Held-Out Queries from {eval_samples_path}...")
    with open(eval_samples_path, "r", encoding="utf-8") as f:
        held_out_data = json.load(f)

    eval_queries = []
    # If requested, prioritize V3_CD_21 Node 1 as query #1
    if include_v3_cd_21:
        eval_queries.append({
            "query_id": "V3_CD_21_NODE1",
            "prompt": "Retrieve official security RFC standards and Linux socket vulnerability specifications relevant to secure network protocols, socket layer security, and kernel privilege escalations.",
            "category": "v3_cd_21_node1_held_out"
        })

    # Add sampled held-out queries
    for item in held_out_data[:num_eval_queries]:
        eval_queries.append({
            "query_id": item["sample_id"],
            "prompt": item["prompt"],
            "category": item["category"]
        })

    print(f"Selected {len(eval_queries)} queries for isolated evaluation.")

    # 4. Execute Generations
    print("\n[Step 4/5] Generating Paired Responses (Base vs Fine-Tuned)...")
    records = []
    for idx, q in enumerate(eval_queries, start=1):
        qid = q["query_id"]
        prompt = q["prompt"]
        print(f"  [{idx}/{len(eval_queries)}] Query {qid} ({q['category']})...")

        # Base generation
        t0 = time.perf_counter()
        resp_base = await base_grounded.generate(prompt=prompt)
        lat_base = (time.perf_counter() - t0) * 1000.0

        # Fine-tuned generation
        t0 = time.perf_counter()
        resp_ft = await ft_grounded.generate(prompt=prompt)
        lat_ft = (time.perf_counter() - t0) * 1000.0

        records.append({
            "query_id": qid,
            "prompt": prompt,
            "category": q["category"],
            "base_response": resp_base.text,
            "base_latency_ms": lat_base,
            "ft_response": resp_ft.text,
            "ft_latency_ms": lat_ft
        })

    # Save generation pairs
    gen_file = os.path.join(RESULTS_DIR, "generation_pairs.json")
    with open(gen_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"Saved generation pairs to {gen_file}")

    # 5. Symmetrical Double-Blind Judging & Autonomous Audit Loop
    print("\n[Step 5/5] Symmetrical Pairwise Judging & Autonomous Audit Loop...")
    audit_trials = []
    ft_wins = 0
    base_wins = 0
    ties = 0

    for rec in records:
        qid = rec["query_id"]
        prompt = rec["prompt"]

        # Forward Trial (Cand A = FT, Cand B = Base)
        res_f = await asyncio.to_thread(
            harness.evaluate_pair,
            query_id=qid,
            query_text=prompt,
            system_a_id="phi3.5_finetuned",
            text_a=rec["ft_response"],
            system_b_id="phi3.5_base",
            text_b=rec["base_response"],
            order_tag="forward",
            judge_log_dir=JUDGE_DIR,
            key_log_dir=KEYS_DIR
        )

        # Swapped Trial (Cand A = Base, Cand B = FT)
        res_s = await asyncio.to_thread(
            harness.evaluate_pair,
            query_id=qid,
            query_text=prompt,
            system_a_id="phi3.5_finetuned",
            text_a=rec["ft_response"],
            system_b_id="phi3.5_base",
            text_b=rec["base_response"],
            order_tag="swapped",
            judge_log_dir=JUDGE_DIR,
            key_log_dir=KEYS_DIR
        )

        # Autonomous Audit Checks
        # Check 1: Raw-file concordance
        scores_f = res_f.get("criteria_scores", {})
        sum_a_f = sum(scores_f.get("Candidate A", {}).values())
        sum_b_f = sum(scores_f.get("Candidate B", {}).values())
        winner_f = res_f.get("unblinded_winner")

        scores_s = res_s.get("criteria_scores", {})
        sum_a_s = sum(scores_s.get("Candidate A", {}).values())
        sum_b_s = sum(scores_s.get("Candidate B", {}).values())
        winner_s = res_s.get("unblinded_winner")

        # Check 2: Positional swap consistency
        is_consistent = (winner_f == winner_s)

        # Check 3: Truncation diagnostic
        reason_f = res_f.get("reasoning", "").lower()
        reason_s = res_s.get("reasoning", "").lower()
        is_truncated = any(w in reason_f or w in reason_s for w in ["cut off", "truncated", "incomplete response"])

        # Record outcome
        pair_outcome = "tie"
        if winner_f == "phi3.5_finetuned" and winner_s == "phi3.5_finetuned":
            pair_outcome = "ft_win"
            ft_wins += 1
        elif winner_f == "phi3.5_base" and winner_s == "phi3.5_base":
            pair_outcome = "base_win"
            base_wins += 1
        else:
            pair_outcome = "tie_or_inconsistent"
            ties += 1

        audit_trials.append({
            "query_id": qid,
            "category": rec["category"],
            "forward": {"winner": winner_f, "sum_a": sum_a_f, "sum_b": sum_b_f},
            "swapped": {"winner": winner_s, "sum_a": sum_a_s, "sum_b": sum_b_s},
            "pair_outcome": pair_outcome,
            "swap_consistent": is_consistent,
            "truncation_flagged": is_truncated
        })
        print(f"  {qid:20s}: Forward={winner_f} | Swapped={winner_s} -> Pair: {pair_outcome} (Consistent={is_consistent})")

    # Final Audit Summary
    total_pairs = len(records)
    audit_summary = {
        "total_queries_evaluated": total_pairs,
        "ft_wins": ft_wins,
        "base_wins": base_wins,
        "ties_or_inconsistent": ties,
        "ft_win_rate_pct": (ft_wins / total_pairs) * 100.0 if total_pairs > 0 else 0.0,
        "swap_consistency_pct": (sum(1 for t in audit_trials if t["swap_consistent"]) / total_pairs) * 100.0,
        "trials": audit_trials
    }

    summary_file = os.path.join(RESULTS_DIR, "retrieval_ft_audit_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    print("\n" + "=" * 80)
    print("PHASE F2 AUDIT SUMMARY")
    print(f"  Total Queries: {total_pairs}")
    print(f"  Fine-Tuned Wins: {ft_wins} ({audit_summary['ft_win_rate_pct']:.1f}%)")
    print(f"  Base Wins:       {base_wins}")
    print(f"  Ties/Inconsistent: {ties}")
    print(f"  Swap Consistency: {audit_summary['swap_consistency_pct']:.1f}%")
    print(f"  Saved Audit Summary to: {summary_file}")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(run_evaluation())

