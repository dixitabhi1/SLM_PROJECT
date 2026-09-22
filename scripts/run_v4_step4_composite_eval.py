"""
AI Search Framework - Version 4 Step 4: Composite Evaluation
Culmination of the v4 Isolated-Intervention Protocol.

Combines:
1. True <=5B Model Capacity: Microsoft Phi-3.5-mini-instruct (3.82B)
2. Deterministic Retrieval-Tool Grounding for retrieval_qa specialist
3. Mechanical Code Verification Gate for coding specialist
Held Fixed: Decomposer (llama3.2:3b), Aggregator (llama3.2:3b), Baseline (openai/gpt-oss-120b), Judge (qwen/qwen3.8-27b)

Evaluates all 4 Compound DAG Queries:
- V3_CD_01 (Coupled MDO & convergence proof)
- V3_CD_21 (Security RFC retrieval, Linux socket vulnerabilities, sandboxed socket runtime)
- V3_CD_41 (Thermodynamic diffusion PDE solver, relational Parquet export)
- V3_CD_61 (First-order logic relational invariants, multi-tenant SQL schema, executive compliance)

Features:
- Pre-flight distinct model assertion (Hard Rule 13)
- Double-blind symmetric pairwise judging (8 trials)
- Full raw JSON file-level audit
- Truncation vs genuine quality parity diagnostic
"""

import os
import sys
import json
import time
import asyncio
from types import SimpleNamespace
from typing import Dict, Any, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.ollama_runner import OllamaModelRunner
from src.v4.tools.retrieval_tool import DeterministicRetrievalTool
from src.v4.tools.code_verifier import MechanicalCodeVerifier
from src.v4.specialists.grounded_retrieval_runner import GroundedRetrievalModelRunner
from src.v4.specialists.verified_coding_runner import VerifiedCodingModelRunner
from src.v3.pipeline import SLMPipeline_v3
from src.instrumentation.logger import ExperimentLogger
from src.v2.judge.pairwise_harness import PairwiseLLMJudgeHarness
from src.v3.preflight import verify_distinct_roster_preflight

V4_STEP4_DIR = "results/v4_step4b"
PIPELINE_LOG_DIR = os.path.join(V4_STEP4_DIR, "pipeline_logs")
KEYS_DIR = "logs/v4_step4b_judge_keys"
JUDGE_DIR = "logs/v4_step4b_judge_pairwise"

COMPOUND_QUERY_IDS = ["V3_CD_01", "V3_CD_21", "V3_CD_41", "V3_CD_61"]

def ensure_dirs():
    os.makedirs(V4_STEP4_DIR, exist_ok=True)
    os.makedirs(PIPELINE_LOG_DIR, exist_ok=True)
    os.makedirs(KEYS_DIR, exist_ok=True)
    os.makedirs(JUDGE_DIR, exist_ok=True)

def build_v4_composite_pipeline() -> SLMPipeline_v3:
    """
    Builds the complete v4 composite pipeline combining true <=5B capacity,
    retrieval grounding, and mechanical code verification.
    """
    decomposer_runner = OllamaModelRunner(
        logical_model_name="llama3.2-3b",
        api_model_name="llama3.2:cpu"
    )

    base_phi = OllamaModelRunner(
        logical_model_name="phi3.5-3.8b",
        api_model_name="phi3.5:cpu",
        max_tokens=2048
    )

    # Wire Grounded Retrieval Specialist (retrieval_qa only)
    retrieval_tool = DeterministicRetrievalTool()
    grounded_retrieval_runner = GroundedRetrievalModelRunner(
        base_runner=base_phi,
        retrieval_tool=retrieval_tool,
        top_k=4
    )

    # Wire Mechanically Verified Coding Specialist (coding only)
    code_verifier = MechanicalCodeVerifier(execution_timeout_sec=8.0)
    verified_coding_runner = VerifiedCodingModelRunner(
        base_runner=base_phi,
        verifier=code_verifier,
        max_retries=1
    )

    aggregator_runner = OllamaModelRunner(
        logical_model_name="llama3.2-3b",
        api_model_name="llama3.2:cpu"
    )

    pool_runners = {
        "coding": verified_coding_runner,
        "retrieval_qa": grounded_retrieval_runner,
        "mathematics": base_phi,
        "formal_reasoning": base_phi,
        "science_tech": base_phi,
        "structured_data": base_phi,
        "creative_synthesis": base_phi,
        "systems_ops": base_phi
    }

    logger = ExperimentLogger(log_dir=PIPELINE_LOG_DIR)

    return SLMPipeline_v3(
        decomposer_runner=decomposer_runner,
        pool_runners=pool_runners,
        aggregator_runner=aggregator_runner,
        logger=logger,
        max_depth=3,
        max_concurrent_slms=2
    )

async def run_composite_generation(pipeline: SLMPipeline_v3, queries: List[Dict[str, Any]]) -> Dict[str, str]:
    responses_file = os.path.join(V4_STEP4_DIR, "slm_composite_responses.jsonl")
    completed = {}

    if os.path.exists(responses_file):
        with open(responses_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    completed[d["query_id"]] = d["response_text"]
        print(f"Resuming: {len(completed)} queries already completed in {responses_file}")

    with open(responses_file, "a", encoding="utf-8") as out_f:
        for idx, q in enumerate(queries, 1):
            qid = q["id"]
            if qid in completed:
                print(f"[{idx}/{len(queries)}] Skipping {qid} (already done)")
                continue

            qtext = q["query"]
            ctier = q.get("complexity_tier", "three_plus_domain")
            print(f"\n[{idx}/{len(queries)}] Executing Composite SLM Pipeline for {qid}...")
            t0 = time.perf_counter()
            pipe_res = await pipeline.execute_query(
                query_id=qid,
                query_text=qtext,
                complexity_tier=ctier,
                seed=42,
                config={"version": "4.4.0-composite"}
            )
            lat = time.perf_counter() - t0
            resp_text = pipe_res.get("response", "") or pipe_res.get("final_response", "")
            print(f"  -> Generated {len(resp_text)} chars in {lat:.2f}s")

            entry = {
                "query_id": qid,
                "complexity_tier": ctier,
                "query_text": qtext,
                "system_type": "v4_composite_slm_pipeline",
                "response_text": resp_text,
                "latency_sec": lat,
                "feedback_loop_fired": pipe_res.get("feedback_loop_fired", False),
                "log_path": pipe_res.get("log_path", "")
            }
            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            out_f.flush()
            completed[qid] = resp_text

    return completed

async def run_composite_judging(slm_responses: Dict[str, str], queries: Dict[str, str]):
    base_120b_responses = {}
    with open("results/v3_pilot/llm_baseline_responses.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["query_id"] in COMPOUND_QUERY_IDS:
                base_120b_responses[d["query_id"]] = d["response_text"]

    harness = PairwiseLLMJudgeHarness(judge_model_name="qwen/qwen3.8-27b")

    # Hard Rule 13 Pre-Flight Assertion
    pipeline_roster = {
        "decomposer": SimpleNamespace(api_model_name="llama3.2:cpu"),
        "specialist": SimpleNamespace(api_model_name="phi3.5:cpu"),
        "aggregator": SimpleNamespace(api_model_name="llama3.2:cpu"),
    }
    baseline_roster = {
        "gpt_120b": SimpleNamespace(api_model_name="openai/gpt-oss-120b")
    }
    verify_distinct_roster_preflight(pipeline_roster, baseline_roster, judge_runner=harness)

    print(f"\nExecuting 8 symmetric double-blind trials against 120B baseline with qwen/qwen3.8-27b...")

    trials = []
    trial_count = 0

    for qid in COMPOUND_QUERY_IDS:
        qtext = queries[qid]
        s_text = slm_responses[qid]
        b_text = base_120b_responses[qid]

        # 1. Forward Trial: Cand A = Composite SLM, Cand B = 120B
        trial_count += 1
        print(f"[{trial_count}/8] Judging {qid} (forward: Composite SLM vs 120B)...", flush=True)
        res_f = await asyncio.to_thread(
            harness.evaluate_pair,
            query_id=qid,
            query_text=qtext,
            system_a_id="slm_composite_v4",
            text_a=s_text,
            system_b_id="gpt_120b",
            text_b=b_text,
            order_tag="forward",
            judge_log_dir=JUDGE_DIR,
            key_log_dir=KEYS_DIR
        )
        print(f"  -> Winner: {res_f.get('unblinded_winner')} | Diff: {res_f.get('primary_differentiator')}")
        trials.append(res_f)
        await asyncio.sleep(2.0)

        # 2. Swapped Trial: Cand A = 120B, Cand B = Composite SLM
        trial_count += 1
        print(f"[{trial_count}/8] Judging {qid} (swapped: 120B vs Composite SLM)...", flush=True)
        res_s = await asyncio.to_thread(
            harness.evaluate_pair,
            query_id=qid,
            query_text=qtext,
            system_a_id="slm_composite_v4",
            text_a=s_text,
            system_b_id="gpt_120b",
            text_b=b_text,
            order_tag="swapped",
            judge_log_dir=JUDGE_DIR,
            key_log_dir=KEYS_DIR
        )
        print(f"  -> Winner: {res_s.get('unblinded_winner')} | Diff: {res_s.get('primary_differentiator')}")
        trials.append(res_s)
        await asyncio.sleep(2.0)

    return trials

def perform_raw_audit_and_diagnostics(trials: List[Dict[str, Any]], slm_responses: Dict[str, str]):
    base_120b_responses = {}
    with open("results/v3_pilot/llm_baseline_responses.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["query_id"] in COMPOUND_QUERY_IDS:
                base_120b_responses[d["query_id"]] = d["response_text"]

    print("\n" + "=" * 105)
    print("V4 STEP 4 RAW FILE AUDIT & TRUNCATION DIAGNOSTICS")
    print("=" * 105)

    all_audit_passed = True
    truncation_wins = []

    for idx, t in enumerate(trials, 1):
        qid = t.get("query_id")
        order = t.get("order_tag")
        winner = t.get("unblinded_winner")
        diff = t.get("primary_differentiator")
        scores = t.get("criteria_scores", {})
        cand_a_alias = "Candidate A"
        cand_b_alias = "Candidate B"
        sel_alias = t.get("selected_candidate")
        reasoning = t.get("reasoning", "")

        score_a = scores.get(cand_a_alias, {})
        score_b = scores.get(cand_b_alias, {})
        sum_a = sum(score_a.values()) if score_a else 0
        sum_b = sum(score_b.values()) if score_b else 0

        # Verify selected alias matches higher score
        if sel_alias == cand_a_alias:
            score_match = (sum_a >= sum_b)
        elif sel_alias == cand_b_alias:
            score_match = (sum_b >= sum_a)
        else:
            score_match = (sum_a == sum_b)

        if not score_match:
            all_audit_passed = False

        s_len = len(slm_responses.get(qid, ""))
        b_len = len(base_120b_responses.get(qid, ""))
        is_trunc = "cut off" in reasoning.lower() or "truncated" in reasoning.lower()

        if winner == "slm_composite_v4" and is_trunc:
            truncation_wins.append(qid)

        print(f"[{idx}/8] {qid:12s} | Order: {order:8s} | Winner: {winner:18s} | Scores (A: {sum_a}, B: {sum_b}) | Score Match: {score_match}")
        print(f"     Chars: SLM = {s_len} vs 120B = {b_len} | Judge Truncation Mentioned: {is_trunc}")
        print(f"     Reasoning: {reasoning[:180]}...\n")

    print("-" * 105)
    print(f"Raw File-Level Audit Status: {'PASSED (100% Consistent)' if all_audit_passed else 'FAILED'}")
    print(f"SLM Wins Attributable to Baseline Truncation: {len(truncation_wins)} ({truncation_wins})")

    slm_wins = sum(1 for t in trials if t.get("unblinded_winner") == "slm_composite_v4")
    total_trials = len(trials)
    win_rate = (slm_wins / total_trials) * 100.0 if total_trials > 0 else 0.0
    print("-" * 105)
    print(f"Overall Composite SLM Win Rate: {slm_wins} / {total_trials} ({win_rate:.1f}%)")
    print(f"Per-Query Breakdown:")
    for qid in COMPOUND_QUERY_IDS:
        q_wins = sum(1 for t in trials if t.get("query_id") == qid and t.get("unblinded_winner") == "slm_composite_v4")
        print(f"  {qid}: {q_wins} / 2 wins ({(q_wins / 2.0)*100.0:.1f}%)")
    print("=" * 105)

async def main():
    ensure_dirs()
    print("=" * 105)
    print("STARTING V4 STEP 4 COMPOSITE EVALUATION")
    print("=" * 105)

    with open("data/v3_queries_dev.json", "r", encoding="utf-8") as f:
        all_q = json.load(f)
    queries = [q for q in all_q if q["id"] in COMPOUND_QUERY_IDS]
    query_map = {q["id"]: q["query"] for q in queries}

    pipeline = build_v4_composite_pipeline()
    slm_resp = await run_composite_generation(pipeline, queries)

    trials = await run_composite_judging(slm_resp, query_map)
    perform_raw_audit_and_diagnostics(trials, slm_resp)

if __name__ == "__main__":
    asyncio.run(main())

