"""
AI Search Framework - Version 4 Step 3: Mechanical Verification Gate Evaluation
Evaluates deterministic AST syntax checking and sandboxed execution loops for the coding specialist on:
1. V3_CD_21 Node 3 (Sandboxed socket runtime)
2. V3_CD_41 Node 2 (Laplacian matrix PDE solver)
3. V3_CD_41 Node 3 (Relational parquet export with physical invariants)

Measures:
- Syntax Pass Rate (before vs after verification gate)
- Subprocess Execution Pass Rate (before vs after verification gate)
- Single-Retry Self-Correction Success Rate
- Pairwise Double-Blind Judging against 120B Baseline with Raw JSON Audit
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
from src.v4.tools.code_verifier import MechanicalCodeVerifier
from src.v4.specialists.verified_coding_runner import VerifiedCodingModelRunner
from src.v2.judge.pairwise_harness import PairwiseLLMJudgeHarness
from src.v3.preflight import verify_distinct_roster_preflight

V4_STEP3_DIR = "results/v4_step3"
KEYS_DIR = "logs/v4_step3_judge_keys"
JUDGE_DIR = "logs/v4_step3_judge_pairwise"

CODING_TASKS = [
    {
        "task_id": "V3_CD_21_NODE3",
        "query_id": "V3_CD_21",
        "title": "Sandboxed Python Socket Runtime",
        "prompt": (
            "Implement a sandboxed Python runtime for socket communication and network protocol testing. "
            "Write a self-contained, executable Python script demonstrating socket creation, proper security options "
            "(such as SO_REUSEADDR and timeout handling), and secure error handling. The code must be runnable without external dependencies."
        )
    },
    {
        "task_id": "V3_CD_41_NODE2",
        "query_id": "V3_CD_41",
        "title": "Laplacian PDE Matrix Solver",
        "prompt": (
            "Solve partial differential matrices for physical thermodynamic diffusion using numerical finite differences in Python. "
            "Construct a 1D or 2D discrete Laplacian matrix using numpy/scipy, implement an explicit or implicit Euler time-stepping loop, "
            "and simulate heat diffusion over 50 time steps. Provide a self-contained, runnable Python script."
        )
    },
    {
        "task_id": "V3_CD_41_NODE3",
        "query_id": "V3_CD_41",
        "title": "Relational Parquet Dataset Export",
        "prompt": (
            "Write a self-contained Python script to export thermodynamic simulation results into a structured dataframe and write to Parquet. "
            "Generate synthetic thermodynamic grid data in-memory (do not read external files), compute summary statistics, "
            "and export to a local parquet or CSV buffer using pandas/pyarrow. Ensure the script executes cleanly."
        )
    }
]

def ensure_dirs():
    os.makedirs(V4_STEP3_DIR, exist_ok=True)
    os.makedirs(KEYS_DIR, exist_ok=True)
    os.makedirs(JUDGE_DIR, exist_ok=True)

async def run_step3_eval():
    ensure_dirs()
    print("=" * 90)
    print("V4 STEP 3: MECHANICAL CODE VERIFICATION GATE EVALUATION")
    print("=" * 90)

    base_runner = OllamaModelRunner(
        logical_model_name="phi3.5-3.8b",
        api_model_name="phi3.5:cpu",
        max_tokens=2048
    )

    verifier = MechanicalCodeVerifier(execution_timeout_sec=8.0)
    verified_runner = VerifiedCodingModelRunner(
        base_runner=base_runner,
        verifier=verifier,
        max_retries=1
    )

    # Load 120B baseline responses for reference
    base_120b_texts = {}
    with open("results/v3_pilot/llm_baseline_responses.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            base_120b_texts[d["query_id"]] = d["response_text"]

    harness = PairwiseLLMJudgeHarness(judge_model_name="qwen/qwen3.8-27b")

    # Hard Rule 13 Pre-Flight Assertion
    pipeline_roster = {
        "coding_specialist": SimpleNamespace(api_model_name="phi3.5:cpu")
    }
    baseline_roster = {
        "gpt_120b": SimpleNamespace(api_model_name="openai/gpt-oss-120b")
    }
    verify_distinct_roster_preflight(pipeline_roster, baseline_roster, judge_runner=harness)

    records = []
    trials = []

    for idx, item in enumerate(CODING_TASKS, 1):
        tid = item["task_id"]
        qid = item["query_id"]
        prompt = item["prompt"]
        title = item["title"]

        print(f"\n--- [{idx}/{len(CODING_TASKS)}] Testing Task: {tid} ({title}) ---")

        # 1. Unverified Generation
        print("  [1/4] Running Unverified Coding Specialist...")
        t0 = time.perf_counter()
        resp_unverified = await base_runner.generate(prompt=prompt)
        dur_unv = time.perf_counter() - t0
        v_unv = verifier.verify_full(resp_unverified.text)
        print(f"    Ungated Output: {len(resp_unverified.text)} chars in {dur_unv:.2f}s | Syntax: {v_unv['syntax_valid']} | Exec: {v_unv['execution_valid']}")

        # 2. Mechanically Verified Generation
        print("  [2/4] Running Mechanically Verified Specialist (Gate + 1 Retry)...")
        t0 = time.perf_counter()
        resp_verified = await verified_runner.generate(prompt=prompt)
        dur_ver = time.perf_counter() - t0
        v_ver = verifier.verify_full(resp_verified.text)
        status = resp_verified.metadata.get("verification_status", "UNKNOWN")
        attempts = resp_verified.metadata.get("attempts_count", 1)
        print(f"    Gated Output: {len(resp_verified.text)} chars in {dur_ver:.2f}s | Status: {status} (attempts={attempts}) | Exec: {v_ver['execution_valid']}")

        task_record = {
            "task_id": tid,
            "query_id": qid,
            "title": title,
            "prompt": prompt,
            "unverified": {
                "text": resp_unverified.text,
                "syntax_valid": v_unv.get("syntax_valid", False),
                "execution_valid": v_unv.get("execution_valid", False),
                "error": v_unv.get("error_message"),
                "latency_s": dur_unv
            },
            "verified": {
                "text": resp_verified.text,
                "status": status,
                "attempts": attempts,
                "syntax_valid": v_ver.get("syntax_valid", False),
                "execution_valid": v_ver.get("execution_valid", False),
                "error": v_ver.get("error_message"),
                "latency_s": dur_ver
            }
        }
        records.append(task_record)

        # 3. Double-Blind Pairwise Judging against 120B Baseline
        base_text = base_120b_texts.get(qid, "No baseline response found.")

        print(f"  [3/4] Judging Forward Trial (Verified SLM vs 120B)...", flush=True)
        res_f = await asyncio.to_thread(
            harness.evaluate_pair,
            query_id=tid,
            query_text=prompt,
            system_a_id="slm_verified_phi35",
            text_a=resp_verified.text,
            system_b_id="gpt_120b",
            text_b=base_text,
            order_tag="forward",
            judge_log_dir=JUDGE_DIR,
            key_log_dir=KEYS_DIR
        )
        print(f"    Forward Winner: {res_f.get('unblinded_winner')} | Diff: {res_f.get('primary_differentiator')}")
        trials.append(res_f)
        await asyncio.sleep(2.0)

        print(f"  [4/4] Judging Swapped Trial (120B vs Verified SLM)...", flush=True)
        res_s = await asyncio.to_thread(
            harness.evaluate_pair,
            query_id=tid,
            query_text=prompt,
            system_a_id="slm_verified_phi35",
            text_a=resp_verified.text,
            system_b_id="gpt_120b",
            text_b=base_text,
            order_tag="swapped",
            judge_log_dir=JUDGE_DIR,
            key_log_dir=KEYS_DIR
        )
        print(f"    Swapped Winner: {res_s.get('unblinded_winner')} | Diff: {res_s.get('primary_differentiator')}")
        trials.append(res_s)
        await asyncio.sleep(2.0)

    # Save summary records
    summary_path = os.path.join(V4_STEP3_DIR, "step3_verification_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"\nStep 3 records saved to {summary_path}")

    # Print Comparison Table
    print("\n" + "=" * 90)
    print("STEP 3 MECHANICAL VERIFICATION RESULTS SUMMARY")
    print("=" * 90)
    print(f"{'Task ID':16s} | {'Unver Syntax':12s} | {'Unver Exec':10s} | {'Gated Status':22s} | {'Gated Exec':10s}")
    print("-" * 90)
    for r in records:
        u_syn = str(r["unverified"]["syntax_valid"])
        u_exc = str(r["unverified"]["execution_valid"])
        g_st = r["verified"]["status"]
        g_exc = str(r["verified"]["execution_valid"])
        print(f"{r['task_id']:16s} | {u_syn:12s} | {u_exc:10s} | {g_st:22s} | {g_exc:10s}")

    print("\n" + "=" * 90)
    print("PAIRWISE JUDGING RESULTS AGAINST 120B BASELINE (6 Trials)")
    print("=" * 90)
    for t in trials:
        print(f"Task: {t.get('query_id', 'unknown'):16s} | Order: {t.get('order_tag', 'unknown'):8s} | Winner: {t.get('unblinded_winner', 'unknown'):20s} | Diff: {t.get('primary_differentiator', 'unknown')}")

if __name__ == "__main__":
    asyncio.run(run_step3_eval())

