"""
v3 Pairwise LLM Judge Benchmark Harness (16 Queries x 4 Baselines x 2 Orders = 128 Trials)
AI Search Framework (Version 3 Architecture)

Evaluates:
  SLMPipeline_v3 (8 Domains <=5B) vs 4 Baselines (Floor >= 30B: Qwen-32B, Llama-70B, Qwen-72B, Gemini-1.5-Pro)
Judge Model:
  qwen/qwen3.8-27b on Groq LPU (temperature=0.0, response_format={"type": "json_object"})
Protocols:
  1. Full double-blind candidate anonymization ("Candidate A", "Candidate B")
  2. Cryptographic key separation (logs/v3_judge_keys/)
  3. Symmetric bidirectional position swapping (forward & swapped)
  4. Robust socket error handling & exponential backoff
"""

import asyncio
import json
import os
import sys
import time
from typing import Dict, List, Any, Tuple, Optional

sys.path.insert(0, os.path.abspath("."))
from types import SimpleNamespace
from src.v2.judge.pairwise_harness import PairwiseLLMJudgeHarness
from src.v3.preflight import verify_distinct_roster_preflight

PILOT_DIR = "results/v3_pilot"
SLM_FILE = os.path.join(PILOT_DIR, "slm_pipeline_responses.jsonl")
BASELINE_FILE = os.path.join(PILOT_DIR, "llm_baseline_responses.jsonl")

JUDGE_LOG_DIR = "logs/v3_judge_pairwise"
KEY_LOG_DIR = "logs/v3_judge_keys"

BASELINES_V3 = ["gpt_120b"]

def load_pilot_records() -> Tuple[Dict[str, str], Dict[str, Dict[str, str]], Dict[str, str]]:
    slm_responses = {}
    if os.path.exists(SLM_FILE):
        with open(SLM_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r.get("status") == "SUCCESS":
                        slm_responses[r["query_id"]] = r["response_text"]

    baseline_responses = {}
    if os.path.exists(BASELINE_FILE):
        with open(BASELINE_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r.get("status") == "SUCCESS":
                        baseline_responses.setdefault(r["query_id"], {})[r["baseline_model_id"]] = r["response_text"]

    query_texts = {}
    comp_file = os.path.join(PILOT_DIR, "comparison.jsonl")
    if os.path.exists(comp_file):
        with open(comp_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    query_texts[r["query_id"]] = r["query_text"]

    return slm_responses, baseline_responses, query_texts

def get_completed_keys() -> set:
    os.makedirs(KEY_LOG_DIR, exist_ok=True)
    completed = set()
    for fname in os.listdir(KEY_LOG_DIR):
        if fname.endswith(".json"):
            fpath = os.path.join(KEY_LOG_DIR, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as fp:
                    d = json.load(fp)
                    if d.get("status") == "SUCCESS":
                        completed.add((d["query_id"], d["candidate_a_system"], d["candidate_b_system"], d["order_tag"]))
            except Exception:
                pass
    return completed

async def run_v3_pairwise_judge(concurrency: int = 1):
    os.makedirs(JUDGE_LOG_DIR, exist_ok=True)
    os.makedirs(KEY_LOG_DIR, exist_ok=True)

    slm_resp, base_resp, q_texts = load_pilot_records()
    common_qids = [qid for qid in slm_resp if qid in base_resp and all(b in base_resp[qid] for b in BASELINES_V3)]

    print(f"=== v3 Pairwise Judge Benchmark ({len(common_qids)} Queries x 1 Baseline (120B) x 2 Orders = {len(common_qids) * 2} Trials) ===")
    print(f"Judge Model: qwen/qwen3.8-27b on Groq (temperature=0.0, max_tokens=2048)")
    print(f"Logging: Public logs -> {JUDGE_LOG_DIR}/ | Key logs -> {KEY_LOG_DIR}/")
    print(f"Target Quality Win Rate: >= 75%\n")

    harness = PairwiseLLMJudgeHarness(judge_model_name="qwen/qwen3.8-27b")

    # Hard Rule 13: Pre-Flight Distinctness Assertion before any judging calls
    pipeline_roster = {
        "decomposer": SimpleNamespace(api_model_name="llama3.2:3b"),
        "coding_specialist": SimpleNamespace(api_model_name="qwen2.5-coder:3b"),
        "math_specialist": SimpleNamespace(api_model_name="deepseek-r1:1.5b"),
        "retrieval_specialist": SimpleNamespace(api_model_name="qwen2.5:1.5b"),
        "aggregator": SimpleNamespace(api_model_name="llama3.2:3b"),
    }
    baseline_roster = {
        "gpt_120b": SimpleNamespace(api_model_name="openai/gpt-oss-120b")
    }
    verify_distinct_roster_preflight(pipeline_roster, baseline_roster, judge_runner=harness)

    completed_keys = get_completed_keys()

    tasks_to_run = []
    for qid in sorted(common_qids):
        qtext = q_texts.get(qid, "")
        text_slm = slm_resp[qid]

        for base_id in BASELINES_V3:
            text_base = base_resp[qid][base_id]

            # Forward: Candidate A = SLM, Candidate B = Baseline
            if (qid, "slm_pipeline_v3", base_id, "forward") not in completed_keys:
                tasks_to_run.append((qid, qtext, "slm_pipeline_v3", text_slm, base_id, text_base, "forward"))

            # Swapped: Candidate A = Baseline, Candidate B = SLM
            if (qid, base_id, "slm_pipeline_v3", "swapped") not in completed_keys:
                tasks_to_run.append((qid, qtext, "slm_pipeline_v3", text_slm, base_id, text_base, "swapped"))

    print(f"Pending Judge Calls: {len(tasks_to_run)} calls to dispatch (Already completed: {len(completed_keys)})")

    semaphore = asyncio.Semaphore(concurrency)
    completed_count = 0
    total_calls = len(tasks_to_run)

    async def _eval_call(item):
        nonlocal completed_count
        qid, qtext, sys_a, text_a, sys_b, text_b, order = item
        async with semaphore:
            for retry_attempt in range(5):
                try:
                    res = await asyncio.to_thread(
                        harness.evaluate_pair,
                        query_id=qid,
                        query_text=qtext,
                        system_a_id=sys_a,
                        text_a=text_a,
                        system_b_id=sys_b,
                        text_b=text_b,
                        order_tag=order,
                        judge_log_dir=JUDGE_LOG_DIR,
                        key_log_dir=KEY_LOG_DIR
                    )
                    if res.get("status") == "SUCCESS":
                        completed_count += 1
                        winner = res.get("unblinded_winner")
                        print(f"  [{completed_count}/{total_calls}] {qid} SLM vs {sys_b} ({order}) -> Winner: {winner}", flush=True)
                        await asyncio.sleep(2.0) # Pacing within Groq TPM limits
                        return res
                    else:
                        print(f"  Warning: call failed ({res.get('error_detail')}). Retrying...", flush=True)
                        await asyncio.sleep(2.0 * (retry_attempt + 1))
                except Exception as e:
                    print(f"  Exception on call: {e}. Retrying...", flush=True)
                    await asyncio.sleep(2.0 * (retry_attempt + 1))

            return {"status": "FAILED", "query_id": qid, "pair": f"{sys_a}_vs_{sys_b}", "order": order}

    if tasks_to_run:
        await asyncio.gather(*[_eval_call(t) for t in tasks_to_run])

    print("\n=== All Pending Calls Dispatched ===")

if __name__ == "__main__":
    asyncio.run(run_v3_pairwise_judge())

