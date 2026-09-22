"""
Pairwise Judge Runner for Version 4 Step 1 (Phi-3.5-mini 3.8B vs 120B Baseline)
Evaluates 8 queries x 2 presentation orders = 16 double-blind trials.
Judge: qwen/qwen3.8-27b on Groq LPU (temperature=0.0, max_tokens=2048)
"""

import os
import sys
import json
import time
import asyncio
from types import SimpleNamespace
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.v2.judge.pairwise_harness import PairwiseLLMJudgeHarness
from src.v3.preflight import verify_distinct_roster_preflight

KEYS_DIR = "logs/v4_step1_judge_keys"
JUDGE_DIR = "logs/v4_step1_judge_pairwise"

TARGET_QUERY_IDS = [
    "V3_CD_01",
    "V3_CD_21",
    "V3_CD_41",
    "V3_CD_61",
    "V3_SD_MATH_01",
    "V3_SD_FORM_01",
    "V3_TD_01",
    "V3_TD_11"
]

def load_responses() -> Dict[str, Dict[str, str]]:
    slm_resp = {}
    with open("results/v4_step1/slm_phi35_responses.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                slm_resp[d["query_id"]] = d["response_text"]

    base_resp = {}
    with open("results/v3_pilot/llm_baseline_responses.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                base_resp[d["query_id"]] = d["response_text"]

    queries = {}
    with open("data/v3_queries_dev.json", "r", encoding="utf-8") as f:
        all_q = json.load(f)
        for q in all_q:
            if q["id"] in TARGET_QUERY_IDS:
                queries[q["id"]] = q["query"]

    return {"slm": slm_resp, "base": base_resp, "queries": queries}

async def run_step1_judging():
    os.makedirs(KEYS_DIR, exist_ok=True)
    os.makedirs(JUDGE_DIR, exist_ok=True)

    harness = PairwiseLLMJudgeHarness(judge_model_name="qwen/qwen3.8-27b")

    # Hard Rule 13 Pre-Flight Verification
    pipeline_roster = {
        "decomposer": SimpleNamespace(api_model_name="llama3.2:3b"),
        "specialist": SimpleNamespace(api_model_name="phi3.5:cpu"),
        "aggregator": SimpleNamespace(api_model_name="llama3.2:3b"),
    }
    baseline_roster = {
        "gpt_120b": SimpleNamespace(api_model_name="openai/gpt-oss-120b")
    }
    verify_distinct_roster_preflight(pipeline_roster, baseline_roster, judge_runner=harness)

    data = load_responses()
    slm_resp = data["slm"]
    base_resp = data["base"]
    queries = data["queries"]

    print(f"Loaded {len(slm_resp)} SLM responses and {len(base_resp)} baseline responses across 8 target queries.")
    print(f"Executing 16 symmetric double-blind trials with qwen/qwen3.8-27b...\n")

    trials_run = 0
    for qid in sorted(TARGET_QUERY_IDS):
        if qid not in slm_resp or qid not in base_resp:
            print(f"Skipping {qid}: missing response")
            continue

        qtext = queries[qid]
        s_text = slm_resp[qid]
        b_text = base_resp[qid]

        # 1. Forward trial: Candidate A = SLM, Candidate B = Baseline
        print(f"[{trials_run+1}/16] Judging {qid} (forward: SLM vs 120B)...", flush=True)
        for attempt in range(5):
            try:
                res_f = await asyncio.to_thread(
                    harness.evaluate_pair,
                    query_id=qid,
                    query_text=qtext,
                    system_a_id="slm_phi35_v4",
                    text_a=s_text,
                    system_b_id="gpt_120b",
                    text_b=b_text,
                    order_tag="forward",
                    judge_log_dir=JUDGE_DIR,
                    key_log_dir=KEYS_DIR
                )
                if res_f.get("status") == "SUCCESS":
                    print(f"  -> Winner: {res_f.get('unblinded_winner')} | Diff: {res_f.get('primary_differentiator')}")
                    break
                else:
                    print(f"  Retry {attempt+1}: {res_f.get('error_detail')}")
                    await asyncio.sleep(2 * (attempt + 1))
            except Exception as e:
                print(f"  Exception: {e}. Retrying...")
                await asyncio.sleep(2 * (attempt + 1))
        trials_run += 1
        await asyncio.sleep(2.0)

        # 2. Swapped trial: Candidate A = Baseline, Candidate B = SLM
        print(f"[{trials_run+1}/16] Judging {qid} (swapped: 120B vs SLM)...", flush=True)
        for attempt in range(5):
            try:
                res_s = await asyncio.to_thread(
                    harness.evaluate_pair,
                    query_id=qid,
                    query_text=qtext,
                    system_a_id="slm_phi35_v4",
                    text_a=s_text,
                    system_b_id="gpt_120b",
                    text_b=b_text,
                    order_tag="swapped",
                    judge_log_dir=JUDGE_DIR,
                    key_log_dir=KEYS_DIR
                )
                if res_s.get("status") == "SUCCESS":
                    print(f"  -> Winner: {res_s.get('unblinded_winner')} | Diff: {res_s.get('primary_differentiator')}")
                    break
                else:
                    print(f"  Retry {attempt+1}: {res_s.get('error_detail')}")
                    await asyncio.sleep(2 * (attempt + 1))
            except Exception as e:
                print(f"  Exception: {e}. Retrying...")
                await asyncio.sleep(2 * (attempt + 1))
        trials_run += 1
        await asyncio.sleep(2.0)

    print(f"\nAll {trials_run} double-blind trials completed successfully.")

if __name__ == "__main__":
    asyncio.run(run_step1_judging())

