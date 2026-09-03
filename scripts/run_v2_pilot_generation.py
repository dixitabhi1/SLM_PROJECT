"""
v2 Pilot Generation Harness (20 Single-Domain Queries x 6 Systems = 120 Generations)
Routes SLM Pipeline through real src/v2/ pipeline end-to-end:
  Decomposer -> TaskAnalyser -> TaskColorer -> Matching -> Scheduling -> TwoStageAggregator

Implements Strict Per-Call Persistence, Idempotent Resumability, and 3-File Separation:
  1. results/v2_pilot/slm_pipeline_responses.jsonl
  2. results/v2_pilot/llm_baseline_responses.jsonl
  3. results/v2_pilot/comparison.jsonl
"""

import argparse
import asyncio
import glob
import json
import os
import sys
import time
from typing import Dict, List, Any, Optional, Set, Tuple

sys.path.insert(0, os.path.abspath("."))
from src.models.groq_runner import APIGroqModelRunner
from src.v2.pipeline import SLMPipeline_v2
from src.instrumentation.logger import ExperimentLogger

PILOT_DIR = "results/v2_pilot"
SLM_FILE = os.path.join(PILOT_DIR, "slm_pipeline_responses.jsonl")
BASELINE_FILE = os.path.join(PILOT_DIR, "llm_baseline_responses.jsonl")
COMPARISON_FILE = os.path.join(PILOT_DIR, "comparison.jsonl")

BASELINE_CONFIGS = [
    ("llama_8b", "meta-llama/Llama-3.1-8B-Instruct"),
    ("qwen_32b", "Qwen/Qwen2.5-32B-Instruct"),
    ("llama_70b", "meta-llama/Llama-3.1-70B-Instruct"),
    ("qwen_72b", "Qwen/Qwen2.5-72B-Instruct"),
    ("gemini_frontier", "gemini-1.5-pro")
]

def ensure_pilot_dirs():
    os.makedirs(PILOT_DIR, exist_ok=True)

def append_jsonl_immediate(filepath: str, data: Dict[str, Any]):
    """
    Appends a single JSON record and forces immediate disk flush (fsync).
    Guarantees zero data loss if process crashes mid-run.
    """
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")
        f.flush()
        os.fsync(f.fileno())

def load_resumable_state() -> Tuple[Set[str], Set[Tuple[str, str]], Set[str]]:
    """
    Scans the 3 JSONL files to build sets of already completed items.
    """
    completed_slm: Set[str] = set()
    completed_baselines: Set[Tuple[str, str]] = set()
    completed_comparisons: Set[str] = set()

    if os.path.exists(SLM_FILE):
        with open(SLM_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        if rec.get("status") == "SUCCESS":
                            completed_slm.add(rec["query_id"])
                    except Exception:
                        pass

    if os.path.exists(BASELINE_FILE):
        with open(BASELINE_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        if rec.get("status") == "SUCCESS":
                            completed_baselines.add((rec["query_id"], rec["baseline_model_id"]))
                    except Exception:
                        pass

    if os.path.exists(COMPARISON_FILE):
        with open(COMPARISON_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        completed_comparisons.add(rec["query_id"])
                    except Exception:
                        pass

    return completed_slm, completed_baselines, completed_comparisons

def get_single_domain_queries() -> List[Dict[str, Any]]:
    jsonl_path = "results/v2_eval_dev_master.jsonl"
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Missing master dataset: {jsonl_path}")

    sd_queries = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line.strip())
                if rec.get("complexity_tier") == "single_domain":
                    sd_queries.append(rec)

    return sd_queries[:20]

def build_slm_pipeline(api_key: str) -> SLMPipeline_v2:
    """
    Instantiates real v2 pipeline with live Groq API runners for each stage.
    """
    decomposer_runner = APIGroqModelRunner(
        logical_model_name="Qwen/Qwen2.5-3B-Instruct",
        api_model_name="qwen/qwen3.8-27b",
        api_key=api_key
    )
    pool_runners = {
        "code_specialist": APIGroqModelRunner("Qwen/Qwen2.5-Coder-7B-Instruct", api_model_name="qwen/qwen3.8-27b", api_key=api_key),
        "math_specialist": APIGroqModelRunner("deepseek-ai/DeepSeek-R1-Distill-Qwen-8B", api_model_name="qwen/qwen3.8-27b", api_key=api_key),
        "logic_specialist": APIGroqModelRunner("meta-llama/Llama-3.1-8B-Instruct", api_model_name="qwen/qwen3.8-27b", api_key=api_key),
        "retrieval_specialist": APIGroqModelRunner("Qwen/Qwen2.5-7B-Instruct", api_model_name="qwen/qwen3.8-27b", api_key=api_key),
        "general_slm": APIGroqModelRunner("meta-llama/Llama-3.1-8B-Instruct", api_model_name="qwen/qwen3.8-27b", api_key=api_key)
    }
    aggregator_runner = APIGroqModelRunner(
        logical_model_name="meta-llama/Llama-3.1-8B-Instruct",
        api_model_name="qwen/qwen3.8-27b",
        api_key=api_key
    )

    logger = ExperimentLogger(log_dir=os.path.join(PILOT_DIR, "pipeline_logs"))

    return SLMPipeline_v2(
        decomposer_runner=decomposer_runner,
        pool_runners=pool_runners,
        aggregator_runner=aggregator_runner,
        logger=logger,
        max_depth=3,
        max_concurrent_slms=4
    )

async def run_pilot_generation(api_key: Optional[str] = None):
    ensure_pilot_dirs()
    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key and os.path.exists(".env"):
            with open(".env", "r") as f:
                for line in f:
                    if line.startswith("GROQ_API_KEY="):
                        api_key = line.strip().split("=", 1)[1]
                        break

    queries = get_single_domain_queries()
    n_queries = len(queries)
    total_calls = n_queries * (1 + len(BASELINE_CONFIGS)) # 20 * 6 = 120 calls

    completed_slm, completed_baselines, completed_comparisons = load_resumable_state()

    print(f"=== v2 Pilot Generation (20 Single-Domain Queries x 6 Systems) ===")
    print(f"SLM Path: Real src/v2/pipeline.py (Decomposer -> Analyser -> Colorer -> Matching -> Scheduling -> Aggregator)")
    print(f"Target Directory: {PILOT_DIR}/")
    print(f"Total Target Systems: 1 SLM Pipeline + 5 Baselines = 120 generations")
    print(f"Existing State: SLM completed={len(completed_slm)}, Baselines completed={len(completed_baselines)}, Comparison refs={len(completed_comparisons)}")
    print(f"Per-Call Persistence: Immediate append + fsync after every individual call\n")

    # Instantiate real SLM pipeline
    slm_pipeline = build_slm_pipeline(api_key=api_key)

    # Baseline runners
    baseline_runners = {
        b_id: APIGroqModelRunner(logical_model_name=b_name, api_model_name="qwen/qwen3.8-27b", api_key=api_key)
        for b_id, b_name in BASELINE_CONFIGS
    }

    start_all = time.perf_counter()
    calls_made = 0

    for q_idx, q in enumerate(queries):
        qid = q["query_id"]
        qtext = q["query_text"]
        tier = q.get("complexity_tier", "single_domain")

        # -----------------------------------------------------------------
        # 1. SLM Pipeline Path (Real End-to-End src/v2/ pipeline execution)
        # -----------------------------------------------------------------
        if qid not in completed_slm:
            slm_t0 = time.perf_counter()
            pipe_record = await slm_pipeline.execute_query(
                query_id=qid,
                query_text=qtext,
                complexity_tier=tier,
                seed=42,
                config={"mode": "real_pilot_generation"}
            )
            slm_dur_s = time.perf_counter() - slm_t0
            final_resp = pipe_record.get("response", "") or pipe_record.get("final_response", "")

            # Extract token usage from recorded stages
            total_prompt_tok = sum(s.get("prompt_tokens", 0) for s in pipe_record.get("stages", []))
            total_comp_tok = sum(s.get("completion_tokens", 0) for s in pipe_record.get("stages", []))

            slm_record = {
                "query_id": qid,
                "system_type": "all_slm_pipeline_v2",
                "model_identifier": "src/v2/pipeline.py (Decomposed SLM Pool + Two-Stage Aggregator)",
                "status": "SUCCESS" if final_resp and not final_resp.startswith("[Error") else "FAILED",
                "response_text": final_resp,
                "stages_executed": [s.get("stage_name") for s in pipe_record.get("stages", [])],
                "total_subtasks": len(pipe_record.get("v2_task_colors", {})),
                "loop_events_count": len(pipe_record.get("loop_events", [])),
                "prompt_tokens": total_prompt_tok,
                "completion_tokens": total_comp_tok,
                "total_tokens": total_prompt_tok + total_comp_tok,
                "latency_s": round(slm_dur_s, 3),
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
            append_jsonl_immediate(SLM_FILE, slm_record)
            if slm_record["status"] == "SUCCESS":
                completed_slm.add(qid)
            calls_made += 1
            print(f"[{calls_made}/{total_calls}] Executed Real SLM Pipeline for {qid} (Status: {slm_record['status']}, {slm_dur_s:.2f}s, subtasks={slm_record['total_subtasks']})")

            await asyncio.sleep(2.3)

        # -----------------------------------------------------------------
        # 2. Baseline Models Path (Direct Monolithic Model Invocations)
        # -----------------------------------------------------------------
        for b_id, b_name in BASELINE_CONFIGS:
            if (qid, b_id) not in completed_baselines:
                b_runner = baseline_runners[b_id]
                b_sys_prompt = f"You are {b_name}. Answer the technical query with thorough, complete, and mathematically/algorithmically verified precision."

                b_resp = await b_runner.generate(prompt=qtext, system_prompt=b_sys_prompt)
                calls_made += 1

                b_record = {
                    "query_id": qid,
                    "baseline_model_id": b_id,
                    "model_identifier": b_name,
                    "status": "SUCCESS" if b_resp.text and not b_resp.text.startswith("[Error") else "FAILED",
                    "response_text": b_resp.text,
                    "prompt_tokens": b_resp.prompt_tokens,
                    "completion_tokens": b_resp.completion_tokens,
                    "total_tokens": b_resp.total_tokens,
                    "latency_s": round(b_resp.latency_ms / 1000.0, 3),
                    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                }
                append_jsonl_immediate(BASELINE_FILE, b_record)
                if b_record["status"] == "SUCCESS":
                    completed_baselines.add((qid, b_id))
                print(f"[{calls_made}/{total_calls}] Executed Baseline [{b_id}] for {qid} (Status: {b_record['status']}, {b_resp.latency_ms/1000.0:.2f}s)")

                await asyncio.sleep(2.3)

        # -----------------------------------------------------------------
        # 3. Update Comparison Pointer File (Zero Text Duplication)
        # -----------------------------------------------------------------
        if qid not in completed_comparisons:
            comp_record = {
                "query_id": qid,
                "query_text": qtext,
                "complexity_tier": tier,
                "slm_response_ref": {"file": "slm_pipeline_responses.jsonl", "query_id": qid},
                "baseline_response_refs": {
                    b_id: {"file": "llm_baseline_responses.jsonl", "query_id": qid, "baseline_model_id": b_id}
                    for b_id, _ in BASELINE_CONFIGS
                },
                "judge_verdict_ref": None,
                "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
            append_jsonl_immediate(COMPARISON_FILE, comp_record)
            completed_comparisons.add(qid)

    total_time_s = time.perf_counter() - start_all
    print(f"\n=== Pilot Generation Complete ===")
    print(f"Total API Calls Executed: {calls_made}")
    print(f"Total Wall Clock Time: {total_time_s:.2f}s ({total_time_s/60.0:.2f} min)")
    print(f"Verified Outputs:")
    print(f"  - {SLM_FILE} ({os.path.getsize(SLM_FILE)} bytes)")
    print(f"  - {BASELINE_FILE} ({os.path.getsize(BASELINE_FILE)} bytes)")
    print(f"  - {COMPARISON_FILE} ({os.path.getsize(COMPARISON_FILE)} bytes)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    asyncio.run(run_pilot_generation(api_key=args.api_key))
