"""
v3 Pilot Generation Harness (16 Stratified Queries x 5 Systems = 80 Generations)
AI Search Framework (Version 3 Architecture)

Systems:
- 1 All-SLM Pipeline v3 (8-Specialist Pool, all <=5B, Fixes 1-3 active)
- 4 Monolithic Baselines (Floor >= 30B: Qwen-32B, Llama-70B, Qwen-72B, Gemini-1.5-Pro)
  * Llama-3.1-8B dropped per mentor constraint 3.

3-File Disk Persistence with Immediate fsync:
1. results/v3_pilot/slm_pipeline_responses.jsonl
2. results/v3_pilot/llm_baseline_responses.jsonl
3. results/v3_pilot/comparison.jsonl
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
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.models.groq_runner import APIGroqModelRunner
from src.models.ollama_runner import OllamaModelRunner
from src.v3.pipeline import SLMPipeline_v3
from src.v3.preflight import verify_distinct_roster_preflight
from src.instrumentation.logger import ExperimentLogger

PILOT_DIR = "results/v3_pilot"
SLM_FILE = os.path.join(PILOT_DIR, "slm_pipeline_responses.jsonl")
BASELINE_FILE = os.path.join(PILOT_DIR, "llm_baseline_responses.jsonl")
COMPARISON_FILE = os.path.join(PILOT_DIR, "comparison.jsonl")

# Finalized Single Monolithic Baseline (Floor >= 30B: openai/gpt-oss-120b)
BASELINE_CONFIGS_V3 = [
    ("gpt_120b", "openai/gpt-oss-120b")
]

def ensure_pilot_dirs():
    os.makedirs(PILOT_DIR, exist_ok=True)
    os.makedirs(os.path.join(PILOT_DIR, "pipeline_logs"), exist_ok=True)

def append_jsonl_immediate(filepath: str, data: Dict[str, Any]):
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")
        f.flush()
        os.fsync(f.fileno())

def load_resumable_state() -> Tuple[Set[str], Set[Tuple[str, str]], Set[str]]:
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

def get_v3_pilot_queries() -> List[Dict[str, Any]]:
    dev_path = "data/v3_queries_dev.json"
    if not os.path.exists(dev_path):
        raise FileNotFoundError(f"Missing dev dataset: {dev_path}")

    with open(dev_path, "r", encoding="utf-8") as f:
        dev_queries = json.load(f)

    selected_ids = [
        # 8 Single-Domain (1 per domain)
        "V3_SD_CODI_01", "V3_SD_MATH_01", "V3_SD_FORM_01", "V3_SD_RETR_01",
        "V3_SD_SCIE_01", "V3_SD_STRU_01", "V3_SD_CREA_01", "V3_SD_SYST_01",
        # 4 Two-Domain Compound
        "V3_TD_01", "V3_TD_11", "V3_TD_21", "V3_TD_31",
        # 4 Three-Plus-Domain Compound
        "V3_CD_01", "V3_CD_21", "V3_CD_41", "V3_CD_61"
    ]

    query_map = {q["id"]: q for q in dev_queries}
    pilot_queries = []
    for qid in selected_ids:
        if qid in query_map:
            pilot_queries.append(query_map[qid])
        else:
            print(f"Warning: {qid} not found in dev queries")

    return pilot_queries

def build_v3_slm_pipeline() -> SLMPipeline_v3:
    """
    Instantiates genuine v3 pipeline with 4-specialist local pool (all <=3.2B)
    running GPU-accelerated on RTX 3050 via Ollama Vulkan.
    """
    decomposer_runner = OllamaModelRunner(
        logical_model_name="llama3.2-3b",
        api_model_name="llama3.2:3b"
    )
    coder_runner = OllamaModelRunner(
        logical_model_name="qwen2.5-coder-3b",
        api_model_name="qwen2.5-coder:3b"
    )
    math_runner = OllamaModelRunner(
        logical_model_name="deepseek-r1-1.5b",
        api_model_name="deepseek-r1:1.5b",
        max_tokens=2048
    )
    retrieval_runner = OllamaModelRunner(
        logical_model_name="qwen2.5-1.5b",
        api_model_name="qwen2.5:1.5b"
    )
    synthesis_runner = OllamaModelRunner(
        logical_model_name="llama3.2-3b",
        api_model_name="llama3.2:3b"
    )
    pool_runners = {
        "coding": coder_runner,
        "mathematics": math_runner,
        "formal_reasoning": math_runner,
        "retrieval_qa": retrieval_runner,
        "science_tech": coder_runner,
        "structured_data": coder_runner,
        "creative_synthesis": synthesis_runner,
        "systems_ops": coder_runner
    }
    aggregator_runner = OllamaModelRunner(
        logical_model_name="llama3.2-3b",
        api_model_name="llama3.2:3b"
    )

    logger = ExperimentLogger(log_dir=os.path.join(PILOT_DIR, "pipeline_logs"))

    return SLMPipeline_v3(
        decomposer_runner=decomposer_runner,
        pool_runners=pool_runners,
        aggregator_runner=aggregator_runner,
        logger=logger,
        max_depth=3,
        max_concurrent_slms=2
    )

async def run_v3_pilot_generation(api_key: Optional[str] = None):
    ensure_pilot_dirs()
    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key and os.path.exists(".env"):
            with open(".env", "r") as f:
                for line in f:
                    if line.startswith("GROQ_API_KEY="):
                        api_key = line.strip().split("=", 1)[1]
                        break

    queries = get_v3_pilot_queries()
    n_queries = len(queries)
    total_calls = n_queries * (1 + len(BASELINE_CONFIGS_V3)) # 16 * 2 = 32 calls

    completed_slm, completed_baselines, completed_comparisons = load_resumable_state()

    print(f"=== v3 Pilot Generation (16 Stratified Queries x 2 Systems = {total_calls} Calls) ===")
    print(f"Proposed Architecture: SLMPipeline_v3 (4-Specialist Local Pool <=3.2B on RTX 3050)")
    print(f"Comparative Baseline: openai/gpt-oss-120b (Monolithic, 120B on Groq LPU)")
    print(f"Target Directory: {PILOT_DIR}/")
    print(f"Resumable State: SLM completed={len(completed_slm)}, Baselines completed={len(completed_baselines)}, Comparison refs={len(completed_comparisons)}")
    print(f"Immediate fsync persistence enabled on all writes.\n")

    slm_pipeline = build_v3_slm_pipeline()

    baseline_runners = {
        b_id: APIGroqModelRunner(
            logical_model_name=b_name,
            api_model_name=b_name,
            api_key=api_key,
            max_tokens=2048
        )
        for b_id, b_name in BASELINE_CONFIGS_V3
    }

    # Permanent Pre-Flight Model Roster Verification (Hard Rule 13)
    pipeline_components = {
        "decomposer": slm_pipeline.decomposer_runner,
        "coding_specialist": slm_pipeline.pool_runners["coding"],
        "math_specialist": slm_pipeline.pool_runners["mathematics"],
        "retrieval_specialist": slm_pipeline.pool_runners["retrieval_qa"],
        "aggregator": slm_pipeline.aggregator_runner
    }
    verify_distinct_roster_preflight(pipeline_components, baseline_runners)

    start_all = time.perf_counter()
    calls_made = 0

    for q_idx, q in enumerate(queries):
        qid = q["id"]
        qtext = q["query"]
        tier = q.get("complexity_tier", "single_domain")

        # 1. SLM Pipeline v3 Generation
        if qid not in completed_slm:
            print(f"[{q_idx+1}/{n_queries}] Generating SLMPipeline_v3 for {qid} ({tier})...", flush=True)
            t0 = time.perf_counter()
            pipe_res = await slm_pipeline.execute_query(
                query_id=qid,
                query_text=qtext,
                complexity_tier=tier,
                seed=42,
                config={"version": "3.0.0"}
            )
            dur_s = time.perf_counter() - t0
            resp_text = pipe_res.get("response", "") or pipe_res.get("final_response", "")

            slm_record = {
                "query_id": qid,
                "system_type": "all_slm_pipeline_v3",
                "model_identifier": "src/v3/pipeline.py (4-Specialist Local Pool <=3.2B on RTX 3050)",
                "status": "SUCCESS" if resp_text and not resp_text.startswith("[Error") else "FAILED",
                "response_text": resp_text,
                "complexity_tier": tier,
                "domains": q.get("domains", []),
                "wall_clock_latency_ms": pipe_res.get("wall_clock_latency_ms", dur_s * 1000.0),
                "matched_tasks_count": pipe_res.get("matched_tasks_count", 1),
                "feedback_loop_fired": pipe_res.get("feedback_loop_fired", False),
                "loop_events_count": pipe_res.get("loop_events_count", 0),
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
            append_jsonl_immediate(SLM_FILE, slm_record)
            completed_slm.add(qid)
            calls_made += 1
            print(f"  -> SLM {qid} completed in {dur_s:.2f}s ({len(resp_text)} chars, loop_events={slm_record['loop_events_count']})", flush=True)
            await asyncio.sleep(1.0) # Graceful pacing

        # 2. Baseline Generations (Floor >= 30B)
        for b_id, b_name in BASELINE_CONFIGS_V3:
            if (qid, b_id) not in completed_baselines:
                print(f"[{q_idx+1}/{n_queries}] Generating baseline {b_id} for {qid}...", flush=True)
                b_runner = baseline_runners[b_id]
                t0 = time.perf_counter()
                b_resp = await b_runner.generate(
                    prompt=qtext,
                    system_prompt="You are an expert technical assistant. Answer the user query thoroughly, with complete precision, detailed reasoning, and full technical implementation."
                )
                dur_s = time.perf_counter() - t0
                b_text = b_resp.text

                base_record = {
                    "query_id": qid,
                    "baseline_model_id": b_id,
                    "model_identifier": b_name,
                    "status": "SUCCESS" if b_text and not b_text.startswith("[Error") else "FAILED",
                    "response_text": b_text,
                    "prompt_tokens": b_resp.prompt_tokens,
                    "completion_tokens": b_resp.completion_tokens,
                    "latency_ms": b_resp.latency_ms,
                    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                }
                append_jsonl_immediate(BASELINE_FILE, base_record)
                completed_baselines.add((qid, b_id))
                calls_made += 1
                print(f"  -> Baseline {b_id} for {qid} completed in {dur_s:.2f}s ({len(b_text)} chars)", flush=True)
                await asyncio.sleep(1.0)

        # 3. Reference Comparison Record
        if qid not in completed_comparisons:
            comp_record = {
                "query_id": qid,
                "complexity_tier": tier,
                "domains": q.get("domains", []),
                "query_text": qtext,
                "systems_compared": ["all_slm_pipeline_v3"] + [b[0] for b in BASELINE_CONFIGS_V3],
                "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
            append_jsonl_immediate(COMPARISON_FILE, comp_record)
            completed_comparisons.add(qid)

    elapsed_all = time.perf_counter() - start_all
    print(f"\n=== v3 Pilot Generation Complete ===")
    print(f"Dispatched {calls_made} new calls in {elapsed_all:.2f}s.")
    print(f"SLM Responses: {len(completed_slm)}/16 in {SLM_FILE}")
    print(f"Baseline Responses: {len(completed_baselines)}/64 in {BASELINE_FILE}")
    print(f"Comparison Records: {len(completed_comparisons)}/16 in {COMPARISON_FILE}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run v3 Pilot Generation")
    parser.add_argument("--api-key", type=str, default=None, help="Groq API Key")
    args = parser.parse_args()
    asyncio.run(run_v3_pilot_generation(api_key=args.api_key))

