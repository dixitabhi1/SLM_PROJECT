"""
AI Search Framework - Version 4 Step 1: True <=5B Model Capacity Evaluation
Evaluates Microsoft Phi-3.5-mini-instruct (3.82B) as pool specialist across 8 target queries:
4 Compound DAG: V3_CD_01, V3_CD_21, V3_CD_41, V3_CD_61
4 Single/Two-Domain: V3_SD_MATH_01, V3_SD_FORM_01, V3_TD_01, V3_TD_11

Held Fixed:
- Decomposer: llama3.2:3b
- Aggregator: llama3.2:3b
- Routing: TaskColorerSLM_v3
- Baseline: openai/gpt-oss-120b (Groq LPU)
- Judge: qwen/qwen3.8-27b (Groq LPU, max_tokens=2048, double-blind)
"""

import os
import sys
import json
import time
import asyncio
from typing import Dict, Any, List, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.ollama_runner import OllamaModelRunner
from src.v3.pipeline import SLMPipeline_v3
from src.instrumentation.logger import ExperimentLogger

V4_STEP1_DIR = "results/v4_step1"
PIPELINE_LOG_DIR = os.path.join(V4_STEP1_DIR, "pipeline_logs")
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

def ensure_dirs():
    os.makedirs(V4_STEP1_DIR, exist_ok=True)
    os.makedirs(PIPELINE_LOG_DIR, exist_ok=True)
    os.makedirs(KEYS_DIR, exist_ok=True)
    os.makedirs(JUDGE_DIR, exist_ok=True)

def load_target_queries() -> List[Dict[str, Any]]:
    with open("data/v3_queries_dev.json", "r", encoding="utf-8") as f:
        all_queries = json.load(f)
    qmap = {q["id"]: q for q in all_queries}
    targets = []
    for qid in TARGET_QUERY_IDS:
        if qid in qmap:
            targets.append(qmap[qid])
        else:
            raise ValueError(f"Target query {qid} not found in data/v3_queries_dev.json")
    return targets

def build_v4_step1_pipeline() -> SLMPipeline_v3:
    """
    Builds v4 Step 1 pipeline with Phi-3.5-mini (3.82B) as the specialist pool,
    holding decomposer and aggregator identical to v3 (llama3.2:3b).
    """
    decomposer_runner = OllamaModelRunner(
        logical_model_name="llama3.2-3b",
        api_model_name="llama3.2:3b"
    )
    
    # 3.82B specialist model evaluating the true <=5B ceiling
    phi_specialist = OllamaModelRunner(
        logical_model_name="phi3.5-3.8b",
        api_model_name="phi3.5:cpu",
        max_tokens=2048
    )

    aggregator_runner = OllamaModelRunner(
        logical_model_name="llama3.2-3b",
        api_model_name="llama3.2:3b"
    )

    pool_runners = {
        "coding": phi_specialist,
        "mathematics": phi_specialist,
        "formal_reasoning": phi_specialist,
        "retrieval_qa": phi_specialist,
        "science_tech": phi_specialist,
        "structured_data": phi_specialist,
        "creative_synthesis": phi_specialist,
        "systems_ops": phi_specialist
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

def verify_preflight_distinct_models(pipeline: SLMPipeline_v3, baseline_model_name: str):
    """
    Hard Rule 13: Distinct-model pre-flight verification requirement.
    Ensures every system in the roster maps to a valid endpoint,
    and that no pipeline component matches any baseline model.
    """
    print("Executing Hard Rule 13 pre-flight verification...")
    pipeline_models = {
        "decomposer": pipeline.decomposer_runner.api_model_name,
        "aggregator": pipeline.aggregator_runner.api_model_name,
    }
    for role, runner in pipeline.pool_runners.items():
        pipeline_models[f"pool_{role}"] = runner.api_model_name

    print("Pipeline model roster:", pipeline_models)
    print("Baseline model name:", baseline_model_name)

    # Check distinctness from baseline
    for role, m in pipeline_models.items():
        assert m != baseline_model_name, f"VIOLATION: Pipeline component {role} ({m}) matches baseline ({baseline_model_name})!"

    # Verify endpoint health and model availability
    import urllib.request
    req = urllib.request.Request("http://localhost:11434/api/tags")
    with urllib.request.urlopen(req) as resp:
        tags_data = json.loads(resp.read().decode("utf-8"))
        avail = [m["name"] for m in tags_data.get("models", [])]
        print("Available Ollama models:", avail)
        assert any(pipeline.decomposer_runner.api_model_name in name for name in avail), f"Decomposer model {pipeline.decomposer_runner.api_model_name} missing from Ollama"
        assert any("phi3.5" in name for name in avail), f"Phi-3.5 model missing from Ollama"
    print("Pre-flight assertion passed successfully.\n")

async def run_step1_generation():
    ensure_dirs()
    queries = load_target_queries()
    print(f"Loaded {len(queries)} target queries for v4 Step 1.")

    pipeline = build_v4_step1_pipeline()
    verify_preflight_distinct_models(pipeline, baseline_model_name="openai/gpt-oss-120b")
    responses_file = os.path.join(V4_STEP1_DIR, "slm_phi35_responses.jsonl")

    # Load existing to support resumption
    completed_qids = set()
    if os.path.exists(responses_file):
        with open(responses_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    completed_qids.add(d["query_id"])

    print(f"Already completed: {len(completed_qids)} / {len(queries)}")

    with open(responses_file, "a", encoding="utf-8") as out_f:
        for idx, q in enumerate(queries, 1):
            qid = q["id"]
            if qid in completed_qids:
                print(f"[{idx}/{len(queries)}] Skipping {qid} (already done)")
                continue

            qtext = q["query"]
            ctier = q.get("complexity_tier", "unknown")
            print(f"\n[{idx}/{len(queries)}] Running SLM Pipeline (Phi-3.5-mini 3.8B) for {qid} ({ctier})...")
            t0 = time.perf_counter()
            pipe_res = await pipeline.execute_query(
                query_id=qid,
                query_text=qtext,
                complexity_tier=ctier,
                seed=42,
                config={"version": "4.1.0"}
            )
            lat = time.perf_counter() - t0
            resp_text = pipe_res.get("response", "")
            print(f"  -> Generated {len(resp_text)} chars in {lat:.2f}s")

            entry = {
                "query_id": qid,
                "complexity_tier": ctier,
                "query_text": qtext,
                "model": "phi3.5-3.8b",
                "response_text": resp_text,
                "latency_sec": lat,
                "feedback_loop_fired": pipe_res.get("feedback_loop_fired", False),
                "log_path": pipe_res.get("log_path", "")
            }
            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            out_f.flush()

    print(f"\nCompleted v4 Step 1 SLM generation. Responses saved to {responses_file}")

if __name__ == "__main__":
    asyncio.run(run_step1_generation())

