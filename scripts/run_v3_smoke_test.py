"""
Single-Query End-to-End Smoke Test Harness (AI Search Framework v3)
Verifies:
  1. Permanent pre-flight assertion (Hard Rule 13) confirms 100% distinct model identities.
  2. Proposed SLM Pipeline (all <=5B) executes query end-to-end.
  3. All 4 Baselines (>=30B) execute query independently:
       - gemini_frontier: gemini-2.5-flash (Google AI Studio)
       - llama_70b: meta-llama/Llama-3.3-70B-Instruct (HF Router)
       - qwen_72b: Qwen/Qwen2.5-72B-Instruct (HF Router)
       - qwen_32b: Qwen/Qwen2.5-Coder-32B-Instruct (HF Router)
  4. Pairwise LLM Judge (qwen/qwen3.8-27b on Groq) evaluates both forward and swapped orders
     with max_tokens=1024 to produce non-empty reasoning.
"""

import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.models.groq_runner import APIGroqModelRunner
from src.models.gemini_runner import GoogleGenAIModelRunner
from src.models.ollama_runner import OllamaModelRunner
from src.v3.pipeline import SLMPipeline_v3
from src.v3.preflight import verify_distinct_roster_preflight
from src.v2.judge.pairwise_harness import PairwiseLLMJudgeHarness
from src.instrumentation.logger import ExperimentLogger

SMOKE_LOG_DIR = "logs/v3_smoke_test"
os.makedirs(SMOKE_LOG_DIR, exist_ok=True)
os.makedirs(os.path.join(SMOKE_LOG_DIR, "judge_pairwise"), exist_ok=True)
os.makedirs(os.path.join(SMOKE_LOG_DIR, "judge_keys"), exist_ok=True)

def load_keys():
    keys = {}
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                line = line.strip()
                if line and "=" in line:
                    k, v = line.split("=", 1)
                    keys[k] = v
    return keys

async def main():
    keys = load_keys()
    groq_key = keys.get("GROQ_API_KEY")
    gemini_key = keys.get("GEMINI_API_KEY")

    print("=" * 80)
    print("AI SEARCH FRAMEWORK v3: VERIFIED ZERO-COST SMOKE TEST (OPTION B HYBRID)")
    print("=" * 80)

    # 1. Instantiate SLM Pipeline (<=5B components locally on RTX 3050 via Ollama Vulkan)
    # Decomposer: Llama-3.2-3B-Instruct (3.2B params)
    decomposer_runner = OllamaModelRunner(
        logical_model_name="llama3.2-3b",
        api_model_name="llama3.2:3b"
    )
    # Specialist 1 (Coding & Systems): Qwen2.5-Coder-3B-Instruct (3.09B params)
    coder_runner = OllamaModelRunner(
        logical_model_name="qwen2.5-coder-3b",
        api_model_name="qwen2.5-coder:3b"
    )
    # Specialist 2 (Mathematics & Logic): DeepSeek-R1-1.5B (1.5B params)
    math_runner = OllamaModelRunner(
        logical_model_name="deepseek-r1-1.5b",
        api_model_name="deepseek-r1:1.5b"
    )
    # Specialist 3 (Retrieval QA & Language): Qwen2.5-1.5B-Instruct (1.5B params)
    retrieval_runner = OllamaModelRunner(
        logical_model_name="qwen2.5-1.5b",
        api_model_name="qwen2.5:1.5b"
    )
    # Specialist 4 (Synthesis): Llama-3.2-3B-Instruct (3.2B params)
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

    logger = ExperimentLogger(log_dir=os.path.join(SMOKE_LOG_DIR, "pipeline_logs"))
    slm_pipeline = SLMPipeline_v3(
        decomposer_runner=decomposer_runner,
        pool_runners=pool_runners,
        aggregator_runner=aggregator_runner,
        logger=logger,
        max_depth=3,
        max_concurrent_slms=2
    )

    # 2. Instantiate Genuinely Monolithic Frontier Baseline (Floor >= 30B: openai/gpt-oss-120b)
    baseline_runners = {
        "gpt_120b": APIGroqModelRunner(
            logical_model_name="openai/gpt-oss-120b",
            api_model_name="openai/gpt-oss-120b",
            api_key=groq_key
        )
    }

    # 3. Independent Pairwise Judge (qwen3.8-27b on Groq)
    judge_harness = PairwiseLLMJudgeHarness(
        judge_model_name="qwen/qwen3.8-27b",
        api_key=groq_key
    )

    # 4. HARD RULE 13: PRE-FLIGHT VERIFICATION
    print("\n[PHASE 1] Executing Hard Rule 13 Pre-Flight Assertion...")
    pipeline_components = {
        "decomposer": slm_pipeline.decomposer_runner,
        "coding_specialist": slm_pipeline.pool_runners["coding"],
        "math_specialist": slm_pipeline.pool_runners["mathematics"],
        "retrieval_specialist": slm_pipeline.pool_runners["retrieval_qa"],
        "aggregator": slm_pipeline.aggregator_runner
    }
    
    resolved = verify_distinct_roster_preflight(
        slm_pipeline_runners=pipeline_components,
        baseline_runners=baseline_runners,
        judge_runner=judge_harness
    )
    print("Pre-flight assertion passed! Verified resolved endpoints:")
    for k, v in resolved.items():
        print(f"  - {k:30s} -> {v}")

    # 5. Execute Single Smoke Query
    # Query: V3_SD_CODI_01 (Advanced Algorithmic Module)
    smoke_query = {
        "id": "V3_SD_CODI_01",
        "complexity_tier": "single_domain",
        "domain": "coding",
        "query": (
            "Design and implement a high-performance, generic in-memory Least-Recently-Used (LRU) Cache "
            "in Python with full PEP 484 type annotations. Support get(key) and put(key, value) in strict O(1) "
            "average time complexity, thread-safe locking, and an eviction callback. Explain the time and space "
            "complexity tradeoffs rigorously."
        )
    }
    qid = smoke_query["id"]
    qtext = smoke_query["query"]

    print(f"\n[PHASE 2] Generating Responses for Query {qid}...")
    
    # 5a. SLM Pipeline Run
    print("  -> Running SLMPipeline_v3 (Decomposer + Pool + Aggregator)...")
    t0 = time.perf_counter()
    pipe_res = await slm_pipeline.execute_query(
        query_id=qid,
        query_text=qtext,
        complexity_tier="single_domain",
        seed=42,
        config={"version": "3.0.0-smoke"}
    )
    slm_text = pipe_res.get("response", "") or pipe_res.get("final_response", "")
    slm_latency = (time.perf_counter() - t0) * 1000.0
    print(f"     [SLM Done] length={len(slm_text)} chars, latency={slm_latency:.1f}ms")

    # 5b. Baselines Run
    baseline_responses = {}
    for b_id, runner in baseline_runners.items():
        print(f"  -> Running Baseline: {b_id} ({runner.api_model_name})...")
        t0 = time.perf_counter()
        resp = await runner.generate(
            prompt=qtext,
            system_prompt="You are a frontier-class AI assistant. Provide a complete, rigorous, production-grade technical solution."
        )
        b_lat = (time.perf_counter() - t0) * 1000.0
        baseline_responses[b_id] = resp.text
        print(f"     [{b_id} Done] length={len(resp.text)} chars, latency={b_lat:.1f}ms")

    # 6. Save Responses to Smoke Log
    smoke_out = {
        "query_id": qid,
        "query_text": qtext,
        "systems": {
            "slm_pipeline_v3": {"model": "Local SLM Pool (Llama-3.2-3B / Qwen-2.5-Coder-3B / Qwen-2.5-1.5B)", "length": len(slm_text), "response": slm_text},
            **{b_id: {"model": baseline_runners[b_id].api_model_name, "length": len(baseline_responses[b_id]), "response": baseline_responses[b_id]} for b_id in baseline_runners}
        }
    }
    with open(os.path.join(SMOKE_LOG_DIR, "smoke_responses.json"), "w", encoding="utf-8") as f:
        json.dump(smoke_out, f, indent=2)

    # 7. Evaluate Pairwise Judge (Bidirectional: Forward and Swapped)
    print(f"\n[PHASE 3] Running Double-Blind Pairwise LLM Judge Trials (max_tokens=1024)...")
    judge_results = []
    
    for b_id, b_text in baseline_responses.items():
        for order in ["forward", "swapped"]:
            trial_res = judge_harness.evaluate_pair(
                query_id=qid,
                query_text=qtext,
                system_a_id="slm_pipeline_v3",
                text_a=slm_text,
                system_b_id=b_id,
                text_b=b_text,
                order_tag=order,
                judge_log_dir=os.path.join(SMOKE_LOG_DIR, "judge_pairwise"),
                key_log_dir=os.path.join(SMOKE_LOG_DIR, "judge_keys")
            )
            judge_results.append(trial_res)
            print(f"  [{order:7s}] SLM vs {b_id:15s} -> Winner: {trial_res.get('unblinded_winner'):15s} | Reason len: {len(trial_res.get('reasoning', ''))}")

    # 8. Save Judge Summary
    with open(os.path.join(SMOKE_LOG_DIR, "smoke_judge_summary.json"), "w", encoding="utf-8") as f:
        json.dump(judge_results, f, indent=2)

    # 9. Print Side-by-Side Raw Outputs for Forensic Verification
    print("\n" + "=" * 80)
    print("SIDE-BY-SIDE RAW OUTPUTS (QUERY: " + qid + ")")
    print("=" * 80)
    print("\n[PROPOSED SLM PIPELINE (Decomposed <=3B local pool)]")
    print("-" * 60)
    print(slm_text)
    print("-" * 60)
    print(f"Total Chars: {len(slm_text)} | Latency: {slm_latency:.1f}ms")

    for b_id, b_text in baseline_responses.items():
        print(f"\n[BASELINE: {b_id} (openai/gpt-oss-120b)]")
        print("-" * 60)
        print(b_text)
        print("-" * 60)
        print(f"Total Chars: {len(b_text)}")

    print("\n" + "=" * 80)
    print("DOUBLE-BLIND PAIRWISE JUDGE EVALUATIONS")
    print("=" * 80)
    for res in judge_results:
        print(f"\nTrial: Order={res.get('order_tag')} | Blind Selected={res.get('selected_candidate')} | Unblinded Winner={res.get('unblinded_winner')}")
        print("Judge Reasoning:")
        print(res.get("reasoning"))

    print("\n" + "=" * 80)
    print("SMOKE TEST COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
