"""
Pre-fetch baseline responses for the 30-query stratified benchmark:
  1. openai/gpt-oss-120b (Groq API)
  2. Qwen/Qwen2.5-72B-Instruct (HF Router API)
"""

import os
import sys
import json
import time
import asyncio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.groq_runner import APIGroqModelRunner
from src.models.hf_runner import HFRouterModelRunner

BENCHMARK_PATH = "data/eval_stratified_30_benchmark.json"
B120_PATH = "results/100_query_eval/baseline_120b_responses.jsonl"
B72_PATH = "results/100_query_eval/baseline_72b_responses.jsonl"

def load_cached(filepath):
    res = set()
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        res.add(json.loads(line)["query_id"])
                    except Exception:
                        pass
    return res

async def prefetch_120b(queries):
    runner = APIGroqModelRunner(
        logical_model_name="gpt-120b",
        api_model_name="openai/gpt-oss-120b",
        max_tokens=2048,
        temperature=0.0
    )
    cached = load_cached(B120_PATH)
    pending = [q for q in queries if q["id"] not in cached]
    print(f"\n>>> Pre-fetching {len(pending)} queries for 120B Baseline (Groq)...")
    for idx, q in enumerate(pending, 1):
        t0 = time.perf_counter()
        resp = await runner.generate(
            prompt=q["query"],
            system_prompt="You are an expert technical AI assistant. Provide authoritative, mathematically sound, syntactically verified solutions."
        )
        t = time.perf_counter() - t0
        with open(B120_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "query_id": q["id"],
                "complexity_tier": q["complexity_tier"],
                "query_text": q["query"],
                "system_type": "openai/gpt-oss-120b",
                "response_text": resp.text,
                "latency_sec": t
            }) + "\n")
        print(f"  [{idx}/{len(pending)}] 120B cached {q['id']} ({q['complexity_tier']}) in {t:.2f}s ({len(resp.text)} chars)")

async def prefetch_72b(queries):
    runner = HFRouterModelRunner(
        logical_model_name="qwen-72b",
        api_model_name="Qwen/Qwen2.5-72B-Instruct",
        max_tokens=1024,
        temperature=0.0
    )
    cached = load_cached(B72_PATH)
    pending = [q for q in queries if q["id"] not in cached]
    print(f"\n>>> Pre-fetching {len(pending)} queries for 72B Baseline (HF Router)...")
    for idx, q in enumerate(pending, 1):
        t0 = time.perf_counter()
        resp = await runner.generate(
            prompt=q["query"],
            system_prompt="You are an expert technical AI assistant. Provide authoritative, mathematically sound, syntactically verified solutions."
        )
        t = time.perf_counter() - t0
        with open(B72_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "query_id": q["id"],
                "complexity_tier": q["complexity_tier"],
                "query_text": q["query"],
                "system_type": "Qwen/Qwen2.5-72B-Instruct",
                "response_text": resp.text,
                "latency_sec": t
            }) + "\n")
        print(f"  [{idx}/{len(pending)}] 72B cached {q['id']} ({q['complexity_tier']}) in {t:.2f}s ({len(resp.text)} chars)")

async def main():
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)
    print(f"Loaded {len(queries)} queries from {BENCHMARK_PATH}.")
    
    # 1. First fetch 120B (fast Groq)
    await prefetch_120b(queries)
    
    # 2. Then fetch 72B (HF Router)
    await prefetch_72b(queries)
    
    print("\nPre-fetching completed for all baselines.")

if __name__ == "__main__":
    asyncio.run(main())

