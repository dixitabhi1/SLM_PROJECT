"""
v2 Unified Evaluation Harness
AI Search Framework (Phase v2.7: Full v2 Runs)

Executes:
1. All-SLM Pipeline v2 with feedback loop & two-stage aggregation
2. Multi-LLM Baseline Roster (5 models: Llama-3.1-8B, Qwen-2.5-32B, Llama-3.1-70B, Qwen-2.5-72B, Gemini-1.5-Pro)
3. Anonymous LLM Judge (Claude-3.5-Sonnet) with per-query shuffling
4. Generates per-query structured JSON records and aggregated dataset logs.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Dict, List, Any

sys.path.insert(0, os.path.abspath("."))

from src.models.mock_runner import MockModelRunner
from src.instrumentation.logger import ExperimentLogger
from src.v2.pipeline import SLMPipeline_v2
from src.v2.judge.judge_evaluator import AnonymousLLMJudge

async def run_v2_benchmark(split: str = "dev", seed: int = 42, max_queries: int = None):
    dataset_path = f"data/v2_queries_{split}.json"
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found.")
        sys.exit(1)

    with open(dataset_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    if max_queries:
        queries = queries[:max_queries]

    print(f"=== Running v2 Benchmark ({split.upper()} split, N={len(queries)}, Seed={seed}) ===")

    # 1. Setup Logger & Configurations
    os.makedirs("results/v2_records", exist_ok=True)
    os.makedirs("logs/runs", exist_ok=True)
    os.makedirs("logs/judge", exist_ok=True)
    
    logger = ExperimentLogger(pricing_table_path="config/pricing_table.json", log_dir="logs/runs")
    with open("config/experiment_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    # 2. Initialize Model Runners
    # SLM Pipeline Runners (all <= 8B)
    decomposer_runner = MockModelRunner("meta-llama/Llama-3.2-3B-Instruct")
    pool_runners = {
        "coding": MockModelRunner("Qwen/Qwen2.5-Coder-7B-Instruct"),
        "math": MockModelRunner("Qwen/Qwen2.5-Math-7B-Instruct"),
        "reasoning": MockModelRunner("microsoft/Phi-3.5-mini-instruct"),
        "retrieval": MockModelRunner("meta-llama/Llama-3.2-3B-Instruct"),
        "general": MockModelRunner("meta-llama/Llama-3.1-8B-Instruct")
    }
    aggregator_runner = MockModelRunner("meta-llama/Llama-3.1-8B-Instruct")

    pipeline_v2 = SLMPipeline_v2(
        decomposer_runner=decomposer_runner,
        pool_runners=pool_runners,
        aggregator_runner=aggregator_runner,
        logger=logger,
        max_depth=3,
        max_concurrent_slms=4
    )

    # Multi-LLM Baseline Roster Runners
    baseline_runners = {
        "llama_8b": MockModelRunner("meta-llama/Llama-3.1-8B-Instruct"),
        "qwen_32b": MockModelRunner("Qwen/Qwen2.5-32B-Instruct"),
        "llama_70b": MockModelRunner("meta-llama/Llama-3.1-70B-Instruct"),
        "qwen_72b": MockModelRunner("Qwen/Qwen2.5-72B-Instruct"),
        "gemini_frontier": MockModelRunner("gemini-1.5-pro")
    }

    # Independent Judge Runner
    judge_runner = MockModelRunner("claude-3-5-sonnet-20241022")
    judge = AnonymousLLMJudge(judge_runner)

    # Gold DAGs lookup
    gold_dags_map = {}
    if os.path.exists("data/v2_gold_dags.json"):
        with open("data/v2_gold_dags.json", "r", encoding="utf-8") as f:
            gold_dags_map = json.load(f)

    # 3. Execution Loop
    all_query_records = []
    judge_summaries = []

    for idx, q in enumerate(queries):
        qid = q["id"]
        qtext = q["query"]
        tier = q["complexity_tier"]
        q_seed = seed + idx

        print(f"[{idx+1}/{len(queries)}] Processing {qid} ({tier})...", end="", flush=True)

        # A. Run SLM Pipeline v2
        pipe_out = await pipeline_v2.execute_query(
            query_id=qid,
            query_text=qtext,
            complexity_tier=tier,
            seed=q_seed,
            config=config
        )

        # B. Run all 5 Baselines
        baseline_responses = {}
        candidate_pool_for_judge = {"slm_pipeline_v2": pipe_out["response"]}

        for b_id, b_runner in baseline_runners.items():
            b_run_id = f"baseline_{b_id}_{qid}_{q_seed}"
            b_record = logger.create_run_record(
                run_id=b_run_id,
                system_type=f"baseline_{b_id}",
                query_id=qid,
                query_text=qtext,
                complexity_tier=tier,
                seed=q_seed,
                config={"baseline_model_id": b_id}
            )
            b_start = time.perf_counter()
            b_resp = await b_runner.generate(
                prompt=f"User Query: {qtext}\n\nProvide a comprehensive, authoritative response:",
                system_prompt="You are a frontier general-purpose AI assistant. Answer the user prompt directly."
            )
            b_end = time.perf_counter()
            logger.record_stage(
                record=b_record,
                stage_name=f"monolithic_inference_{b_id}",
                model_name=b_runner.model_name,
                model_revision=b_runner.revision,
                start_time_s=b_start,
                end_time_s=b_end,
                prompt_tokens=b_resp.prompt_tokens,
                completion_tokens=b_resp.completion_tokens,
                input_data=qtext,
                output_data=b_resp.text,
                extra_metadata={"baseline_model_id": b_id}
            )
            b_log_path = logger.finalize_run(b_record, b_start, b_end, b_resp.text)
            
            baseline_responses[b_id] = {
                "model_name": b_runner.model_name,
                "response": b_resp.text,
                "run_id": b_run_id,
                "log_path": b_log_path,
                "latency_ms": (b_end - b_start) * 1000.0,
                "cost_usd": b_record.get("total_cost_usd", 0.0)
            }
            candidate_pool_for_judge[b_id] = b_resp.text

        # C. Run Anonymous LLM Judge
        judge_out = await judge.judge_query_candidates(
            query_id=qid,
            query_text=qtext,
            candidate_pool=candidate_pool_for_judge,
            shuffle_seed=q_seed
        )

        judge_summaries.append({
            "query_id": qid,
            "complexity_tier": tier,
            "selected_system": judge_out["selected_system"],
            "primary_differentiator": judge_out["primary_differentiator"]
        })

        # D. Assemble Standard v2 Per-Query JSON Record
        record_json = {
            "query_id": qid,
            "query_text": qtext,
            "complexity_tier": tier,
            "triggers_feedback_loop": q.get("triggers_feedback_loop", False),
            "gold_dag": gold_dags_map.get(qid),
            "slm_pipeline_response": {
                "final_response": pipe_out["response"],
                "run_id": pipe_out["record"].get("run_id"),
                "log_path": pipe_out.get("log_path"),
                "wall_clock_latency_ms": pipe_out["wall_clock_latency_ms"],
                "simulated_parallel_latency_ms": pipe_out["simulated_parallel_latency_ms"],
                "cost_usd": pipe_out["cost_usd"],
                "feedback_loop_fired": pipe_out["feedback_loop_fired"],
                "loop_events_count": pipe_out["loop_events_count"]
            },
            "baseline_responses": baseline_responses,
            "comparison": {
                "cost_ratio_vs_70b": pipe_out["cost_usd"] / max(baseline_responses["llama_70b"]["cost_usd"], 1e-8),
                "speedup_ratio_parallel_vs_70b": baseline_responses["llama_70b"]["latency_ms"] / max(pipe_out["simulated_parallel_latency_ms"], 1e-8)
            },
            "judge_result": {
                "judge_model_id": judge_out["judge_model"],
                "selected_system": judge_out["selected_system"],
                "selected_alias": judge_out["selected_alias"],
                "criteria_scores": judge_out["criteria_scores"],
                "primary_differentiator": judge_out["primary_differentiator"],
                "reasoning": judge_out["reasoning"],
                "shuffle_seed": judge_out["shuffle_seed"]
            }
        }

        # Save individual per-query JSON
        q_record_path = f"results/v2_records/{qid}.json"
        with open(q_record_path, "w", encoding="utf-8") as f:
            json.dump(record_json, f, indent=2)

        all_query_records.append(record_json)
        print(f" [Done - Judge Picked: {judge_out['selected_system']}]")

    # 4. Save Master JSONL & Aggregated Benchmark Log
    jsonl_path = f"results/v2_eval_{split}_master.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in all_query_records:
            f.write(json.dumps(rec) + "\n")

    summary_path = f"results/v2_benchmark_{split}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "split": split,
            "total_queries": len(all_query_records),
            "seed": seed,
            "judge_model": judge_runner.model_name,
            "baseline_roster": list(baseline_runners.keys()),
            "judge_summaries": judge_summaries
        }, f, indent=2)

    print(f"\nSUCCESS: Completed v2 Benchmark ({len(all_query_records)} queries).")
    print(f"Master records written to: {jsonl_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "held_out"], default="dev")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-queries", type=int, default=None)
    args = parser.parse_args()

    asyncio.run(run_v2_benchmark(split=args.split, seed=args.seed, max_queries=args.max_queries))
