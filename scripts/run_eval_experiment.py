"""
AI Search Framework — Full Experiment Runner Harness
Supports both live endpoints and simulated/mock execution.

Usage:
  python scripts/run_eval_experiment.py --system baseline --split dev --mode mock
  python scripts/run_eval_experiment.py --system pipeline --split dev --mode mock
  python scripts/run_eval_experiment.py --system baseline --split held_out --mode mock
  python scripts/run_eval_experiment.py --system pipeline --split held_out --mode mock
"""

import asyncio
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, List, Any
from src.models.mock_runner import MockModelRunner
from src.models.vllm_runner import VLLMModelRunner
from src.router.capability_router import CapabilityRouter
from src.instrumentation.logger import ExperimentLogger
from src.pipeline import SLMPipeline
from src.baseline.runner import BaselineRunner

def load_queries(split: str) -> List[Dict[str, Any]]:
    if split == "dev":
        path = "data/queries_dev.json"
    elif split == "held_out":
        path = "data/queries_held_out.json"
    elif split == "master":
        path = "data/eval_dataset_master.json"
    else:
        raise ValueError(f"Unknown split: {split}")
    
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_config() -> Dict[str, Any]:
    with open("config/experiment_config.json", "r", encoding="utf-8") as f:
        return json.load(f)

def build_models(config: Dict[str, Any], mode: str):
    m_cfg = config["models"]
    seed = config.get("seed", 42)

    if mode == "mock":
        baseline_runner = MockModelRunner(
            model_name=m_cfg["baseline"]["name"],
            revision=m_cfg["baseline"]["revision"],
            base_latency_ms=650.0, # 70B monolithic baseline latency emulation
            tokens_per_sec=25.0,
            fixed_seed=seed
        )
        decomposer_runner = MockModelRunner(
            model_name=m_cfg["decomposer"]["name"],
            revision=m_cfg["decomposer"]["revision"],
            base_latency_ms=90.0, # 3B SLM fast latency
            tokens_per_sec=65.0,
            fixed_seed=seed
        )
        pool_runners = {
            "coding": MockModelRunner(m_cfg["pool"]["coding"]["name"], m_cfg["pool"]["coding"]["revision"], base_latency_ms=130.0, tokens_per_sec=50.0, fixed_seed=seed),
            "math": MockModelRunner(m_cfg["pool"]["math"]["name"], m_cfg["pool"]["math"]["revision"], base_latency_ms=125.0, tokens_per_sec=50.0, fixed_seed=seed),
            "reasoning": MockModelRunner(m_cfg["pool"]["reasoning"]["name"], m_cfg["pool"]["reasoning"]["revision"], base_latency_ms=110.0, tokens_per_sec=55.0, fixed_seed=seed),
            "retrieval": MockModelRunner(m_cfg["pool"]["retrieval"]["name"], m_cfg["pool"]["retrieval"]["revision"], base_latency_ms=95.0, tokens_per_sec=60.0, fixed_seed=seed),
            "general": MockModelRunner(m_cfg["pool"]["general"]["name"], m_cfg["pool"]["general"]["revision"], base_latency_ms=140.0, tokens_per_sec=48.0, fixed_seed=seed)
        }
        aggregator_runner = MockModelRunner(
            model_name=m_cfg["aggregator"]["name"],
            revision=m_cfg["aggregator"]["revision"],
            base_latency_ms=150.0, # 8B SLM aggregator
            tokens_per_sec=45.0,
            fixed_seed=seed
        )
    else: # live vLLM / Ollama
        baseline_runner = VLLMModelRunner(
            model_name=m_cfg["baseline"]["name"],
            revision=m_cfg["baseline"]["revision"],
            endpoint=m_cfg["baseline"]["endpoint"]
        )
        decomposer_runner = VLLMModelRunner(
            model_name=m_cfg["decomposer"]["name"],
            revision=m_cfg["decomposer"]["revision"],
            endpoint=m_cfg["decomposer"]["endpoint"]
        )
        pool_runners = {
            k: VLLMModelRunner(v["name"], v["revision"], endpoint=v["endpoint"])
            for k, v in m_cfg["pool"].items()
        }
        aggregator_runner = VLLMModelRunner(
            model_name=m_cfg["aggregator"]["name"],
            revision=m_cfg["aggregator"]["revision"],
            endpoint=m_cfg["aggregator"]["endpoint"]
        )

    return baseline_runner, decomposer_runner, pool_runners, aggregator_runner

async def run_experiment(system_type: str, split: str, mode: str, limit: int = None):
    config = load_config()
    queries = load_queries(split)
    if limit:
        queries = queries[:limit]

    logger = ExperimentLogger(pricing_table_path="config/pricing_table.json", log_dir="logs/runs")
    baseline_runner, decomposer_runner, pool_runners, aggregator_runner = build_models(config, mode)
    router = CapabilityRouter(
        confidence_threshold=config["router"]["confidence_threshold"],
        replication_cutoff=config["router"]["replication_confidence_cutoff"]
    )

    print(f"=== Starting Experiment Run ===")
    print(f"System: {system_type.upper()}")
    print(f"Split: {split} ({len(queries)} queries)")
    print(f"Mode: {mode.upper()}")
    print(f"Seed: {config['seed']}")
    print("-" * 50)

    results = []
    
    if system_type == "baseline":
        runner = BaselineRunner(baseline_runner, logger=logger)
        for idx, q in enumerate(queries):
            res = await runner.execute_query(
                query_id=q["id"],
                query_text=q["query"],
                complexity_tier=q["complexity_tier"],
                seed=config["seed"],
                config=config
            )
            results.append(res)
            print(f"[{idx+1}/{len(queries)}] {q['id']} ({q['complexity_tier']}) | Latency: {res['wall_clock_latency_ms']:.1f}ms | Cost: ${res['cost_usd']:.6f}")
    
    elif system_type == "pipeline":
        pipeline = SLMPipeline(
            decomposer_runner=decomposer_runner,
            pool_runners=pool_runners,
            aggregator_runner=aggregator_runner,
            router=router,
            logger=logger,
            parallelism_mode=config["hardware"]["parallelism"]
        )
        for idx, q in enumerate(queries):
            res = await pipeline.execute_query(
                query_id=q["id"],
                query_text=q["query"],
                complexity_tier=q["complexity_tier"],
                seed=config["seed"],
                config=config
            )
            results.append(res)
            print(f"[{idx+1}/{len(queries)}] {q['id']} ({q['complexity_tier']}) | Wall Latency: {res['wall_clock_latency_ms']:.1f}ms | Sim Parallel: {res['simulated_parallel_latency_ms']:.1f}ms | Cost: ${res['cost_usd']:.6f}")

    print("-" * 50)
    print(f"Experiment completed successfully. All runs logged to logs/runs/")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AI Search Framework Experiment")
    parser.add_argument("--system", choices=["baseline", "pipeline"], required=True, help="System to evaluate")
    parser.add_argument("--split", choices=["dev", "held_out", "master"], default="dev", help="Dataset split")
    parser.add_argument("--mode", choices=["mock", "live"], default="mock", help="Execution mode")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries")
    args = parser.parse_args()

    asyncio.run(run_experiment(args.system, args.split, args.mode, args.limit))
