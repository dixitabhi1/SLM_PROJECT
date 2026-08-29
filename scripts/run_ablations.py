"""
Ablation Study Runner
AI Search Framework (Phase 9: Ablations)

Runs systematic ablations across:
1. Replication Mechanism (No replication vs Confidence-based vs Full replication)
2. Pool Specialization (Specialized SLM Pool vs Single General SLM)
3. Decomposer Sensitivity across complexity tiers
"""

import asyncio
import os
import sys
import json
import csv
from typing import Dict, List, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.mock_runner import MockModelRunner
from src.router.capability_router import CapabilityRouter
from src.instrumentation.logger import ExperimentLogger
from src.pipeline import SLMPipeline
from src.analysis.metrics import StatisticalAnalyzer

os.makedirs("results/ablations", exist_ok=True)

async def run_ablation_suite():
    with open("config/experiment_config.json", "r") as f:
        config = json.load(f)
    with open("data/queries_dev.json", "r") as f:
        queries = json.load(f)[:20] # Representative dev slice

    logger = ExperimentLogger(pricing_table_path="config/pricing_table.json", log_dir="logs/ablation_runs")
    analyzer = StatisticalAnalyzer()

    # --- Ablation 1: Replication Strategy ---
    print("\n=== ABLATION 1: REPLICATION STRATEGY ===")
    replication_configs = [
        {"name": "no_replication", "cutoff": 0.0},
        {"name": "confidence_gated_0.65", "cutoff": 0.65},
        {"name": "aggressive_replication_0.95", "cutoff": 0.95}
    ]

    ablation_1_results = []
    for rep in replication_configs:
        router = CapabilityRouter(replication_cutoff=rep["cutoff"])
        decomposer = MockModelRunner("meta-llama/Llama-3.2-3B-Instruct")
        pool = {
            "coding": MockModelRunner("Qwen/Qwen2.5-Coder-7B-Instruct"),
            "math": MockModelRunner("Qwen/Qwen2.5-Math-7B-Instruct"),
            "reasoning": MockModelRunner("microsoft/Phi-3.5-mini-instruct"),
            "retrieval": MockModelRunner("meta-llama/Llama-3.2-3B-Instruct"),
            "general": MockModelRunner("meta-llama/Llama-3.1-8B-Instruct")
        }
        aggregator = MockModelRunner("meta-llama/Llama-3.1-8B-Instruct")
        
        pipe = SLMPipeline(decomposer, pool, aggregator, router, logger=logger)
        
        latencies, costs, tokens = [], [], []
        replications_count = 0
        for q in queries:
            res = await pipe.execute_query(q["id"], q["query"], q["complexity_tier"], 42, config)
            latencies.append(res["wall_clock_latency_ms"])
            costs.append(res["cost_usd"])
            tokens.append(res["total_tokens"])
            for trace in res["record"].get("dispatch_trace", []):
                if trace.get("replicated"):
                    replications_count += 1

        entry = {
            "condition": rep["name"],
            "mean_wall_latency_ms": round(sum(latencies)/len(latencies), 2),
            "mean_cost_usd": round(sum(costs)/len(costs), 6),
            "mean_tokens": round(sum(tokens)/len(tokens), 1),
            "total_replications": replications_count
        }
        ablation_1_results.append(entry)
        print(f"  [{rep['name']}] Mean Latency: {entry['mean_wall_latency_ms']}ms | Mean Cost: ${entry['mean_cost_usd']} | Replications: {replications_count}")

    # --- Ablation 2: Specialized Pool vs General-Only SLM ---
    print("\n=== ABLATION 2: SPECIALIZED POOL VS GENERAL-ONLY SLM ===")
    general_pool = {
        k: MockModelRunner("meta-llama/Llama-3.1-8B-Instruct") for k in ["coding", "math", "reasoning", "retrieval", "general"]
    }
    specialized_pool = {
        "coding": MockModelRunner("Qwen/Qwen2.5-Coder-7B-Instruct"),
        "math": MockModelRunner("Qwen/Qwen2.5-Math-7B-Instruct"),
        "reasoning": MockModelRunner("microsoft/Phi-3.5-mini-instruct"),
        "retrieval": MockModelRunner("meta-llama/Llama-3.2-3B-Instruct"),
        "general": MockModelRunner("meta-llama/Llama-3.1-8B-Instruct")
    }

    ablation_2_results = []
    for p_name, p_dict in [("Specialized_Pool", specialized_pool), ("Homogeneous_General_SLM_Pool", general_pool)]:
        router = CapabilityRouter(replication_cutoff=0.65)
        pipe = SLMPipeline(MockModelRunner("meta-llama/Llama-3.2-3B-Instruct"), p_dict, MockModelRunner("meta-llama/Llama-3.1-8B-Instruct"), router, logger=logger)
        
        latencies, costs = [], []
        for q in queries:
            res = await pipe.execute_query(q["id"], q["query"], q["complexity_tier"], 42, config)
            latencies.append(res["wall_clock_latency_ms"])
            costs.append(res["cost_usd"])

        entry = {
            "pool_architecture": p_name,
            "mean_latency_ms": round(sum(latencies)/len(latencies), 2),
            "mean_cost_usd": round(sum(costs)/len(costs), 6)
        }
        ablation_2_results.append(entry)
        print(f"  [{p_name}] Mean Latency: {entry['mean_latency_ms']}ms | Mean Cost: ${entry['mean_cost_usd']}")

    # Save ablation results
    with open("results/ablations/ablation_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "ablation_1_replication": ablation_1_results,
            "ablation_2_specialization": ablation_2_results
        }, f, indent=2)

    print("\nSaved ablation results to results/ablations/ablation_results.json")

if __name__ == "__main__":
    asyncio.run(run_ablation_suite())

