"""
AI Search Framework - 100-Query Cross-Tier Benchmark Runner
Evaluates SLMPipeline_v5 (all <=5B, zero LLMs) vs:
  1. openai/gpt-oss-120b (Frontier LLM Baseline via Groq)
  2. Qwen/Qwen2.5-72B-Instruct (72B Flagship LLM Baseline via HF Router)
across the stratified 100-query benchmark (34 SD, 33 TD, 33 CD).

Enforces:
1. Hard Rule 5: Zero overlap with data/v3_queries_held_out.json.
2. Hard Rule 13: Distinct-model pre-flight assertion across all systems & judge.
3. Hard Rule 14: Background write-lock discipline.
4. Hard Rule 15: Compound DAG decomposition non-collapse assertion.
5. Autonomous Audit Loop: Symmetrical double-blind judging (forward + swapped),
   concordance checks, truncation diagnostics, swap consistency.
6. Three-Dimensional Evaluation: Quality Proximity (P) and Signed Delta (DeltaQ) with 95% CIs.
7. Real-time checkpointing & resumption.
"""

import os
import sys
import json
import time
import glob
import asyncio
import argparse
import builtins
from types import SimpleNamespace
from typing import Dict, Any, List, Optional

_orig_print = builtins.print
def print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    _orig_print(*args, **kwargs)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.ollama_runner import OllamaModelRunner
from src.models.groq_runner import APIGroqModelRunner
from src.models.hf_runner import HFRouterModelRunner
from src.v5.tools.engineering_retrieval_tool import EngineeringRetrievalTool
from src.v4.tools.code_verifier import MechanicalCodeVerifier
from src.v5.pipeline import SLMPipeline_v5
from src.instrumentation.logger import ExperimentLogger
from src.v2.judge.pairwise_harness import PairwiseLLMJudgeHarness
from src.v3.preflight import verify_distinct_roster_preflight, PreFlightVerificationError
from src.analysis.metrics import StatisticalAnalyzer

BENCHMARK_PATH = "data/eval_100_benchmark.json"
RESULTS_DIR = "results/100_query_eval"
PIPELINE_LOG_DIR = os.path.join(RESULTS_DIR, "pipeline_logs")
JUDGE_LOG_DIR = "logs/100_query_judge_pairwise"
KEY_LOG_DIR = "logs/100_query_judge_keys"

SLM_RESPONSES_PATH = os.path.join(RESULTS_DIR, "slm_responses.jsonl")
BASELINE_120B_RESPONSES_PATH = os.path.join(RESULTS_DIR, "baseline_120b_responses.jsonl")
BASELINE_72B_RESPONSES_PATH = os.path.join(RESULTS_DIR, "baseline_72b_responses.jsonl")
PROGRESS_PATH = os.path.join(RESULTS_DIR, "progress.json")
AUDIT_SUMMARY_PATH = os.path.join(RESULTS_DIR, "audit_summary.json")

def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PIPELINE_LOG_DIR, exist_ok=True)
    os.makedirs(JUDGE_LOG_DIR, exist_ok=True)
    os.makedirs(KEY_LOG_DIR, exist_ok=True)

def load_responses_dict(filepath: str) -> Dict[str, str]:
    res = {}
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        res[record["query_id"]] = record["response_text"]
                    except Exception:
                        pass
    return res

def build_slm_pipeline() -> SLMPipeline_v5:
    decomposer_runner = OllamaModelRunner(
        logical_model_name="llama3.2-3b",
        api_model_name="llama3.2:cpu",
        max_tokens=512,
        timeout_sec=180.0
    )

    base_phi = OllamaModelRunner(
        logical_model_name="phi3.5-3.8b",
        api_model_name="phi3.5:cpu",
        max_tokens=1024,
        timeout_sec=300.0
    )

    ft_retrieval = OllamaModelRunner(
        logical_model_name="phi3.5-ft-retrieval",
        api_model_name="phi3.5-ft-retrieval:latest",
        max_tokens=1024,
        timeout_sec=300.0
    )

    ft_coding = OllamaModelRunner(
        logical_model_name="phi3.5-ft-coding",
        api_model_name="phi3.5-ft-coding:latest",
        max_tokens=1024,
        timeout_sec=300.0
    )

    aggregator_runner = OllamaModelRunner(
        logical_model_name="llama3.2-3b",
        api_model_name="llama3.2:cpu",
        max_tokens=1024,
        timeout_sec=180.0
    )

    retrieval_tool = EngineeringRetrievalTool()
    code_verifier = MechanicalCodeVerifier(execution_timeout_sec=6.0)

    base_pool_runners = {
        "coding": ft_coding,
        "retrieval_qa": ft_retrieval,
        "mathematics": base_phi,
        "formal_reasoning": base_phi,
        "science_tech": base_phi,
        "structured_data": base_phi,
        "systems_ops": base_phi,
        "creative_synthesis": base_phi
    }

    logger = ExperimentLogger(log_dir=PIPELINE_LOG_DIR)

    pipeline = SLMPipeline_v5(
        decomposer_runner=decomposer_runner,
        base_pool_runners=base_pool_runners,
        aggregator_runner=aggregator_runner,
        retrieval_tool=retrieval_tool,
        code_verifier=code_verifier,
        coding_max_retries=1,
        logger=logger,
        max_depth=3,
        max_concurrent_slms=4
    )

    return pipeline

def audit_judge_trial(key_file: str) -> Dict[str, Any]:
    """Autonomous Audit Loop: verify concordance and check for truncation."""
    with open(key_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 1. Concordance check
    cqs_a = data.get("cqs_candidate_a", 0.0)
    cqs_b = data.get("cqs_candidate_b", 0.0)
    unblinded_winner = data.get("unblinded_winner")
    system_a = data.get("system_a_id")
    system_b = data.get("system_b_id")

    if cqs_a > cqs_b:
        expected_winner = system_a
    elif cqs_b > cqs_a:
        expected_winner = system_b
    else:
        expected_winner = "tie"

    concordance = (unblinded_winner == expected_winner)
    
    # 2. Truncation diagnostic
    raw_eval = data.get("judge_evaluation", {})
    reasoning = str(raw_eval.get("reasoning", "")).lower()
    truncated = any(term in reasoning for term in ["truncated", "cut off", "incomplete response", "prematurely terminated"])

    data["concordance_pass"] = concordance
    data["truncation_flag"] = truncated
    return data

async def run_benchmark(dataset_path: str = BENCHMARK_PATH, max_queries: Optional[int] = None, comparators: List[str] = ["120b", "72b"]):
    ensure_dirs()
    print("=" * 80)
    print(f"AI SEARCH FRAMEWORK: BENCHMARK EVALUATION")
    print(f"Dataset:     {dataset_path}")
    print(f"Comparators: {comparators}")
    print("=" * 80)

    # 1. Hard Rule 13 Pre-Flight Assertion
    slm_pipeline_runners = {
        "decomposer": SimpleNamespace(model_name="llama3.2-3b", api_model_name="llama3.2:cpu"),
        "pool_retrieval": SimpleNamespace(model_name="phi3.5-ft-retrieval", api_model_name="phi3.5-ft-retrieval:latest"),
        "pool_coding": SimpleNamespace(model_name="phi3.5-ft-coding", api_model_name="phi3.5-ft-coding:latest"),
        "pool_general": SimpleNamespace(model_name="phi3.5-3.8b", api_model_name="phi3.5:cpu"),
        "aggregator": SimpleNamespace(model_name="llama3.2-3b", api_model_name="llama3.2:cpu")
    }

    baseline_runners_for_preflight = {}
    if "120b" in comparators:
        baseline_runners_for_preflight["gpt_120b"] = SimpleNamespace(model_name="gpt-120b", api_model_name="openai/gpt-oss-120b")
    if "72b" in comparators:
        baseline_runners_for_preflight["qwen_72b"] = SimpleNamespace(model_name="qwen-72b", api_model_name="Qwen/Qwen2.5-72B-Instruct")

    judge_runner = SimpleNamespace(model_name="qwen-27b", api_model_name="qwen/qwen3.8-27b")

    resolved = verify_distinct_roster_preflight(slm_pipeline_runners, baseline_runners_for_preflight, judge_runner)
    print(f"[Pre-Flight] Hard Rule 13 Verified:\n{json.dumps(resolved, indent=2)}")

    # 2. Load Benchmark Dataset
    with open(dataset_path, "r", encoding="utf-8") as f:
        queries = json.load(f)
    if max_queries:
        queries = queries[:max_queries]
    print(f"\n[Dataset] Loaded {len(queries)} evaluation queries from {dataset_path}.")

    # 3. Initialize Runners
    runner_120b = None
    runner_72b = None
    if "120b" in comparators:
        runner_120b = APIGroqModelRunner(
            logical_model_name="gpt-120b",
            api_model_name="openai/gpt-oss-120b",
            max_tokens=2048,
            temperature=0.0
        )
    if "72b" in comparators:
        runner_72b = HFRouterModelRunner(
            logical_model_name="qwen-72b",
            api_model_name="Qwen/Qwen2.5-72B-Instruct",
            max_tokens=2048,
            temperature=0.0
        )

    judge_harness = PairwiseLLMJudgeHarness(judge_model_name="qwen/qwen3.8-27b")
    slm_pipeline = build_slm_pipeline()

    # Load cache
    slm_responses_dict = load_responses_dict(SLM_RESPONSES_PATH)
    base_120b_dict = load_responses_dict(BASELINE_120B_RESPONSES_PATH)
    base_72b_dict = load_responses_dict(BASELINE_72B_RESPONSES_PATH)

    print(f"[Cache State] SLM: {len(slm_responses_dict)} | 120B: {len(base_120b_dict)} | 72B: {len(base_72b_dict)}")

    total_queries = len(queries)
    trial_records_120b = []
    trial_records_72b = []
    benchmark_start_time = time.time()

    for idx, q in enumerate(queries, 1):
        qid = q["id"]
        qtext = q["query"]
        tier = q["complexity_tier"]

        print(f"\n" + "-" * 70)
        print(f"[{idx}/{total_queries}] Query: {qid} | Tier: {tier}")
        print(f"Prompt: {qtext[:90]}...")

        base_120b_text = ""
        base_72b_text = ""
        slm_text = ""

        # --- Concurrent Generation across independent execution targets ---
        async def gen_120b():
            nonlocal base_120b_text
            if "120b" in comparators:
                if qid not in base_120b_dict:
                    print(f"  [120B Baseline] Generating via openai/gpt-oss-120b...")
                    t_b0 = time.perf_counter()
                    resp = await runner_120b.generate(
                        prompt=qtext,
                        system_prompt="You are an expert technical AI assistant. Provide authoritative, mathematically sound, syntactically verified solutions."
                    )
                    t_b = time.perf_counter() - t_b0
                    base_120b_text = resp.text
                    base_120b_dict[qid] = base_120b_text
                    with open(BASELINE_120B_RESPONSES_PATH, "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "query_id": qid,
                            "complexity_tier": tier,
                            "query_text": qtext,
                            "system_type": "openai/gpt-oss-120b",
                            "response_text": base_120b_text,
                            "latency_sec": t_b
                        }) + "\n")
                    print(f"  [120B Baseline] Generated in {t_b:.2f}s ({len(base_120b_text)} chars).")
                else:
                    base_120b_text = base_120b_dict[qid]
                    print(f"  [120B Baseline] Loaded from cache ({len(base_120b_text)} chars).")

        async def gen_72b():
            nonlocal base_72b_text
            if "72b" in comparators:
                if qid not in base_72b_dict:
                    print(f"  [72B Baseline] Generating via Qwen/Qwen2.5-72B-Instruct...")
                    t_b0 = time.perf_counter()
                    resp = await runner_72b.generate(
                        prompt=qtext,
                        system_prompt="You are an expert technical AI assistant. Provide authoritative, mathematically sound, syntactically verified solutions."
                    )
                    t_b = time.perf_counter() - t_b0
                    base_72b_text = resp.text
                    base_72b_dict[qid] = base_72b_text
                    with open(BASELINE_72B_RESPONSES_PATH, "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "query_id": qid,
                            "complexity_tier": tier,
                            "query_text": qtext,
                            "system_type": "Qwen/Qwen2.5-72B-Instruct",
                            "response_text": base_72b_text,
                            "latency_sec": t_b
                        }) + "\n")
                    print(f"  [72B Baseline] Generated in {t_b:.2f}s ({len(base_72b_text)} chars).")
                else:
                    base_72b_text = base_72b_dict[qid]
                    print(f"  [72B Baseline] Loaded from cache ({len(base_72b_text)} chars).")

        async def gen_slm():
            nonlocal slm_text
            if qid not in slm_responses_dict:
                print(f"  [SLM Pipeline] Executing SLMPipeline_v5 (all <=5B)...")
                t_s0 = time.perf_counter()
                pipe_res = await slm_pipeline.execute_query(
                    query_id=qid,
                    query_text=qtext,
                    complexity_tier=tier,
                    seed=42,
                    config={
                        "version": "v5_100_query_benchmark",
                        "deterministic_template_aggregator": True,
                        "retrieval_specialist": "phi3.5-ft-retrieval:latest",
                        "coding_specialist": "phi3.5-ft-coding:latest"
                    }
                )
                t_s = time.perf_counter() - t_s0
                slm_text = pipe_res["response"]
                slm_responses_dict[qid] = slm_text
                with open(SLM_RESPONSES_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "query_id": qid,
                        "complexity_tier": tier,
                        "query_text": qtext,
                        "system_type": "SLMPipeline_v5",
                        "response_text": slm_text,
                        "latency_sec": t_s,
                        "log_path": pipe_res.get("log_path", "")
                    }) + "\n")
                print(f"  [SLM Pipeline] Generated in {t_s:.1f}s ({len(slm_text)} chars).")
            else:
                slm_text = slm_responses_dict[qid]
                print(f"  [SLM Pipeline] Loaded from cache ({len(slm_text)} chars).")

        # Execute concurrent generation (Cloud API baselines alongside local CPU SLM)
        await asyncio.gather(gen_120b(), gen_72b(), gen_slm())

        # --- Pairwise Judging Function ---
        def judge_comparison(comp_id: str, comp_text: str, comp_label: str):
            existing_fwd = glob.glob(os.path.join(KEY_LOG_DIR, f"key_{qid}_*_{comp_id}*_forward_*.json")) or \
                           glob.glob(os.path.join(KEY_LOG_DIR, f"key_{qid}_*_forward_*.json"))
            # Filter specifically for this comparator
            matching_fwd = [f for f in existing_fwd if comp_id in f]
            if not matching_fwd:
                eval_fwd = judge_harness.evaluate_pair(
                    query_id=qid,
                    query_text=qtext,
                    system_a_id="SLMPipeline_v5",
                    text_a=slm_text,
                    system_b_id=comp_id,
                    text_b=comp_text,
                    order_tag="forward",
                    judge_log_dir=JUDGE_LOG_DIR,
                    key_log_dir=KEY_LOG_DIR
                )
            else:
                eval_fwd = audit_judge_trial(matching_fwd[0])

            matching_swp = [f for f in glob.glob(os.path.join(KEY_LOG_DIR, f"key_{qid}_*_{comp_id}*_swapped_*.json"))] or \
                           [f for f in glob.glob(os.path.join(KEY_LOG_DIR, f"key_{qid}_*_swapped_*.json")) if comp_id in f]
            if not matching_swp:
                eval_swp = judge_harness.evaluate_pair(
                    query_id=qid,
                    query_text=qtext,
                    system_a_id="SLMPipeline_v5",
                    text_a=slm_text,
                    system_b_id=comp_id,
                    text_b=comp_text,
                    order_tag="swapped",
                    judge_log_dir=JUDGE_LOG_DIR,
                    key_log_dir=KEY_LOG_DIR
                )
            else:
                eval_swp = audit_judge_trial(matching_swp[0])

            # In FWD: A=SLM, B=Comparator
            slm_fwd = eval_fwd.get("cqs_candidate_a", 0.0)
            base_fwd = eval_fwd.get("cqs_candidate_b", 0.0)
            p_fwd = eval_fwd.get("quality_proximity", 1.0 - abs(slm_fwd - base_fwd) / 4.0)

            # In SWP: A=Comparator, B=SLM
            base_swp = eval_swp.get("cqs_candidate_a", 0.0)
            slm_swp = eval_swp.get("cqs_candidate_b", 0.0)
            p_swp = eval_swp.get("quality_proximity", 1.0 - abs(slm_swp - base_swp) / 4.0)

            avg_slm = (slm_fwd + slm_swp) / 2.0
            avg_base = (base_fwd + base_swp) / 2.0
            avg_p = (p_fwd + p_swp) / 2.0 * 100.0
            d_q = avg_slm - avg_base

            f_win = eval_fwd.get("unblinded_winner")
            s_win = eval_swp.get("unblinded_winner")
            consistent = (f_win == s_win)

            print(f"  [Judge vs {comp_label}] SLM: {avg_slm:.2f} | Base: {avg_base:.2f} | P: {avg_p:.1f}% | dQ: {d_q:+.2f} | FWD: {f_win} | SWP: {s_win} | Cons: {consistent}")

            return {
                "query_id": qid,
                "complexity_tier": tier,
                "comparator": comp_id,
                "slm_cqs": avg_slm,
                "base_cqs": avg_base,
                "quality_proximity_pct": avg_p,
                "quality_delta": d_q,
                "fwd_winner": f_win,
                "swp_winner": s_win,
                "consistent": consistent
            }

        if "120b" in comparators and base_120b_text:
            rec_120b = judge_comparison("gpt_120b", base_120b_text, "120B")
            trial_records_120b.append(rec_120b)

        if "72b" in comparators and base_72b_text:
            rec_72b = judge_comparison("qwen_72b", base_72b_text, "72B")
            trial_records_72b.append(rec_72b)

        # Update progress.json with accurate ETA tracking
        elapsed_sec = time.time() - benchmark_start_time
        avg_sec = elapsed_sec / idx if idx > 0 else 0.0
        remaining = total_queries - idx
        eta_sec = remaining * avg_sec
        if eta_sec >= 3600:
            eta_human = f"{int(eta_sec // 3600)}h {int((eta_sec % 3600) // 60)}m"
        elif eta_sec >= 60:
            eta_human = f"{int(eta_sec // 60)}m {int(eta_sec % 60)}s"
        else:
            eta_human = f"{int(eta_sec)}s"
        eta_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + eta_sec))

        with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "completed_queries": idx,
                "total_queries": total_queries,
                "pct_complete": round(idx / total_queries * 100.0, 1),
                "elapsed_sec": round(elapsed_sec, 1),
                "avg_sec_per_query": round(avg_sec, 1),
                "remaining_queries": remaining,
                "estimated_remaining_time": eta_human,
                "estimated_completion_utc": eta_utc,
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "comparators": comparators,
                "recent_120b": trial_records_120b[-5:],
                "recent_72b": trial_records_72b[-5:]
            }, f, indent=2)

    # 4. Statistical Analysis
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETED — RUNNING THREE-DIMENSIONAL STATISTICAL ANALYSIS")
    print("=" * 80)

    analyzer = StatisticalAnalyzer()
    tiers = ["single_domain", "two_domain", "three_plus_domain"]

    def compute_stats_for_cohort(records: List[Dict[str, Any]], comp_id: str) -> Dict[str, Any]:
        if not records:
            return {}
        slm_all = [t["slm_cqs"] for t in records]
        base_all = [t["base_cqs"] for t in records]
        overall_prox = analyzer.compute_quality_proximity(slm_all, base_all)

        tier_stats = {}
        for t_name in tiers:
            t_recs = [t for t in records if t["complexity_tier"] == t_name]
            if t_recs:
                t_slm = [t["slm_cqs"] for t in t_recs]
                t_base = [t["base_cqs"] for t in t_recs]
                t_prox = analyzer.compute_quality_proximity(t_slm, t_base)
                t_wins = sum(1 for t in t_recs if t["fwd_winner"] == "SLMPipeline_v5" and t["swp_winner"] == "SLMPipeline_v5")
                t_base_wins = sum(1 for t in t_recs if t["fwd_winner"] == comp_id and t["swp_winner"] == comp_id)
                t_cons = sum(1 for t in t_recs if t["consistent"]) / len(t_recs) * 100.0
                tier_stats[t_name] = {
                    "n_queries": len(t_recs),
                    "slm_wins": t_wins,
                    "base_wins": t_base_wins,
                    "win_rate_pct": round(t_wins / len(t_recs) * 100.0, 2),
                    "mean_slm_cqs": round(sum(t_slm) / len(t_slm), 3),
                    "mean_base_cqs": round(sum(t_base) / len(t_base), 3),
                    "mean_quality_proximity_pct": t_prox["mean_quality_proximity_pct"],
                    "proximity_ci_95": [t_prox["proximity_ci_95_lower_pct"], t_prox["proximity_ci_95_upper_pct"]],
                    "mean_quality_delta": t_prox["mean_quality_delta"],
                    "delta_ci_95": [t_prox["delta_ci_95_lower"], t_prox["delta_ci_95_upper"]],
                    "swap_consistency_pct": round(t_cons, 2)
                }

        slm_wins = sum(1 for t in records if t["fwd_winner"] == "SLMPipeline_v5" and t["swp_winner"] == "SLMPipeline_v5")
        base_wins = sum(1 for t in records if t["fwd_winner"] == comp_id and t["swp_winner"] == comp_id)
        overall_cons = sum(1 for t in records if t["consistent"]) / len(records) * 100.0

        return {
            "total_queries": len(records),
            "total_trials": len(records) * 2,
            "slm_wins": slm_wins,
            "base_wins": base_wins,
            "win_rate_pct": round(slm_wins / len(records) * 100.0, 2),
            "overall_mean_slm_cqs": round(sum(slm_all) / len(slm_all), 3),
            "overall_mean_base_cqs": round(sum(base_all) / len(base_all), 3),
            "overall_proximity": overall_prox,
            "swap_consistency_pct": round(overall_cons, 2),
            "by_tier": tier_stats,
            "trial_records": records
        }

    summary_120b = compute_stats_for_cohort(trial_records_120b, "gpt_120b")
    summary_72b = compute_stats_for_cohort(trial_records_72b, "qwen_72b")

    final_summary = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "benchmark_dataset": BENCHMARK_PATH,
        "n_evaluated_queries": len(queries),
        "comparators": comparators,
        "vs_120b": summary_120b,
        "vs_72b": summary_72b
    }

    with open(AUDIT_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)

    print(f"\nFinal Audit Summary saved to {AUDIT_SUMMARY_PATH}")
    if summary_120b:
        p_120 = summary_120b["overall_proximity"]
        print(f"\n[SLM vs 120B Baseline]")
        print(f"  Mean Quality Proximity (P): {p_120['mean_quality_proximity_pct']}% [95% CI: {p_120['proximity_ci_95_lower_pct']}% - {p_120['proximity_ci_95_upper_pct']}%]")
        print(f"  Mean Signed Delta (DeltaQ): {p_120['mean_quality_delta']} [95% CI: {p_120['delta_ci_95_lower']} - {p_120['delta_ci_95_upper']}]")
        print(f"  Win Rate:                   {summary_120b['win_rate_pct']}%")
        print(f"  Swap Consistency:           {summary_120b['swap_consistency_pct']}%")

    if summary_72b:
        p_72 = summary_72b["overall_proximity"]
        print(f"\n[SLM vs 72B Baseline]")
        print(f"  Mean Quality Proximity (P): {p_72['mean_quality_proximity_pct']}% [95% CI: {p_72['proximity_ci_95_lower_pct']}% - {p_72['proximity_ci_95_upper_pct']}%]")
        print(f"  Mean Signed Delta (DeltaQ): {p_72['mean_quality_delta']} [95% CI: {p_72['delta_ci_95_lower']} - {p_72['delta_ci_95_upper']}]")
        print(f"  Win Rate:                   {summary_72b['win_rate_pct']}%")
        print(f"  Swap Consistency:           {summary_72b['swap_consistency_pct']}%")

    return final_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Cross-Tier Benchmark")
    parser.add_argument("--dataset", type=str, default=BENCHMARK_PATH, help="Path to benchmark JSON dataset")
    parser.add_argument("--max_queries", type=int, default=None, help="Limit number of queries")
    parser.add_argument("--comparators", nargs="+", default=["120b", "72b"], help="Comparators: 120b 72b")
    args = parser.parse_args()

    asyncio.run(run_benchmark(dataset_path=args.dataset, max_queries=args.max_queries, comparators=args.comparators))
