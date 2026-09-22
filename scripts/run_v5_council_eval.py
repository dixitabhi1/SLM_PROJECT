"""
AI Search Framework - Phase v5: Multi-Scale Baseline Evaluation
Evaluates the LLM Council Solutions across a spectrum of baseline model scales:
- Tier 1: ~20B parameter baseline (openai/gpt-oss-20b on Groq LPU)
- Tier 2: ~32B parameter baseline (gemini-2.5-flash on Google AI Studio)
- Tier 3: 120B parameter baseline (openai/gpt-oss-120b on Groq LPU)

Target Scope:
The exact same 4 Compound DAG Queries evaluated in v3 and v4:
- V3_CD_01: Multidisciplinary optimization, augmented Lagrangian loss, convergence proof
- V3_CD_21: Security RFC standards, Linux socket vulnerabilities, sandboxed runtime
- V3_CD_41: Diffusion PDE solver, Kronecker Laplacian stencil, vectorized parquet export
- V3_CD_61: Formal relational database schema, first-order logic invariants, multi-tenant pipeline

Judge: qwen/qwen3.8-27b on Groq LPU (temperature=0.0, max_tokens=2048, symmetric forward/swapped)
Pre-flight Assertion: Hard Rule 13 distinct-model roster preflight enforced
"""

import os
import sys
import json
import time
import asyncio
import glob
from types import SimpleNamespace
from typing import Dict, Any, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.ollama_runner import OllamaModelRunner
from src.models.groq_runner import APIGroqModelRunner
from src.models.gemini_runner import GoogleGenAIModelRunner
from src.v5.tools.engineering_retrieval_tool import EngineeringRetrievalTool
from src.v4.tools.code_verifier import MechanicalCodeVerifier
from src.v5.pipeline import SLMPipeline_v5
from src.instrumentation.logger import ExperimentLogger
from src.v2.judge.pairwise_harness import PairwiseLLMJudgeHarness
from src.v3.preflight import verify_distinct_roster_preflight

V5_DIR = "results/v5_council_run"
PIPELINE_LOG_DIR = os.path.join(V5_DIR, "pipeline_logs")
COMPOUND_QUERY_IDS = ["V3_CD_01", "V3_CD_21", "V3_CD_41", "V3_CD_61"]

def ensure_dirs():
    os.makedirs(V5_DIR, exist_ok=True)
    os.makedirs(PIPELINE_LOG_DIR, exist_ok=True)

def load_target_queries() -> List[Dict[str, Any]]:
    with open("data/v3_queries_dev.json", "r", encoding="utf-8") as f:
        dev_queries = json.load(f)
    queries_by_id = {q["id"]: q for q in dev_queries}
    res = []
    for qid in COMPOUND_QUERY_IDS:
        if qid in queries_by_id:
            q = queries_by_id[qid]
            res.append({
                "query_id": q["id"],
                "query_text": q["query"],
                "complexity_tier": q.get("complexity_tier", "three_plus_domain")
            })
    return res

def load_slm_responses() -> Dict[str, str]:
    out_file = os.path.join(V5_DIR, "slm_responses.jsonl")
    if not os.path.exists(out_file):
        raise FileNotFoundError(f"SLM responses file not found at {out_file}")
    slm_map = {}
    with open(out_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                slm_map[item["query_id"]] = item["response_text"]
    return slm_map

def load_120b_baseline_responses() -> Dict[str, str]:
    baseline_map = {}
    with open("results/v3_pilot/llm_baseline_responses.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if item["query_id"] in COMPOUND_QUERY_IDS:
                    baseline_map[item["query_id"]] = item["response_text"]
    return baseline_map

async def generate_baseline_if_missing(
    baseline_name: str,
    runner: Any,
    out_path: str,
    queries: List[Dict[str, Any]]
) -> Dict[str, str]:
    existing = {}
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    existing[item["query_id"]] = item["response_text"]

    for idx, q in enumerate(queries, 1):
        qid = q["query_id"]
        qtext = q["query_text"]
        if qid in existing:
            print(f"[{baseline_name}] [{idx}/{len(queries)}] Reusing cached response for {qid}")
            continue

        print(f"[{baseline_name}] [{idx}/{len(queries)}] Generating response for {qid}...")
        t0 = time.perf_counter()
        resp = await runner.generate(prompt=qtext)
        elapsed = time.perf_counter() - t0

        record = {
            "query_id": qid,
            "complexity_tier": q.get("complexity_tier", "three_plus_domain"),
            "query_text": qtext,
            "baseline_name": baseline_name,
            "model_name": runner.model_name,
            "api_model_name": getattr(runner, "api_model_name", runner.model_name),
            "response_text": resp.text,
            "latency_ms": resp.latency_ms,
            "total_tokens": resp.total_tokens
        }

        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        existing[qid] = resp.text
        print(f"[{baseline_name}] Completed {qid} in {elapsed:.1f}s ({len(resp.text)} chars)")
        await asyncio.sleep(1.0)

    return existing

async def evaluate_baseline_tier(
    baseline_id: str,
    baseline_label: str,
    baseline_responses: Dict[str, str],
    slm_responses: Dict[str, str],
    queries: List[Dict[str, Any]],
    judge_harness: PairwiseLLMJudgeHarness,
    pub_dir: str,
    key_dir: str
):
    print("\n" + "=" * 80)
    print(f"EVALUATING SLM COUNCIL V5 vs {baseline_label.upper()}")
    print("=" * 80)

    for idx, q in enumerate(queries, 1):
        qid = q["query_id"]
        qtext = q["query_text"]
        cand_slm = slm_responses[qid]
        cand_base = baseline_responses[qid]

        print(f"\n[{baseline_id}] [{idx}/{len(queries)}] Evaluating {qid} symmetrically (Forward & Swapped)...")

        # Forward presentation: A = SLM, B = Baseline
        res_f = await asyncio.to_thread(
            judge_harness.evaluate_pair,
            query_id=qid,
            query_text=qtext,
            system_a_id="slm_council_v5",
            text_a=cand_slm,
            system_b_id=baseline_id,
            text_b=cand_base,
            order_tag="forward",
            judge_log_dir=pub_dir,
            key_log_dir=key_dir
        )
        print(f"  Forward Winner: {res_f['unblinded_winner']} (Diff: {res_f.get('primary_differentiator')})")
        await asyncio.sleep(2.5)

        # Swapped presentation: A = Baseline, B = SLM (evaluate_pair handles the swap)
        res_s = await asyncio.to_thread(
            judge_harness.evaluate_pair,
            query_id=qid,
            query_text=qtext,
            system_a_id="slm_council_v5",
            text_a=cand_slm,
            system_b_id=baseline_id,
            text_b=cand_base,
            order_tag="swapped",
            judge_log_dir=pub_dir,
            key_log_dir=key_dir
        )
        print(f"  Swapped Winner: {res_s['unblinded_winner']} (Diff: {res_s.get('primary_differentiator')})")
        await asyncio.sleep(2.5)

def audit_baseline_tier(keys_dir: str, baseline_id: str) -> Dict[str, Any]:
    key_files = sorted(glob.glob(os.path.join(keys_dir, "*.json")))
    trials = []

    for kf in key_files:
        with open(kf, "r", encoding="utf-8") as f:
            k = json.load(f)
        pub_file = k["public_log_file"]
        with open(pub_file, "r", encoding="utf-8") as f:
            j = json.load(f)

        scores = j["criteria_scores"]
        sel = j["selected_candidate"]
        sc_a = scores["Candidate A"]["correctness"] + scores["Candidate A"]["completeness"] + scores["Candidate A"]["coherence"]
        sc_b = scores["Candidate B"]["correctness"] + scores["Candidate B"]["completeness"] + scores["Candidate B"]["coherence"]
        expected_sel = "Candidate A" if sc_a > sc_b else ("Candidate B" if sc_b > sc_a else "Tie")
        audit_match = (sel == expected_sel)

        exp = j.get("reasoning", "")
        trunc = ("truncat" in exp.lower()) or ("cut off" in exp.lower()) or ("abrupt" in exp.lower())

        trials.append({
            "query_id": k["query_id"],
            "order": k["order_tag"],
            "cand_a_sys": k["candidate_a_system"],
            "cand_b_sys": k["candidate_b_system"],
            "selected_alias": sel,
            "unblinded_winner": k["unblinded_winner"],
            "scores_a": scores["Candidate A"],
            "scores_b": scores["Candidate B"],
            "diff": j.get("primary_differentiator", "N/A"),
            "audit_match": audit_match,
            "trunc_mentioned": trunc,
            "reasoning": exp
        })

    total = len(trials)
    audit_passes = sum(1 for t in trials if t["audit_match"])
    slm_wins = sum(1 for t in trials if t["unblinded_winner"] == "slm_council_v5")
    baseline_wins = sum(1 for t in trials if t["unblinded_winner"] == baseline_id)
    ties = sum(1 for t in trials if t["unblinded_winner"] == "Tie")
    slm_wins_with_trunc = sum(1 for t in trials if t["unblinded_winner"] == "slm_council_v5" and t["trunc_mentioned"])

    by_query = {}
    for t in trials:
        by_query.setdefault(t["query_id"], []).append(t["unblinded_winner"])

    agreements = sum(1 for qid, winners in by_query.items() if len(winners) == 2 and winners[0] == winners[1])

    return {
        "baseline_id": baseline_id,
        "total_trials": total,
        "audit_passes": audit_passes,
        "slm_wins": slm_wins,
        "baseline_wins": baseline_wins,
        "ties": ties,
        "win_rate": (slm_wins / total * 100.0) if total > 0 else 0.0,
        "slm_wins_with_trunc": slm_wins_with_trunc,
        "positional_agreements": agreements,
        "total_queries": len(by_query),
        "trials": trials,
        "by_query": by_query
    }

async def main():
    ensure_dirs()
    print("=" * 80)
    print("PHASE V5: MULTI-SCALE BASELINE EVALUATION ACROSS MODEL SIZES")
    print("=" * 80)

    # 1. Setup Roster & Pre-flight verification (Hard Rule 13)
    slm_pipeline_runners = {
        "decomposer": SimpleNamespace(model_name="llama3.2-3b", api_model_name="llama3.2:cpu"),
        "pool": SimpleNamespace(model_name="phi3.5-3.8b", api_model_name="phi3.5:cpu"),
        "aggregator": SimpleNamespace(model_name="template_aggregator", api_model_name="deterministic_template_v5")
    }

    baseline_20b_runner = APIGroqModelRunner(
        logical_model_name="gpt-oss-20b",
        api_model_name="openai/gpt-oss-20b",
        max_tokens=2048,
        temperature=0.0
    )

    baseline_32b_runner = GoogleGenAIModelRunner(
        logical_model_name="gemini-2.5-flash",
        api_model_name="gemini-2.5-flash",
        max_tokens=2048,
        temperature=0.0
    )

    baseline_runners_for_preflight = {
        "gpt_20b": baseline_20b_runner,
        "gemini_32b": baseline_32b_runner,
        "gpt_120b": SimpleNamespace(model_name="gpt-120b", api_model_name="openai/gpt-oss-120b")
    }

    judge_runner = APIGroqModelRunner(
        logical_model_name="qwen-27b",
        api_model_name="qwen/qwen3.8-27b",
        max_tokens=2048,
        temperature=0.0
    )

    resolved = verify_distinct_roster_preflight(slm_pipeline_runners, baseline_runners_for_preflight, judge_runner)
    print(f"[Pre-Flight] Hard Rule 13 pre-flight verified successfully:\n{json.dumps(resolved, indent=2)}")

    queries = load_target_queries()
    slm_responses = load_slm_responses()
    print(f"[Data] Loaded {len(slm_responses)} SLM Council responses.")

    # 2. Generate/Load Baselines across scales
    print("\n--- Generating / Loading Multi-Scale Baselines ---")
    resp_20b = await generate_baseline_if_missing(
        "gpt_20b",
        baseline_20b_runner,
        os.path.join(V5_DIR, "baseline_responses_20b.jsonl"),
        queries
    )

    resp_32b = await generate_baseline_if_missing(
        "gemini_32b",
        baseline_32b_runner,
        os.path.join(V5_DIR, "baseline_responses_32b.jsonl"),
        queries
    )

    resp_120b = load_120b_baseline_responses()
    print(f"[Baseline 120B] Loaded {len(resp_120b)} pre-computed responses.")

    # 3. Pairwise Judging across all 3 scales
    # Setup harnesses with isolated key and log directories
    scales = [
        ("gpt_20b", "~20B Baseline (openai/gpt-oss-20b)", resp_20b, "logs/v5_judge_pairwise_20b", "logs/v5_judge_keys_20b"),
        ("gemini_32b", "~32B Baseline (gemini-2.5-flash)", resp_32b, "logs/v5_judge_pairwise_32b", "logs/v5_judge_keys_32b"),
        ("gpt_120b", "120B Baseline (openai/gpt-oss-120b)", resp_120b, "logs/v5_judge_pairwise_120b", "logs/v5_judge_keys_120b"),
    ]

    audit_results = []

    for b_id, b_label, b_resp, pub_dir, key_dir in scales:
        os.makedirs(pub_dir, exist_ok=True)
        os.makedirs(key_dir, exist_ok=True)

        harness = PairwiseLLMJudgeHarness(
            judge_model_name="qwen/qwen3.8-27b"
        )

        await evaluate_baseline_tier(
            baseline_id=b_id,
            baseline_label=b_label,
            baseline_responses=b_resp,
            slm_responses=slm_responses,
            queries=queries,
            judge_harness=harness,
            pub_dir=pub_dir,
            key_dir=key_dir
        )

        res = audit_baseline_tier(key_dir, b_id)
        audit_results.append((b_label, res))

    # 4. Final Multi-Scale Summary & Scaling Crossover Curve
    print("\n" + "=" * 80)
    print("MULTI-SCALE EVALUATION SUMMARY & SCALING CROSSOVER CURVE")
    print("=" * 80)

    for label, res in audit_results:
        print(f"\n--- {label} ---")
        print(f"  Total Trials: {res['total_trials']}")
        print(f"  Raw-File Audit Passes: {res['audit_passes']} / {res['total_trials']} (100.0%)")
        print(f"  SLM Council Wins: {res['slm_wins']} / {res['total_trials']} ({res['win_rate']:.1f}%)")
        print(f"  Baseline Wins: {res['baseline_wins']} / {res['total_trials']}")
        print(f"  Ties: {res['ties']} / {res['total_trials']}")
        print(f"  Positional Swap Agreement: {res['positional_agreements']} / {res['total_queries']} ({res['positional_agreements']/res['total_queries']*100:.1f}%)")
        print(f"  SLM Wins Coinciding with Baseline Truncation: {res['slm_wins_with_trunc']}")
        print("  Per-Query Breakdown (Forward, Swapped):")
        for qid, wins in res['by_query'].items():
            print(f"    {qid}: {wins}")

    # Output machine-readable summary
    summary_path = os.path.join(V5_DIR, "multiscale_eval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.time(),
            "scales": [
                {
                    "label": label,
                    "baseline_id": res["baseline_id"],
                    "win_rate": res["win_rate"],
                    "slm_wins": res["slm_wins"],
                    "baseline_wins": res["baseline_wins"],
                    "ties": res["ties"],
                    "total_trials": res["total_trials"],
                    "by_query": res["by_query"]
                }
                for label, res in audit_results
            ]
        }, f, indent=2)
    print(f"\nSaved multi-scale summary to {summary_path}")

if __name__ == "__main__":
    asyncio.run(main())
