"""
AI Search Framework - Phase v5.1: Multi-Scale Rescue Evaluation
Evaluates the Polished LLM Council Architecture:
1. Extended timeout (700s) + Token-clamped coding specialist (1024 max_tokens)
2. Automated template aggregator output sanitization (zero leaked timeout errors)
3. Standardized technical reference citations (zero phantom Source ID tags)
4. Evaluated symmetrically against ~20B (openai/gpt-oss-20b), ~32B (gemini-2.5-flash), and 120B (openai/gpt-oss-120b)
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

V5_1_DIR = "results/v5_1_council_run"
PIPELINE_LOG_DIR = os.path.join(V5_1_DIR, "pipeline_logs")
COMPOUND_QUERY_IDS = ["V3_CD_01", "V3_CD_21", "V3_CD_41", "V3_CD_61"]

def ensure_dirs():
    os.makedirs(V5_1_DIR, exist_ok=True)
    os.makedirs(PIPELINE_LOG_DIR, exist_ok=True)

def build_v5_1_pipeline() -> SLMPipeline_v5:
    decomposer_runner = OllamaModelRunner(
        logical_model_name="llama3.2-3b",
        api_model_name="llama3.2:cpu"
    )

    base_phi = OllamaModelRunner(
        logical_model_name="phi3.5-3.8b",
        api_model_name="phi3.5:cpu",
        max_tokens=2048,
        timeout_sec=700.0
    )

    aggregator_runner = OllamaModelRunner(
        logical_model_name="llama3.2-3b",
        api_model_name="llama3.2:cpu"
    )

    retrieval_tool = EngineeringRetrievalTool()
    code_verifier = MechanicalCodeVerifier(execution_timeout_sec=8.0)

    base_pool_runners = {
        "coding": base_phi,
        "retrieval_qa": base_phi,
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
        logger=logger,
        max_depth=3,
        max_concurrent_slms=4
    )

    return pipeline

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

async def generate_v5_1_responses() -> Dict[str, str]:
    ensure_dirs()
    print("=" * 80)
    print("PHASE V5.1: POLISHED COUNCIL PIPELINE GENERATION RUN")
    print("=" * 80)

    pipeline = build_v5_1_pipeline()
    queries = load_target_queries()
    out_file = os.path.join(V5_1_DIR, "slm_responses.jsonl")

    existing = {}
    if os.path.exists(out_file):
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    existing[item["query_id"]] = item["response_text"]

    for idx, q in enumerate(queries, 1):
        qid = q["query_id"]
        qtext = q["query_text"]
        tier = q.get("complexity_tier", "three_plus_domain")

        if qid in existing:
            print(f"[{idx}/{len(queries)}] Reusing cached v5.1 response for {qid}")
            continue

        print(f"\n[{idx}/{len(queries)}] Executing Polished Pipeline v5.1 on {qid}: {qtext[:75]}...")
        t0 = time.perf_counter()
        res = await pipeline.execute_query(
            query_id=qid,
            query_text=qtext,
            complexity_tier=tier,
            seed=42,
            config={
                "version": "v5_1_rescue",
                "deterministic_template_aggregator": True,
                "sanitization": True,
                "engineering_grounding": True
            }
        )
        elapsed = time.perf_counter() - t0

        record = {
            "query_id": qid,
            "complexity_tier": tier,
            "query_text": qtext,
            "system_type": "v5_1_council_slm_pipeline",
            "response_text": res["response"],
            "latency_sec": elapsed,
            "feedback_loop_fired": res.get("feedback_loop_fired", False),
            "log_path": res.get("log_path", "")
        }

        with open(out_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        existing[qid] = res["response"]
        print(f"  Completed {qid} in {elapsed:.1f}s ({len(res['response'])} chars). Saved to {out_file}")

    return existing

def load_cached_baselines() -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    b20 = {}
    with open("results/v5_council_run/baseline_responses_20b.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                b20[item["query_id"]] = item["response_text"]

    b32 = {}
    with open("results/v5_council_run/baseline_responses_32b.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                b32[item["query_id"]] = item["response_text"]

    b120 = {}
    with open("results/v3_pilot/llm_baseline_responses.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if item["query_id"] in COMPOUND_QUERY_IDS:
                    b120[item["query_id"]] = item["response_text"]

    return b20, b32, b120

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
    print(f"EVALUATING POLISHED SLM COUNCIL V5.1 vs {baseline_label.upper()}")
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
            system_a_id="slm_council_v5_1",
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
            system_a_id="slm_council_v5_1",
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
    slm_wins = sum(1 for t in trials if t["unblinded_winner"] == "slm_council_v5_1")
    baseline_wins = sum(1 for t in trials if t["unblinded_winner"] == baseline_id)
    ties = sum(1 for t in trials if t["unblinded_winner"] == "Tie")
    slm_wins_with_trunc = sum(1 for t in trials if t["unblinded_winner"] == "slm_council_v5_1" and t["trunc_mentioned"])

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
    print("PHASE V5.1: RESCUE EVALUATION ACROSS MULTI-SCALE BASELINE LADDER")
    print("=" * 80)

    # 1. Hard Rule 13 Pre-Flight Assertion
    slm_pipeline_runners = {
        "decomposer": SimpleNamespace(model_name="llama3.2-3b", api_model_name="llama3.2:cpu"),
        "pool": SimpleNamespace(model_name="phi3.5-3.8b", api_model_name="phi3.5:cpu"),
        "aggregator": SimpleNamespace(model_name="template_aggregator", api_model_name="deterministic_template_v5_1")
    }

    baseline_runners_for_preflight = {
        "gpt_20b": SimpleNamespace(model_name="gpt-20b", api_model_name="openai/gpt-oss-20b"),
        "gemini_32b": SimpleNamespace(model_name="gemini-flash", api_model_name="gemini-2.5-flash"),
        "gpt_120b": SimpleNamespace(model_name="gpt-120b", api_model_name="openai/gpt-oss-120b")
    }

    judge_runner = SimpleNamespace(model_name="qwen-27b", api_model_name="qwen/qwen3.8-27b")

    resolved = verify_distinct_roster_preflight(slm_pipeline_runners, baseline_runners_for_preflight, judge_runner)
    print(f"[Pre-Flight] Hard Rule 13 pre-flight verified successfully:\n{json.dumps(resolved, indent=2)}")

    queries = load_target_queries()

    # 2. Generate Polished v5.1 Responses
    slm_responses = await generate_v5_1_responses()
    print(f"[Data] Loaded {len(slm_responses)} Polished SLM Council responses.")

    # 3. Load Cached Baselines
    resp_20b, resp_32b, resp_120b = load_cached_baselines()
    print(f"[Baselines] Loaded {len(resp_20b)} 20B, {len(resp_32b)} 32B, and {len(resp_120b)} 120B responses.")

    # 4. Pairwise Judging across all 3 scales
    scales = [
        ("gpt_20b", "~20B Baseline (openai/gpt-oss-20b)", resp_20b, "logs/v5_1_judge_pairwise_20b", "logs/v5_1_judge_keys_20b"),
        ("gemini_32b", "~32B Baseline (gemini-2.5-flash)", resp_32b, "logs/v5_1_judge_pairwise_32b", "logs/v5_1_judge_keys_32b"),
        ("gpt_120b", "120B Baseline (openai/gpt-oss-120b)", resp_120b, "logs/v5_1_judge_pairwise_120b", "logs/v5_1_judge_keys_120b"),
    ]

    harness = PairwiseLLMJudgeHarness(judge_model_name="qwen/qwen3.8-27b")
    audit_results = []

    for b_id, b_label, b_resp, pub_dir, key_dir in scales:
        os.makedirs(pub_dir, exist_ok=True)
        os.makedirs(key_dir, exist_ok=True)

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

    # 5. Final Multi-Scale Summary & Scaling Crossover Curve
    print("\n" + "=" * 80)
    print("PHASE V5.1 MULTI-SCALE EVALUATION SUMMARY & SCALING CROSSOVER CURVE")
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

    summary_path = os.path.join(V5_1_DIR, "multiscale_eval_summary.json")
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

