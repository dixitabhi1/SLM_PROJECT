"""
AI Search Framework - Step B: Multi-Turn Execution Feedback for Coding Specialist
Evaluates multi-turn execution feedback (3-5 iterations with runtime tracebacks fed back)
isolated to target pair: V3_CD_01 and V3_CD_41:
1. Coding Specialist: Multi-turn execution feedback with traceback injection up to 4 attempts.
2. Science & Tech Specialist: Grounded with pinned science_tech_reference_corpus.json (from Step A).
3. Decomposer and Aggregator strictly UNTOUCHED (llama3.2:cpu, Deterministic Template Aggregator).
4. Symmetrically judged vs 20B, 32B, and 120B baselines with full 3-check audit.
5. Computes composite quality score: (correctness + completeness + coherence) / 3 (1-5 scale)
   measured against the audited Step A baseline: 2.000 / 5.00 (40.0%).
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
from src.v5.tools.engineering_retrieval_tool import EngineeringRetrievalTool
from src.v5.tools.coding_retrieval_tool import CodingRetrievalTool
from src.v5.tools.science_tech_retrieval_tool import ScienceTechRetrievalTool
from src.v4.tools.code_verifier import MechanicalCodeVerifier
from src.v5.pipeline import SLMPipeline_v5
from src.instrumentation.logger import ExperimentLogger
from src.v2.judge.pairwise_harness import PairwiseLLMJudgeHarness
from src.v3.preflight import verify_distinct_roster_preflight

STEP_B_DIR = "results/step_b_run"
PIPELINE_LOG_DIR = os.path.join(STEP_B_DIR, "pipeline_logs")
TARGET_QUERY_IDS = ["V3_CD_01", "V3_CD_41"]

def ensure_dirs():
    os.makedirs(STEP_B_DIR, exist_ok=True)
    os.makedirs(PIPELINE_LOG_DIR, exist_ok=True)

def build_step_b_pipeline(coding_max_retries: int = 3) -> SLMPipeline_v5:
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

    eng_retrieval_tool = EngineeringRetrievalTool()
    coding_retrieval_tool = CodingRetrievalTool()
    science_retrieval_tool = ScienceTechRetrievalTool()
    code_verifier = MechanicalCodeVerifier(execution_timeout_sec=10.0)

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
        retrieval_tool=eng_retrieval_tool,
        coding_retrieval_tool=coding_retrieval_tool,
        science_retrieval_tool=science_retrieval_tool,
        code_verifier=code_verifier,
        coding_max_retries=coding_max_retries,
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
    for qid in TARGET_QUERY_IDS:
        if qid in queries_by_id:
            q = queries_by_id[qid]
            res.append({
                "query_id": q["id"],
                "query_text": q["query"],
                "complexity_tier": q.get("complexity_tier", "three_plus_domain")
            })
    return res

async def generate_step_b_responses() -> Dict[str, str]:
    ensure_dirs()
    print("=" * 80)
    print("STEP B: MULTI-TURN EXECUTION FEEDBACK GENERATION RUN")
    print("=" * 80)

    pipeline = build_step_b_pipeline(coding_max_retries=3)
    queries = load_target_queries()
    out_file = os.path.join(STEP_B_DIR, "slm_responses.jsonl")

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
            print(f"[{idx}/{len(queries)}] Reusing cached Step B response for {qid}")
            continue

        print(f"\n[{idx}/{len(queries)}] Executing Step B Pipeline on {qid}: {qtext[:75]}...")
        t0 = time.perf_counter()
        res = await pipeline.execute_query(
            query_id=qid,
            query_text=qtext,
            complexity_tier=tier,
            seed=42,
            config={
                "step": "step_b_multi_turn_execution_feedback",
                "targeted_specialists": ["coding"],
                "coding_max_retries": 3,
                "deterministic_template_aggregator": True,
                "sanitization": True
            }
        )
        elapsed = time.perf_counter() - t0

        record = {
            "query_id": qid,
            "complexity_tier": tier,
            "query_text": qtext,
            "system_type": "step_b_multiturn_feedback_slm_pipeline",
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
                if item["query_id"] in TARGET_QUERY_IDS:
                    b20[item["query_id"]] = item["response_text"]

    b32 = {}
    with open("results/v5_council_run/baseline_responses_32b.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if item["query_id"] in TARGET_QUERY_IDS:
                    b32[item["query_id"]] = item["response_text"]

    b120 = {}
    with open("results/v3_pilot/llm_baseline_responses.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if item["query_id"] in TARGET_QUERY_IDS:
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
    print(f"EVALUATING STEP B SLM PIPELINE vs {baseline_label.upper()}")
    print("=" * 80)

    for idx, q in enumerate(queries, 1):
        qid = q["query_id"]
        qtext = q["query_text"]
        cand_slm = slm_responses[qid]
        cand_base = baseline_responses[qid]

        existing_keys = glob.glob(os.path.join(key_dir, "*.json"))
        existing_orders = set()
        for kf in existing_keys:
            try:
                with open(kf, "r", encoding="utf-8") as f:
                    k_data = json.load(f)
                    if k_data.get("query_id") == qid and k_data.get("unblinded_winner") not in [None, "ERROR"]:
                        existing_orders.add(k_data.get("order_tag"))
            except Exception:
                pass

        print(f"\n[{baseline_id}] [{idx}/{len(queries)}] Evaluating {qid} symmetrically (Forward & Swapped)...")

        # Forward presentation: A = SLM, B = Baseline
        if "forward" in existing_orders:
            print(f"  Forward trial already completed and cached for {qid}")
        else:
            res_f = await asyncio.to_thread(
                judge_harness.evaluate_pair,
                query_id=qid,
                query_text=qtext,
                system_a_id="step_b_multiturn_slm",
                text_a=cand_slm,
                system_b_id=baseline_id,
                text_b=cand_base,
                order_tag="forward",
                judge_log_dir=pub_dir,
                key_log_dir=key_dir
            )
            if res_f.get("status") == "SUCCESS":
                print(f"  Forward Winner: {res_f.get('unblinded_winner')} (Diff: {res_f.get('primary_differentiator')})")
            else:
                print(f"  Forward FAILED: {res_f.get('error_detail')}")
            await asyncio.sleep(2.5)

        # Swapped presentation: A = Baseline, B = SLM
        if "swapped" in existing_orders:
            print(f"  Swapped trial already completed and cached for {qid}")
        else:
            res_s = await asyncio.to_thread(
                judge_harness.evaluate_pair,
                query_id=qid,
                query_text=qtext,
                system_a_id="step_b_multiturn_slm",
                text_a=cand_slm,
                system_b_id=baseline_id,
                text_b=cand_base,
                order_tag="swapped",
                judge_log_dir=pub_dir,
                key_log_dir=key_dir
            )
            if res_s.get("status") == "SUCCESS":
                print(f"  Swapped Winner: {res_s.get('unblinded_winner')} (Diff: {res_s.get('primary_differentiator')})")
            else:
                print(f"  Swapped FAILED: {res_s.get('error_detail')}")
            await asyncio.sleep(2.5)

def audit_and_compute_scores(keys_dir: str, baseline_id: str) -> Dict[str, Any]:
    key_files = sorted(glob.glob(os.path.join(keys_dir, "*.json")))
    trials = []

    for kf in key_files:
        with open(kf, "r", encoding="utf-8") as f:
            k = json.load(f)
        if k.get("status") != "SUCCESS":
            continue
        pub_file = k["public_log_file"]
        with open(pub_file, "r", encoding="utf-8") as f:
            j = json.load(f)

        scores = j["criteria_scores"]
        sel = j["selected_candidate"]
        sc_a = scores["Candidate A"]["correctness"] + scores["Candidate A"]["completeness"] + scores["Candidate A"]["coherence"]
        sc_b = scores["Candidate B"]["correctness"] + scores["Candidate B"]["completeness"] + scores["Candidate B"]["coherence"]
        expected_sel = "Candidate A" if sc_a > sc_b else ("Candidate B" if sc_b > sc_a else "Tie")
        audit_match = (sel == expected_sel)

        # Audited winner
        if sc_a > sc_b:
            audited_winner = k["candidate_a_system"]
        elif sc_b > sc_a:
            audited_winner = k["candidate_b_system"]
        else:
            audited_winner = "Tie"

        # Extract SLM scores
        slm_cand = "Candidate A" if k["candidate_a_system"].startswith("step_b") else "Candidate B"
        slm_scores = scores[slm_cand]
        slm_composite = (slm_scores["correctness"] + slm_scores["completeness"] + slm_scores["coherence"]) / 3.0

        base_cand = "Candidate B" if slm_cand == "Candidate A" else "Candidate A"
        base_scores = scores[base_cand]
        base_composite = (base_scores["correctness"] + base_scores["completeness"] + base_scores["coherence"]) / 3.0

        exp = j.get("reasoning", "")
        trunc = ("truncat" in exp.lower()) or ("cut off" in exp.lower()) or ("abrupt" in exp.lower())

        trials.append({
            "query_id": k["query_id"],
            "order": k["order_tag"],
            "cand_a_sys": k["candidate_a_system"],
            "cand_b_sys": k["candidate_b_system"],
            "selected_alias": sel,
            "raw_winner": k["unblinded_winner"],
            "audited_winner": audited_winner,
            "slm_scores": slm_scores,
            "slm_composite": slm_composite,
            "base_scores": base_scores,
            "base_composite": base_composite,
            "diff": j.get("primary_differentiator", "N/A"),
            "audit_match": audit_match,
            "trunc_mentioned": trunc,
            "reasoning": exp
        })

    total = len(trials)
    audit_passes = sum(1 for t in trials if t["audit_match"])
    slm_wins = sum(1 for t in trials if t["audited_winner"] == "step_b_multiturn_slm")
    baseline_wins = sum(1 for t in trials if t["audited_winner"] == baseline_id)
    ties = sum(1 for t in trials if t["audited_winner"] == "Tie")
    slm_wins_with_trunc = sum(1 for t in trials if t["audited_winner"] == "step_b_multiturn_slm" and t["trunc_mentioned"])

    by_query = {}
    for t in trials:
        by_query.setdefault(t["query_id"], []).append(t)

    agreements = sum(1 for qid, t_list in by_query.items() if len(t_list) == 2 and t_list[0]["audited_winner"] == t_list[1]["audited_winner"])

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
    print("AI SEARCH FRAMEWORK: STEP B MULTI-TURN EXECUTION FEEDBACK EVALUATION")
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

    # 2. Generate Step B Responses
    slm_responses = await generate_step_b_responses()
    print(f"[Data] Loaded {len(slm_responses)} Step B responses.")

    # 3. Load Cached Baselines
    resp_20b, resp_32b, resp_120b = load_cached_baselines()
    print(f"[Baselines] Loaded {len(resp_20b)} 20B, {len(resp_32b)} 32B, and {len(resp_120b)} 120B responses.")

    # 4. Pairwise Judging across all 3 scales
    scales = [
        ("gpt_20b", "~20B Baseline (openai/gpt-oss-20b)", resp_20b, "logs/step_b_judge_pairwise_20b", "logs/step_b_judge_keys_20b"),
        ("gemini_32b", "~32B Baseline (gemini-2.5-flash)", resp_32b, "logs/step_b_judge_pairwise_32b", "logs/step_b_judge_keys_32b"),
        ("gpt_120b", "120B Baseline (openai/gpt-oss-120b)", resp_120b, "logs/step_b_judge_pairwise_120b", "logs/step_b_judge_keys_120b"),
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

        res = audit_and_compute_scores(key_dir, b_id)
        audit_results.append((b_label, res))

    # 5. Compile Score Matrix & Before/After Comparison
    print("\n" + "=" * 80)
    print("STEP B EVALUATION SUMMARY & BEFORE/AFTER COMPOSITE QUALITY AUDIT")
    print("=" * 80)

    # Collect all SLM composite scores across all 3 tiers
    all_scores_by_query = {"V3_CD_01": [], "V3_CD_41": []}
    all_trials = []

    for label, res in audit_results:
        print(f"\n--- {label} ---")
        print(f"  Total Trials: {res['total_trials']}")
        print(f"  Raw-File Audit Passes: {res['audit_passes']} / {res['total_trials']}")
        print(f"  Step B SLM Wins: {res['slm_wins']} / {res['total_trials']} ({res['win_rate']:.1f}%)")
        print(f"  Baseline Wins: {res['baseline_wins']} / {res['total_trials']}")
        print(f"  Ties: {res['ties']} / {res['total_trials']}")
        print(f"  Positional Swap Agreement: {res['positional_agreements']} / {res['total_queries']} ({res['positional_agreements']/res['total_queries']*100:.1f}%)")
        print(f"  SLM Wins Coinciding with Baseline Truncation: {res['slm_wins_with_trunc']}")
        
        for qid, t_list in res["by_query"].items():
            for t in t_list:
                all_scores_by_query[qid].append(t["slm_composite"])
                all_trials.append(t)
                print(f"    [{qid} | {t['order']}] Winner: {t['audited_winner']} | SLM: {t['slm_composite']:.2f}/5.0 (C={t['slm_scores']['correctness']}, Comp={t['slm_scores']['completeness']}, Coh={t['slm_scores']['coherence']}) vs Base: {t['base_composite']:.2f}/5.0 | Diff: {t['diff']}")

    # Baseline anchors: Step 6 Baseline and Step A Audited Baseline
    STEP6_BASELINES = {
        "V3_CD_01": 1.722,
        "V3_CD_41": 2.056,
        "target_pair": 1.889
    }
    STEPA_AUDITED_BASELINES = {
        "V3_CD_01": 1.667,
        "V3_CD_41": 2.333,
        "target_pair": 2.000
    }

    mean_01 = sum(all_scores_by_query["V3_CD_01"]) / len(all_scores_by_query["V3_CD_01"]) if all_scores_by_query["V3_CD_01"] else 0.0
    mean_41 = sum(all_scores_by_query["V3_CD_41"]) / len(all_scores_by_query["V3_CD_41"]) if all_scores_by_query["V3_CD_41"] else 0.0
    all_target_scores = all_scores_by_query["V3_CD_01"] + all_scores_by_query["V3_CD_41"]
    mean_target_pair = sum(all_target_scores) / len(all_target_scores) if all_target_scores else 0.0

    print("\n" + "=" * 80)
    print("BEFORE / AFTER COMPOSITE QUALITY COMPARISON (vs Step A Audited Baseline: 40.0%)")
    print("=" * 80)
    print(f"Target Established: 70.0% (3.50 / 5.00)")
    print(f"\nV3_CD_01 (MDO Augmented Lagrangian):")
    print(f"  Step 6 Baseline:       {STEP6_BASELINES['V3_CD_01']:.3f} / 5.00 ({STEP6_BASELINES['V3_CD_01']/5*100:.1f}%)")
    print(f"  Step A Audited:        {STEPA_AUDITED_BASELINES['V3_CD_01']:.3f} / 5.00 ({STEPA_AUDITED_BASELINES['V3_CD_01']/5*100:.1f}%)")
    print(f"  Step B Multi-Turn:     {mean_01:.3f} / 5.00 ({mean_01/5*100:.1f}%)")
    print(f"  Delta vs Step A:       {mean_01 - STEPA_AUDITED_BASELINES['V3_CD_01']:+.3f} points ({(mean_01 - STEPA_AUDITED_BASELINES['V3_CD_01'])/5*100:+.1f}%)")

    print(f"\nV3_CD_41 (2D Diffusion PDE & Parquet):")
    print(f"  Step 6 Baseline:       {STEP6_BASELINES['V3_CD_41']:.3f} / 5.00 ({STEP6_BASELINES['V3_CD_41']/5*100:.1f}%)")
    print(f"  Step A Audited:        {STEPA_AUDITED_BASELINES['V3_CD_41']:.3f} / 5.00 ({STEPA_AUDITED_BASELINES['V3_CD_41']/5*100:.1f}%)")
    print(f"  Step B Multi-Turn:     {mean_41:.3f} / 5.00 ({mean_41/5*100:.1f}%)")
    print(f"  Delta vs Step A:       {mean_41 - STEPA_AUDITED_BASELINES['V3_CD_41']:+.3f} points ({(mean_41 - STEPA_AUDITED_BASELINES['V3_CD_41'])/5*100:+.1f}%)")

    print(f"\nTarget Subtask Pair Combined (V3_CD_01 + V3_CD_41):")
    print(f"  Step 6 Baseline:       {STEP6_BASELINES['target_pair']:.3f} / 5.00 ({STEP6_BASELINES['target_pair']/5*100:.1f}%)")
    print(f"  Step A Audited:        {STEPA_AUDITED_BASELINES['target_pair']:.3f} / 5.00 ({STEPA_AUDITED_BASELINES['target_pair']/5*100:.1f}%)")
    print(f"  Step B Multi-Turn:     {mean_target_pair:.3f} / 5.00 ({mean_target_pair/5*100:.1f}%)")
    print(f"  Delta vs Step A:       {mean_target_pair - STEPA_AUDITED_BASELINES['target_pair']:+.3f} points ({(mean_target_pair - STEPA_AUDITED_BASELINES['target_pair'])/5*100:+.1f}%)")
    print(f"  Target Gap:            {3.500 - mean_target_pair:.3f} points remaining to reach 70.0% (3.50/5.00)")

    # Save summary JSON
    summary_path = os.path.join(STEP_B_DIR, "step_b_eval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.time(),
            "target_metric": "mean_composite_quality_score",
            "target_threshold": 3.50,
            "target_percentage": 70.0,
            "step6_baselines": STEP6_BASELINES,
            "step_a_audited_baselines": STEPA_AUDITED_BASELINES,
            "step_b_scores": {
                "V3_CD_01": mean_01,
                "V3_CD_41": mean_41,
                "target_pair": mean_target_pair
            },
            "deltas_vs_step_a": {
                "V3_CD_01": mean_01 - STEPA_AUDITED_BASELINES["V3_CD_01"],
                "V3_CD_41": mean_41 - STEPA_AUDITED_BASELINES["V3_CD_41"],
                "target_pair": mean_target_pair - STEPA_AUDITED_BASELINES["target_pair"]
            },
            "trials_count": len(all_trials),
            "scales": [
                {
                    "label": label,
                    "baseline_id": res["baseline_id"],
                    "win_rate": res["win_rate"],
                    "slm_wins": res["slm_wins"],
                    "baseline_wins": res["baseline_wins"],
                    "ties": res["ties"],
                    "total_trials": res["total_trials"]
                }
                for label, res in audit_results
            ]
        }, f, indent=2)
    print(f"\nSaved Step B summary to {summary_path}")

if __name__ == "__main__":
    asyncio.run(main())

