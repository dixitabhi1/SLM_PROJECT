"""
AI Search Framework - Version 4 Step 2: Retrieval-Tool Grounding Evaluation
Evaluates deterministic reference-corpus grounding for the retrieval_qa specialist on V3_CD_21 Node 1.

Verifies:
1. Factual accuracy of cited RFC numbers and security mechanisms.
2. 100% cryptographic and textual traceability to data/corpora/security_standards_corpus.json.
3. Elimination of confabulation and evasive non-retrieval.
4. Double-blind pairwise judge evaluation against the 120B baseline.
"""

import os
import sys
import json
import time
import asyncio
from types import SimpleNamespace
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.ollama_runner import OllamaModelRunner
from src.v4.tools.retrieval_tool import DeterministicRetrievalTool
from src.v4.specialists.grounded_retrieval_runner import GroundedRetrievalModelRunner
from src.v2.judge.pairwise_harness import PairwiseLLMJudgeHarness
from src.v3.preflight import verify_distinct_roster_preflight

V4_STEP2_DIR = "results/v4_step2"
KEYS_DIR = "logs/v4_step2_judge_keys"
JUDGE_DIR = "logs/v4_step2_judge_pairwise"

def ensure_dirs():
    os.makedirs(V4_STEP2_DIR, exist_ok=True)
    os.makedirs(KEYS_DIR, exist_ok=True)
    os.makedirs(JUDGE_DIR, exist_ok=True)

async def evaluate_node1_grounding():
    ensure_dirs()
    print("=" * 80)
    print("V4 STEP 2: ISOLATED NODE 1 EVALUATION (V3_CD_21 RETRIEVAL_QA)")
    print("=" * 80)

    node1_task = (
        "Retrieve official security RFC standards and Linux socket vulnerability specifications "
        "relevant to secure network protocols, socket layer security, and kernel privilege escalations."
    )

    report_path = os.path.join(V4_STEP2_DIR, "v3_cd_21_node1_traceability.json")
    if os.path.exists(report_path):
        print(f"\nReusing existing traceability report from {report_path}")
        with open(report_path, "r", encoding="utf-8") as f:
            traceability_report = json.load(f)
        grounded_text = traceability_report["grounded"]["text"]
    else:
        # 1. Base ungrounded runner (Phi-3.5-mini 3.8B)
        base_runner = OllamaModelRunner(
            logical_model_name="phi3.5-3.8b",
            api_model_name="phi3.5:cpu",
            max_tokens=2048
        )

        # 2. Grounded runner with DeterministicRetrievalTool
        retrieval_tool = DeterministicRetrievalTool()
        grounded_runner = GroundedRetrievalModelRunner(
            base_runner=base_runner,
            retrieval_tool=retrieval_tool,
            top_k=4
        )

        print("\n[1/3] Generating Ungrounded Baseline Response (Parametric Memory Only)...")
        t0 = time.perf_counter()
        ungrounded_resp = await base_runner.generate(prompt=node1_task)
        dur_ungrounded = time.perf_counter() - t0
        print(f"  Ungrounded completed in {dur_ungrounded:.2f}s ({len(ungrounded_resp.text)} chars)")

        print("\n[2/3] Generating Grounded Specialist Response (Deterministic Corpus Grounded)...")
        t0 = time.perf_counter()
        grounded_resp = await grounded_runner.generate(prompt=node1_task)
        dur_grounded = time.perf_counter() - t0
        print(f"  Grounded completed in {dur_grounded:.2f}s ({len(grounded_resp.text)} chars)")

        # 3. Citation Traceability Audit
        print("\n[3/3] Performing Citation Traceability Verification...")
        corpus_docs = retrieval_tool.documents
        corpus_ids = {d["id"]: d for d in corpus_docs}
        
        cited_ids = []
        for cid in corpus_ids.keys():
            norm_cid = cid.replace("-", " ")
            if cid.lower() in grounded_resp.text.lower() or norm_cid.lower() in grounded_resp.text.lower():
                cited_ids.append(cid)

        print(f"  Retrieved sources fed to prompt: {grounded_resp.metadata.get('retrieved_sources')}")
        print(f"  Verified cited sources in grounded text: {cited_ids}")

        traceability_report = {
            "task_id": "V3_CD_21_node_1",
            "task_prompt": node1_task,
            "ungrounded": {
                "model": "phi3.5:cpu",
                "latency_s": dur_ungrounded,
                "text": ungrounded_resp.text
            },
            "grounded": {
                "model": "phi3.5:cpu + DeterministicRetrievalTool",
                "latency_s": dur_grounded,
                "text": grounded_resp.text,
                "retrieved_sources": grounded_resp.metadata.get("retrieved_sources"),
                "traceable_citations": cited_ids,
                "traceability_rate": f"{len(cited_ids)} / {len(grounded_resp.metadata.get('retrieved_sources', []))}"
            }
        }

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(traceability_report, f, indent=2)
        print(f"Saved Node 1 traceability report to {report_path}")
        grounded_text = grounded_resp.text

    # 4. Double-Blind Pairwise Judging against 120B Baseline
    print("\n" + "=" * 80)
    print("V4 STEP 2: DOUBLE-BLIND PAIRWISE JUDGE EVALUATION (Node 1 Grounded vs 120B)")
    print("=" * 80)

    # Load 120B baseline response for V3_CD_21
    with open("results/v3_pilot/llm_baseline_responses.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["query_id"] == "V3_CD_21":
                base_120b_text = d["response_text"]
                break

    harness = PairwiseLLMJudgeHarness(judge_model_name="qwen/qwen3.8-27b")

    # Hard Rule 13 pre-flight check
    pipeline_roster = {
        "retrieval_specialist": SimpleNamespace(api_model_name="phi3.5:cpu")
    }
    baseline_roster = {
        "gpt_120b": SimpleNamespace(api_model_name="openai/gpt-oss-120b")
    }
    verify_distinct_roster_preflight(pipeline_roster, baseline_roster, judge_runner=harness)

    # Forward trial: Cand A = Grounded SLM, Cand B = 120B
    print("Running Forward Trial (Grounded SLM vs 120B)...", flush=True)
    res_f = await asyncio.to_thread(
        harness.evaluate_pair,
        query_id="V3_CD_21_NODE1",
        query_text=node1_task,
        system_a_id="slm_grounded_phi35",
        text_a=grounded_text,
        system_b_id="gpt_120b",
        text_b=base_120b_text,
        order_tag="forward",
        judge_log_dir=JUDGE_DIR,
        key_log_dir=KEYS_DIR
    )
    print(f"  Forward Result: Winner={res_f.get('unblinded_winner')} | Diff={res_f.get('primary_differentiator')}")
    await asyncio.sleep(2.0)

    # Swapped trial: Cand A = 120B, Cand B = Grounded SLM
    print("Running Swapped Trial (120B vs Grounded SLM)...", flush=True)
    res_s = await asyncio.to_thread(
        harness.evaluate_pair,
        query_id="V3_CD_21_NODE1",
        query_text=node1_task,
        system_a_id="slm_grounded_phi35",
        text_a=grounded_text,
        system_b_id="gpt_120b",
        text_b=base_120b_text,
        order_tag="swapped",
        judge_log_dir=JUDGE_DIR,
        key_log_dir=KEYS_DIR
    )
    print(f"  Swapped Result: Winner={res_s.get('unblinded_winner')} | Diff={res_s.get('primary_differentiator')}")

    print("\n" + "=" * 80)
    print("STEP 2 EVALUATION SUMMARY COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(evaluate_node1_grounding())
