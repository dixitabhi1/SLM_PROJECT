"""
Test Calibrated DecomposerSLM_v3 in Isolation on Mixed Set:
- 4 Compound Queries (V3_CD_01, V3_CD_21, V3_CD_41, V3_CD_61) -> Expected: >= 2 subtask nodes
- 3 Single-Domain Queries (V3_SD_CODI_01, V3_SD_MATH_01, V3_SD_FORM_01) -> Expected: exactly 1 subtask node
"""

import os
import sys
import json
import asyncio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.ollama_runner import OllamaModelRunner
from src.v3.decomposer.decomposer import DecomposerSLM_v3

TEST_QUERIES = [
    # Compound Queries (Must decompose into >= 2 nodes)
    {"id": "V3_CD_01", "type": "compound", "expected_min": 2},
    {"id": "V3_CD_21", "type": "compound", "expected_min": 2},
    {"id": "V3_CD_41", "type": "compound", "expected_min": 2},
    {"id": "V3_CD_61", "type": "compound", "expected_min": 2},
    # Single-Domain Queries (Must collapse into exactly 1 node)
    {"id": "V3_SD_CODI_01", "type": "single_domain", "expected_min": 1, "expected_max": 1},
    {"id": "V3_SD_MATH_01", "type": "single_domain", "expected_min": 1, "expected_max": 1},
    {"id": "V3_SD_FORM_01", "type": "single_domain", "expected_min": 1, "expected_max": 1},
]

async def main():
    with open("data/v3_queries_dev.json", "r", encoding="utf-8") as f:
        dev_queries = {q["id"]: q for q in json.load(f)}

    runner = OllamaModelRunner(
        logical_model_name="llama3.2-3b",
        api_model_name="llama3.2:cpu",
        max_tokens=1024
    )
    decomposer = DecomposerSLM_v3(runner)

    print("=" * 70)
    print("TESTING CALIBRATED DECOMPOSER (llama3.2:3b) IN ISOLATION")
    print("=" * 70)

    results = []
    all_passed = True

    for t in TEST_QUERIES:
        qid = t["id"]
        qdata = dev_queries[qid]
        qtext = qdata["query"]
        qtype = t["type"]

        print(f"\n--- Testing {qid} [{qtype.upper()}] ---")
        print(f"Query: {qtext[:100]}...")

        res = await decomposer.decompose_initial(qtext)
        subtasks = res["subtasks"]
        is_valid = res["is_schema_valid"]
        count = len(subtasks)

        caps = [s.get("capability", "unknown") for s in subtasks]
        print(f"Schema Valid: {is_valid}")
        print(f"Subtask Count: {count}")
        for idx, s in enumerate(subtasks, 1):
            print(f"  Node {idx} ({s.get('id')}): [{s.get('capability')}] {s.get('text')[:80]}... (deps: {s.get('dependencies')})")

        passed = True
        if qtype == "compound":
            if count < t["expected_min"]:
                passed = False
                print(f"  FAILED: Expected >= {t['expected_min']} nodes for compound query, got {count}!")
            else:
                print(f"  PASSED: Successfully decomposed compound query into {count} nodes (capabilities: {caps}).")
        elif qtype == "single_domain":
            if count != 1:
                passed = False
                print(f"  FAILED: Expected exactly 1 node for single-domain query, got {count} (Over-fragmentation regression)!")
            else:
                print(f"  PASSED: Successfully kept single-domain query atomic (1 node: {caps[0]}).")

        if not passed:
            all_passed = False

        results.append({
            "query_id": qid,
            "type": qtype,
            "subtask_count": count,
            "capabilities": caps,
            "passed": passed,
            "raw_response": res.get("raw_response", "")
        })

    print("\n" + "=" * 70)
    print("ISOLATION TEST SUMMARY:")
    for r in results:
        status_str = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status_str}] {r['query_id']} ({r['type']}): {r['subtask_count']} subtasks -> {r['capabilities']}")
    print("=" * 70)

    if all_passed:
        print("ALL 7 TEST CASES PASSED! Decomposer is correctly calibrated.")
    else:
        print("SOME TEST CASES FAILED! Calibration needs refinement.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
