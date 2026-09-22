"""
Build Stratified 100-Query Benchmark Dataset
Strictly preserves Hard Rule 5: Zero overlap with data/v3_queries_held_out.json.
Strata breakdown:
- 34 Single-Domain
- 33 Two-Domain
- 33 Compound DAGs (three_plus_domain)
Total: 100 Queries.
"""

import os
import json
import hashlib

V3_DEV_PATH = "data/v3_queries_dev.json"
V1_DEV_PATH = "data/queries_dev.json"
HELD_OUT_PATH = "data/v3_queries_held_out.json"
OUT_BENCHMARK_PATH = "data/eval_100_benchmark.json"
OUT_SHA_PATH = "data/eval_100_benchmark.sha256"

def main():
    with open(V3_DEV_PATH, "r", encoding="utf-8") as f:
        v3_dev = json.load(f)
    with open(V1_DEV_PATH, "r", encoding="utf-8") as f:
        v1_dev = json.load(f)
    with open(HELD_OUT_PATH, "r", encoding="utf-8") as f:
        held_out = json.load(f)

    held_out_ids = set(q["id"] for q in held_out)
    held_out_texts = set(q["query"].strip() for q in held_out)

    # Validate zero contamination
    for q in v3_dev:
        assert q["id"] not in held_out_ids, f"Contamination in v3_dev: {q['id']}"
        assert q["query"].strip() not in held_out_texts, f"Query text in held-out: {q['id']}"
    for q in v1_dev:
        assert q["id"] not in held_out_ids, f"Contamination in v1_dev: {q['id']}"

    # Partition v3_dev
    v3_sd = [q for q in v3_dev if q["complexity_tier"] == "single_domain"]
    v3_td = [q for q in v3_dev if q["complexity_tier"] == "two_domain"]
    v3_cd = [q for q in v3_dev if q["complexity_tier"] == "three_plus_domain"]

    # Partition v1_dev
    v1_sd = [q for q in v1_dev if q["complexity_tier"] == "single_domain"]
    v1_td = [q for q in v1_dev if q["complexity_tier"] == "two_domain"]
    v1_cd = [q for q in v1_dev if q["complexity_tier"] == "three_plus_domain"]

    # Select queries: 34 SD, 33 TD, 33 CD
    selected_sd = v3_sd + v1_sd[: (34 - len(v3_sd))]
    selected_td = v3_td + v1_td[: (33 - len(v3_td))]
    selected_cd = v3_cd + v1_cd[: (33 - len(v3_cd))]

    final_100 = selected_sd + selected_td + selected_cd
    assert len(final_100) == 100, f"Expected 100 queries, got {len(final_100)}"
    assert len(selected_sd) == 34, f"Expected 34 SD, got {len(selected_sd)}"
    assert len(selected_td) == 33, f"Expected 33 TD, got {len(selected_td)}"
    assert len(selected_cd) == 33, f"Expected 33 CD, got {len(selected_cd)}"

    # Check for duplicate IDs
    ids = [q["id"] for q in final_100]
    assert len(ids) == len(set(ids)), f"Duplicate query IDs found: {len(ids)} vs {len(set(ids))}"

    with open(OUT_BENCHMARK_PATH, "w", encoding="utf-8") as f:
        json.dump(final_100, f, indent=2)

    with open(OUT_BENCHMARK_PATH, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()

    with open(OUT_SHA_PATH, "w", encoding="utf-8") as f:
        f.write(sha + "\n")

    print(f"Successfully constructed {OUT_BENCHMARK_PATH} with 100 queries.")
    print(f"  Single-Domain: {len(selected_sd)}")
    print(f"  Two-Domain:    {len(selected_td)}")
    print(f"  Compound DAG:  {len(selected_cd)}")
    print(f"  SHA256:        {sha}")

if __name__ == "__main__":
    main()
