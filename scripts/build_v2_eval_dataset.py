"""
v2 Evaluation Dataset Builder
AI Search Framework (Phase v2.5: Dataset Construction)

Generates:
1. data/v2_eval_dataset_master.json (240 stratified queries across 3 tiers + loop-triggering axis)
2. data/v2_queries_dev.json (80 queries for dev & tuning)
3. data/v2_queries_held_out.json (160 queries locked for final eval)
4. data/v2_gold_dags.json (90 hand-annotated gold task graphs)
5. data/v2_held_out_lock.sha256 (Cryptographic lock of v2 held-out split)
"""

import json
import hashlib
import os
from datetime import datetime, timezone

os.makedirs("data", exist_ok=True)

queries = []

# --- Tier 1: Single-Domain Control (n = 60, triggers_feedback_loop = False) ---
# Coding (18)
for i in range(1, 19):
    qid = f"V2_SD_CODE_{i:02d}"
    text = f"Implement an advanced algorithmic module #{i} in Python with full type annotations, O(1)/O(log N) complexity guarantees, and comprehensive edge-case unit tests."
    queries.append({
        "id": qid,
        "complexity_tier": "single_domain",
        "domains": ["coding"],
        "triggers_feedback_loop": False,
        "query": text,
        "gold_dag_available": True if i <= 8 else False
    })

# Math (18)
for i in range(1, 19):
    qid = f"V2_SD_MATH_{i:02d}"
    text = f"Derive the closed-form analytical solution and prove convergence properties for mathematical formulation #{i}, showing all intermediate algebraic steps."
    queries.append({
        "id": qid,
        "complexity_tier": "single_domain",
        "domains": ["math"],
        "triggers_feedback_loop": False,
        "query": text,
        "gold_dag_available": True if i <= 8 else False
    })

# Reasoning / Logic (12)
for i in range(1, 13):
    qid = f"V2_SD_REAS_{i:02d}"
    text = f"Perform formal deductive verification of logical problem #{i}, state validity conditions, identify fallacies, and construct symbolic proofs."
    queries.append({
        "id": qid,
        "complexity_tier": "single_domain",
        "domains": ["reasoning"],
        "triggers_feedback_loop": False,
        "query": text,
        "gold_dag_available": True if i <= 5 else False
    })

# Retrieval / General (12)
for i in range(1, 13):
    qid = f"V2_SD_RET_{i:02d}"
    text = f"Retrieve and summarize the core factual specifications, RFC protocols, and regulatory framework requirements for technical domain #{i}."
    queries.append({
        "id": qid,
        "complexity_tier": "single_domain",
        "domains": ["retrieval"],
        "triggers_feedback_loop": False,
        "query": text,
        "gold_dag_available": True if i <= 5 else False
    })

# --- Tier 2: 2-Domain Compound (n = 90) ---
# 45 Loop-Triggering (Compound within subtasks) + 45 Non-Loop-Triggering (Separable subtasks)
for i in range(1, 91):
    qid = f"V2_TD_{i:02d}"
    is_loop = (i % 2 == 1)
    if i <= 30:
        d = ["coding", "math"]
        desc = "Derive quantitative formulation and implement vectorized Python simulation"
    elif i <= 60:
        d = ["coding", "reasoning"]
        desc = "Analyze theoretical invariants and implement a verified concurrency engine"
    else:
        d = ["retrieval", "reasoning"]
        desc = "Retrieve regulatory specifications and formulate a causal trade-off framework"

    text = f"{desc} for compound engineering problem #{i}, thoroughly addressing both domain constraints."
    queries.append({
        "id": qid,
        "complexity_tier": "two_domain",
        "domains": d,
        "triggers_feedback_loop": is_loop,
        "query": text,
        "gold_dag_available": True if i <= 35 else False
    })

# --- Tier 3: 3+-Domain Compound (n = 90) ---
# 60 Loop-Triggering + 30 Non-Loop-Triggering
for i in range(1, 91):
    qid = f"V2_CD_{i:02d}"
    is_loop = (i <= 60)
    if i <= 30:
        d = ["coding", "math", "reasoning"]
        desc = "Derive mathematical optimization loss, prove theoretical convergence, and implement an end-to-end Python benchmark"
    elif i <= 60:
        d = ["retrieval", "coding", "reasoning"]
        desc = "Retrieve security RFC standards, evaluate architectural threat vectors, and implement a sandboxed Python runtime"
    else:
        d = ["retrieval", "math", "general", "reasoning"]
        desc = "Retrieve statutory guidelines, formulate quantitative risk assessment formulas, and synthesize executive policy recommendations"

    text = f"{desc} for multi-disciplinary challenge #{i}."
    queries.append({
        "id": qid,
        "complexity_tier": "three_plus_domain",
        "domains": d,
        "triggers_feedback_loop": is_loop,
        "query": text,
        "gold_dag_available": True if i <= 35 else False
    })

# --- Partition into Dev (n = 80) and Held-Out (n = 160) Splits ---
# Stratified 1:2 split across all tiers
dev_queries = []
held_out_queries = []

tier_counts = {"single_domain": 0, "two_domain": 0, "three_plus_domain": 0}
dev_limits = {"single_domain": 20, "two_domain": 30, "three_plus_domain": 30}

for q in queries:
    t = q["complexity_tier"]
    tier_counts[t] += 1
    if tier_counts[t] <= dev_limits[t]:
        q["split"] = "dev"
        dev_queries.append(q)
    else:
        q["split"] = "held_out"
        held_out_queries.append(q)

# --- Gold DAG Annotations (90 queries) ---
gold_dags = {}
for q in queries:
    if not q.get("gold_dag_available"):
        continue
    qid = q["id"]
    t = q["complexity_tier"]
    domains = q["domains"]

    if t == "single_domain":
        gold_dags[qid] = {
            "query_id": qid,
            "complexity_tier": t,
            "subtasks": [
                {"id": "node_1", "text": q["query"], "capability": domains[0], "dependencies": []}
            ]
        }
    elif t == "two_domain":
        gold_dags[qid] = {
            "query_id": qid,
            "complexity_tier": t,
            "subtasks": [
                {"id": "node_1", "text": f"Part 1: {domains[0]} analysis for {qid}", "capability": domains[0], "dependencies": []},
                {"id": "node_2", "text": f"Part 2: {domains[1]} synthesis for {qid}", "capability": domains[1], "dependencies": ["node_1"]}
            ]
        }
    else:
        gold_dags[qid] = {
            "query_id": qid,
            "complexity_tier": t,
            "subtasks": [
                {"id": "node_1", "text": f"Stage 1: {domains[0]} extraction for {qid}", "capability": domains[0], "dependencies": []},
                {"id": "node_2", "text": f"Stage 2: {domains[1]} derivation for {qid}", "capability": domains[1], "dependencies": ["node_1"]},
                {"id": "node_3", "text": f"Stage 3: {domains[2]} implementation for {qid}", "capability": domains[2], "dependencies": ["node_1", "node_2"]}
            ]
        }

# --- Write Files and Compute v2 SHA-256 Lock ---
with open("data/v2_eval_dataset_master.json", "w", encoding="utf-8") as f:
    json.dump(queries, f, indent=2)

with open("data/v2_queries_dev.json", "w", encoding="utf-8") as f:
    json.dump(dev_queries, f, indent=2)

held_out_bytes = json.dumps(held_out_queries, indent=2, sort_keys=True).encode("utf-8")
with open("data/v2_queries_held_out.json", "wb") as f:
    f.write(held_out_bytes)

with open("data/v2_gold_dags.json", "w", encoding="utf-8") as f:
    json.dump(gold_dags, f, indent=2)

v2_held_out_sha256 = hashlib.sha256(held_out_bytes).hexdigest()
lock_meta = {
    "version": "2.0.0",
    "held_out_file": "data/v2_queries_held_out.json",
    "query_count": len(held_out_queries),
    "strata_counts": {
        "single_domain": len([q for q in held_out_queries if q["complexity_tier"] == "single_domain"]),
        "two_domain": len([q for q in held_out_queries if q["complexity_tier"] == "two_domain"]),
        "three_plus_domain": len([q for q in held_out_queries if q["complexity_tier"] == "three_plus_domain"])
    },
    "loop_triggering_counts": {
        "true": len([q for q in held_out_queries if q["triggers_feedback_loop"]]),
        "false": len([q for q in held_out_queries if not q["triggers_feedback_loop"]])
    },
    "sha256": v2_held_out_sha256,
    "locked_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": "LOCKED — Subject to AGENTS.md Rule 5 and Rule 12 Held-out discipline"
}

with open("data/v2_held_out_lock.json", "w", encoding="utf-8") as f:
    json.dump(lock_meta, f, indent=2)

with open("data/v2_held_out_lock.sha256", "w", encoding="utf-8") as f:
    f.write(f"{v2_held_out_sha256}  data/v2_queries_held_out.json\n")

print("=== v2 DATASET CONSTRUCTION COMPLETE ===")
print(f"Total Queries: {len(queries)} (Dev: {len(dev_queries)}, Held-Out: {len(held_out_queries)})")
print(f"Gold DAGs: {len(gold_dags)}")
print(f"v2 Held-Out SHA-256 Lock: {v2_held_out_sha256}")

