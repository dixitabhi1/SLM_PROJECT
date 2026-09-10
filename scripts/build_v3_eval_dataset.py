"""
v3 Evaluation Dataset Builder (8 Domains, <=5B Model Pool, >=30B Baselines)
AI Search Framework (Version 3 Architecture)

Generates:
1. data/v3_eval_dataset_master.json (240 stratified queries across 8 domains and 3 tiers)
2. data/v3_queries_dev.json (80 queries for dev & tuning)
3. data/v3_queries_held_out.json (160 queries locked for held-out evaluation)
4. data/v3_gold_dags.json (100 gold task graphs)
5. data/v3_held_out_lock.sha256 & data/v3_held_out_lock.json (Cryptographic lock of v3 held-out split)
"""

import json
import hashlib
import os
from datetime import datetime, timezone

os.makedirs("data", exist_ok=True)

DOMAINS_V3 = [
    "coding",
    "mathematics",
    "formal_reasoning",
    "retrieval_qa",
    "science_tech",
    "structured_data",
    "creative_synthesis",
    "systems_ops"
]

DOMAIN_PROMPTS = {
    "coding": "Implement an advanced algorithmic module #{i} in Python with full type annotations, O(1)/O(log N) complexity guarantees, and comprehensive edge-case unit tests.",
    "mathematics": "Derive the closed-form analytical solution and prove convergence properties for mathematical formulation #{i}, showing all intermediate algebraic steps.",
    "formal_reasoning": "Perform formal deductive verification of logical problem #{i}, state validity conditions, identify fallacies, and construct symbolic proofs.",
    "retrieval_qa": "Retrieve and synthesize factual specifications, RFC standard protocols, and regulatory framework requirements for technical domain #{i}.",
    "science_tech": "Analyze physical and thermodynamic principles, semiconductor bandgap states, and quantum optical transitions for technical system #{i}.",
    "structured_data": "Construct an optimized relational database schema #{i} with SQL queries, indexes, foreign key constraints, and JSON transformation pipelines.",
    "creative_synthesis": "Synthesize a comprehensive technical report and executive overview for multi-disciplinary architecture #{i}, ensuring stylistic harmonization.",
    "systems_ops": "Configure a secure POSIX shell environment, Docker containerization pipeline, and Linux network socket management script for infrastructure system #{i}."
}

queries = []

# --- Tier 1: Single-Domain Control (n = 80 queries, 10 per domain) ---
for dom in DOMAINS_V3:
    for i in range(1, 11):
        qid = f"V3_SD_{dom[:4].upper()}_{i:02d}"
        text = DOMAIN_PROMPTS[dom].format(i=i)
        queries.append({
            "id": qid,
            "complexity_tier": "single_domain",
            "domains": [dom],
            "triggers_feedback_loop": False,
            "query": text,
            "gold_dag_available": True if i <= 5 else False
        })

# --- Tier 2: 2-Domain Compound (n = 80 queries) ---
compound_2d_pairs = [
    ("coding", "mathematics", "Derive quantitative formulation and implement vectorized Python simulation"),
    ("coding", "structured_data", "Design relational database schema and implement Python ORM access layer"),
    ("formal_reasoning", "coding", "Analyze theoretical state invariants and implement verified concurrency engine"),
    ("retrieval_qa", "formal_reasoning", "Retrieve regulatory RFC standards and construct formal deductive compliance proof"),
    ("science_tech", "mathematics", "Formulate quantum Hamiltonian equations and compute eigenvalue matrix solutions"),
    ("systems_ops", "coding", "Construct Linux network socket handler and implement asynchronous Python event loop"),
    ("structured_data", "retrieval_qa", "Extract technical specifications and construct relational SQL query schema"),
    ("creative_synthesis", "science_tech", "Analyze thermodynamic efficiency models and author comprehensive engineering report")
]

for idx, (d1, d2, desc) in enumerate(compound_2d_pairs):
    for i in range(1, 11):
        pair_num = idx * 10 + i
        qid = f"V3_TD_{pair_num:02d}"
        is_loop = (i % 2 == 1) # Alternating loop-triggering
        text = f"{desc} for compound engineering problem #{pair_num}, thoroughly addressing both domain constraints."
        queries.append({
            "id": qid,
            "complexity_tier": "two_domain",
            "domains": [d1, d2],
            "triggers_feedback_loop": is_loop,
            "query": text,
            "gold_dag_available": True if i <= 5 else False
        })

# --- Tier 3: 3+-Domain Compound (n = 80 queries) ---
compound_3d_triplets = [
    (["coding", "mathematics", "formal_reasoning"], "Derive mathematical optimization loss, prove theoretical convergence, and implement an end-to-end Python benchmark"),
    (["retrieval_qa", "coding", "systems_ops"], "Retrieve security RFC standards, evaluate Linux socket vulnerabilities, and implement a sandboxed Python runtime"),
    (["science_tech", "mathematics", "structured_data"], "Simulate physical thermodynamic diffusion, solve partial differential matrices, and export relational parquet datasets"),
    (["formal_reasoning", "structured_data", "creative_synthesis"], "Formulate formal relational integrity invariants, design normalized SQL schemas, and synthesize executive architectural overview")
]

for idx, (doms, desc) in enumerate(compound_3d_triplets):
    for i in range(1, 21):
        trip_num = idx * 20 + i
        qid = f"V3_CD_{trip_num:02d}"
        is_loop = (i <= 14)
        text = f"{desc} for multi-disciplinary challenge #{trip_num}, fulfilling all domain requirements."
        queries.append({
            "id": qid,
            "complexity_tier": "three_plus_domain",
            "domains": doms,
            "triggers_feedback_loop": is_loop,
            "query": text,
            "gold_dag_available": True if i <= 10 else False
        })

# --- Partition into Dev (n = 80) and Held-Out (n = 160) Splits ---
# Stratified round-robin 1:2 split across all 8 domains and compound groups
dev_queries = []
held_out_queries = []

for q in queries:
    qid = q["id"]
    t = q["complexity_tier"]
    num = int(qid.split("_")[-1])
    
    if t == "single_domain":
        # First 3 (and 4th for first 2 domains) go to dev
        dom = q["domains"][0]
        dom_idx = DOMAINS_V3.index(dom)
        is_dev = (num <= 3) or (num == 4 and dom_idx < 2)
    elif t == "two_domain":
        pair_sub = (num - 1) % 10 + 1
        pair_idx = (num - 1) // 10
        is_dev = (pair_sub <= 3) or (pair_sub == 4 and pair_idx < 3)
    else:
        trip_sub = (num - 1) % 20 + 1
        trip_idx = (num - 1) // 20
        is_dev = (trip_sub <= 6) or (trip_sub == 7 and trip_idx < 3)

    if is_dev:
        q["split"] = "dev"
        dev_queries.append(q)
    else:
        q["split"] = "held_out"
        held_out_queries.append(q)

# --- Gold DAG Annotations (100 queries) ---
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

# --- Write Files and Compute v3 SHA-256 Lock ---
with open("data/v3_eval_dataset_master.json", "w", encoding="utf-8") as f:
    json.dump(queries, f, indent=2)

with open("data/v3_queries_dev.json", "w", encoding="utf-8") as f:
    json.dump(dev_queries, f, indent=2)

held_out_bytes = json.dumps(held_out_queries, indent=2, sort_keys=True).encode("utf-8")
with open("data/v3_queries_held_out.json", "wb") as f:
    f.write(held_out_bytes)

with open("data/v3_gold_dags.json", "w", encoding="utf-8") as f:
    json.dump(gold_dags, f, indent=2)

v3_held_out_sha256 = hashlib.sha256(held_out_bytes).hexdigest()
lock_meta = {
    "version": "3.0.0",
    "held_out_file": "data/v3_queries_held_out.json",
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
    "sha256": v3_held_out_sha256,
    "locked_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": "LOCKED — Subject to AGENTS.md Rule 5 and Rule 12 Held-out discipline"
}

with open("data/v3_held_out_lock.json", "w", encoding="utf-8") as f:
    json.dump(lock_meta, f, indent=2)

with open("data/v3_held_out_lock.sha256", "w", encoding="utf-8") as f:
    f.write(f"{v3_held_out_sha256}  data/v3_queries_held_out.json\n")

print("=== v3 DATASET CONSTRUCTION COMPLETE ===")
print(f"Total Queries: {len(queries)} (Dev: {len(dev_queries)}, Held-Out: {len(held_out_queries)})")
print(f"Gold DAGs: {len(gold_dags)}")
print(f"v3 Held-Out SHA-256 Lock: {v3_held_out_sha256}")

