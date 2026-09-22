import os
import json
import glob
from collections import defaultdict

# 1. Load response lengths
slm_lengths = {}
with open("results/v3_pilot/slm_pipeline_responses.jsonl", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            d = json.loads(line)
            slm_lengths[d["query_id"]] = len(d.get("response_text", ""))

base_lengths = {}
with open("results/v3_pilot/llm_baseline_responses.jsonl", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            d = json.loads(line)
            base_lengths[d["query_id"]] = len(d.get("response_text", ""))

query_info = {}
with open("results/v3_pilot/comparison.jsonl", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            d = json.loads(line)
            query_info[d["query_id"]] = d

# 2. Load all 32 trials
key_files = glob.glob("logs/v3_judge_keys/key_*.json")
trials = []
for kf in key_files:
    with open(kf, encoding="utf-8") as f:
        kdata = json.load(f)
    
    # Load corresponding public judge file
    pub_file = kdata.get("public_log_file")
    if pub_file and os.path.exists(pub_file):
        with open(pub_file, encoding="utf-8") as pf:
            pdata = json.load(pf)
    else:
        # try relative path
        base_name = os.path.basename(pub_file) if pub_file else ""
        rel_path = os.path.join("logs/v3_judge_pairwise", base_name)
        if os.path.exists(rel_path):
            with open(rel_path, encoding="utf-8") as pf:
                pdata = json.load(pf)
        else:
            pdata = {}

    trials.append({
        "query_id": kdata["query_id"],
        "order": kdata["order_tag"],
        "winner": kdata["unblinded_winner"],
        "cand_a_sys": kdata["candidate_a_system"],
        "cand_b_sys": kdata["candidate_b_system"],
        "primary_differentiator": pdata.get("primary_differentiator", "unknown"),
        "reasoning": pdata.get("reasoning", ""),
        "scores": pdata.get("criteria_scores", {}),
        "pub_file": pub_file
    })

# Sort by query_id and order
trials.sort(key=lambda x: (x["query_id"], x["order"]))

print("=" * 100)
print(f"{'Query ID':16s} | {'Tier':14s} | {'Order':8s} | {'Winner':16s} | {'SLM Chars':10s} | {'Base Chars':10s} | {'Differentiator':15s}")
print("=" * 100)

tier_map = {
    "V3_SD": "single_domain",
    "V3_TD": "two_domain",
    "V3_CD": "compound_dag"
}

tier_counts = defaultdict(lambda: {"total": 0, "slm_wins": 0, "base_wins": 0})

for t in trials:
    qid = t["query_id"]
    q_prefix = qid[:5]
    tier = tier_map.get(q_prefix, "unknown")
    
    tier_counts[tier]["total"] += 1
    if t["winner"] == "slm_pipeline_v3":
        tier_counts[tier]["slm_wins"] += 1
    elif t["winner"] == "gpt_120b":
        tier_counts[tier]["base_wins"] += 1

    s_len = slm_lengths.get(qid, 0)
    b_len = base_lengths.get(qid, 0)
    
    print(f"{qid:16s} | {tier:14s} | {t['order']:8s} | {t['winner']:16s} | {s_len:10d} | {b_len:10d} | {t['primary_differentiator']:15s}")

print("\n" + "=" * 100)
print("TIER BREAKDOWN SUMMARY:")
print("=" * 100)
for tier, c in tier_counts.items():
    wr = (c["slm_wins"] / c["total"] * 100.0) if c["total"] else 0.0
    print(f"Tier: {tier:16s} | Trials: {c['total']:2d} | SLM Wins: {c['slm_wins']:2d} | 120B Wins: {c['base_wins']:2d} | Win Rate: {wr:5.1f}%")

print("\n" + "=" * 100)
print("VERBATIM JUDGE REASONING FOR COMPOUND (TD & CD) LOSSES:")
print("=" * 100)

compound_losses = [t for t in trials if ("V3_TD" in t["query_id"] or "V3_CD" in t["query_id"]) and t["winner"] == "gpt_120b"]
for i, t in enumerate(compound_losses[:8]):
    qid = t["query_id"]
    order = t["order"]
    s_len = slm_lengths.get(qid, 0)
    b_len = base_lengths.get(qid, 0)
    scores = t["scores"]
    print(f"\n--- [Loss {i+1}] {qid} ({order}) | SLM Chars: {s_len} vs 120B Chars: {b_len} ---")
    print(f"Candidate A: {t['cand_a_sys']} | Candidate B: {t['cand_b_sys']}")
    print(f"Scores: {scores}")
    print(f"Primary Differentiator: {t['primary_differentiator']}")
    print(f"Judge Reasoning:\n{t['reasoning']}")

