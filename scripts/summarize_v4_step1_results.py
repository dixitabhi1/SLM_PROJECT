"""
Summarize results for Version 4 Step 1: True <=5B Model Capacity Evaluation
Analyzes all trials in logs/v4_step1_judge_keys/ and logs/v4_step1_judge_pairwise/
"""

import os
import glob
import json
from collections import defaultdict

def summarize_step1():
    key_files = glob.glob("logs/v4_step1_judge_keys/key_*.json")
    if not key_files:
        print("No v4 Step 1 judge keys found.")
        return

    # Load lengths
    slm_lengths = {}
    if os.path.exists("results/v4_step1/slm_phi35_responses.jsonl"):
        with open("results/v4_step1/slm_phi35_responses.jsonl", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    slm_lengths[d["query_id"]] = len(d.get("response_text", ""))

    base_lengths = {}
    if os.path.exists("results/v3_pilot/llm_baseline_responses.jsonl"):
        with open("results/v3_pilot/llm_baseline_responses.jsonl", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    base_lengths[d["query_id"]] = len(d.get("response_text", ""))

    trials = []
    for kf in key_files:
        with open(kf, encoding="utf-8") as f:
            kdata = json.load(f)
        pub_file = kdata.get("public_log_file")
        pdata = {}
        if pub_file and os.path.exists(pub_file):
            with open(pub_file, encoding="utf-8") as pf:
                pdata = json.load(pf)
        else:
            base_name = os.path.basename(pub_file) if pub_file else ""
            rel_path = os.path.join("logs/v4_step1_judge_pairwise", base_name)
            if os.path.exists(rel_path):
                with open(rel_path, encoding="utf-8") as pf:
                    pdata = json.load(pf)

        trials.append({
            "query_id": kdata["query_id"],
            "order": kdata["order_tag"],
            "winner": kdata["unblinded_winner"],
            "differentiator": pdata.get("primary_differentiator", "unknown"),
            "reasoning": pdata.get("reasoning", ""),
            "scores": pdata.get("criteria_scores", {})
        })

    trials.sort(key=lambda x: (x["query_id"], x["order"]))

    print("=" * 105)
    print(f"{'Query ID':16s} | {'Tier':14s} | {'Order':8s} | {'Winner':18s} | {'Phi Chars':10s} | {'120B Chars':10s} | {'Differentiator':15s}")
    print("=" * 105)

    tier_map = {"V3_SD": "single_domain", "V3_TD": "two_domain", "V3_CD": "compound_dag"}
    tier_counts = defaultdict(lambda: {"total": 0, "slm_wins": 0, "base_wins": 0})

    for t in trials:
        qid = t["query_id"]
        tier = tier_map.get(qid[:5], "unknown")
        tier_counts[tier]["total"] += 1
        if t["winner"] == "slm_phi35_v4":
            tier_counts[tier]["slm_wins"] += 1
        elif t["winner"] == "gpt_120b":
            tier_counts[tier]["base_wins"] += 1

        s_len = slm_lengths.get(qid, 0)
        b_len = base_lengths.get(qid, 0)
        print(f"{qid:16s} | {tier:14s} | {t['order']:8s} | {t['winner']:18s} | {s_len:10d} | {b_len:10d} | {t['differentiator']:15s}")

    print("\n" + "=" * 105)
    print("TIER BREAKDOWN SUMMARY (v4 Step 1: Phi-3.5-mini 3.8B vs 120B Baseline):")
    print("=" * 105)
    tot_trials = sum(c["total"] for c in tier_counts.values())
    tot_slm = sum(c["slm_wins"] for c in tier_counts.values())
    tot_base = sum(c["base_wins"] for c in tier_counts.values())
    for tier, c in tier_counts.items():
        wr = (c["slm_wins"] / c["total"] * 100.0) if c["total"] else 0.0
        print(f"Tier: {tier:16s} | Trials: {c['total']:2d} | SLM Wins: {c['slm_wins']:2d} | 120B Wins: {c['base_wins']:2d} | Win Rate: {wr:5.1f}%")

    tot_wr = (tot_slm / tot_trials * 100.0) if tot_trials else 0.0
    print("-" * 105)
    print(f"OVERALL BENCHMARK: Trials: {tot_trials:2d} | SLM Wins: {tot_slm:2d} | 120B Wins: {tot_base:2d} | Win Rate: {tot_wr:5.1f}%")

    print("\n" + "=" * 105)
    print("VERBATIM JUDGE REASONING FOR COMPOUND QUERIES:")
    print("=" * 105)
    cd_trials = [t for t in trials if "V3_CD" in t["query_id"]]
    for i, t in enumerate(cd_trials):
        qid = t["query_id"]
        order = t["order"]
        s_len = slm_lengths.get(qid, 0)
        b_len = base_lengths.get(qid, 0)
        print(f"\n--- [{i+1}] {qid} ({order}) | Winner: {t['winner']} | Diff: {t['differentiator']} | Phi Chars: {s_len} vs 120B Chars: {b_len} ---")
        print(f"Reasoning:\n{t['reasoning']}")

if __name__ == "__main__":
    summarize_step1()

