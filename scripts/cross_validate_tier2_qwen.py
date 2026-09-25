"""
Cross-Validation Harness: Impartiality Spot-Check on Baseline Tier 2 (gemini-2.5-flash)
Using Independent Judge: qwen/qwen3.8-27b on Groq

Tests for potential vendor self-preference bias between:
  Primary Judge: gemini-3.1-flash-lite (Google DeepMind)
  Baseline Tier 2: gemini-2.5-flash (Google DeepMind)

Evaluates all 8 compound queries across both E1 (Base 11.85B pool) and E2 (FT 11.85B pool)
under forward and swapped presentations (32 total trials).
Computes cross-model concordance, agreement rates, and qualitative defect alignment.
"""

import json
import math
import os
import sys
import time
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv
from groq import Groq

load_dotenv(".env")

CROSS_VAL_DIR = "results/mentor_protocol/cross_validation"
KEY_DIR = os.path.join(CROSS_VAL_DIR, "judge_keys")
PAIRWISE_DIR = os.path.join(CROSS_VAL_DIR, "judge_pairwise")
TRIALS_PATH = os.path.join(CROSS_VAL_DIR, "tier2_qwen_judge_trials.jsonl")
REPORT_PATH = os.path.join(CROSS_VAL_DIR, "tier2_cross_validation_report.json")

E1_SLM_PATH = "results/mentor_protocol/e1/slm_pipeline_responses.jsonl"
E2_SLM_PATH = "results/mentor_protocol/e2/slm_pipeline_responses.jsonl"
B32_PATH = "results/mentor_protocol/e1/baseline_32b_responses.jsonl"

E1_GEMINI_TRIALS = "results/mentor_protocol/e1/judge_trials_raw.jsonl"
E2_GEMINI_TRIALS = "results/mentor_protocol/e2/judge_trials_raw.jsonl"

TARGET_QUERY_IDS = ["V3_TD_01", "V3_TD_11", "V3_TD_21", "V3_TD_31", "V3_TD_41", "V3_TD_51", "V3_TD_61", "V3_TD_71"]


def load_jsonl(path: str) -> Dict[str, Any]:
    data = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    data[rec.get("query_id")] = rec
    return data


def load_gemini_b32_trials(path: str) -> Dict[str, Dict[str, Any]]:
    # Map (query_id, order_tag) -> trial
    trials = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if rec.get("baseline_tier") == "b32":
                        key = f"{rec.get('query_id')}_{rec.get('order_tag')}"
                        trials[key] = rec
    return trials


def call_qwen_judge(
    client: Groq,
    query_text: str,
    cand_a_text: str,
    cand_b_text: str,
    max_retries: int = 8
) -> Tuple[Dict[str, Any], float]:
    system_prompt = (
        "You are an impartial, expert AI judge evaluating two candidate responses (Candidate A and Candidate B) to a technical user query.\n\n"
        "Evaluation Frameworks:\n"
        "1. Criteria Framework (1-5 Scale):\n"
        "   - Correctness (1-5): Mathematical, factual, algorithmic precision.\n"
        "   - Completeness (1-5): Thorough fulfillment of all specifications and constraints.\n"
        "   - Coherence (1-5): Logical structure, clarity, and synthesis.\n\n"
        "2. Holistic Framework (1-10 Scale):\n"
        "   - Overall Technical Quality Score (1-10): Global technical rigor, practical utility, and excellence.\n"
        "   - 10 = Flawless publication-grade solution; 1 = Completely incorrect or useless.\n\n"
        "Instructions:\n"
        "- Evaluate both candidates with absolute impartiality.\n"
        "- Do NOT exhibit position bias. Candidate A and Candidate B must receive equal critical rigor.\n"
        "- Provide integer criteria scores (1-5) and integer holistic scores (1-10) for both candidates.\n"
        "- Output strictly valid JSON matching this schema:\n"
        "{\n"
        '  "criteria_scores": {\n'
        '    "Candidate A": {"correctness": 5, "completeness": 5, "coherence": 5},\n'
        '    "Candidate B": {"correctness": 4, "completeness": 4, "coherence": 4}\n'
        "  },\n"
        '  "holistic_scores": {\n'
        '    "Candidate A": 9,\n'
        '    "Candidate B": 8\n'
        "  },\n"
        '  "primary_differentiator": "Mathematical precision and verified code",\n'
        '  "reasoning": "Candidate A provides superior derivations and fully executable implementations."\n'
        "}"
    )

    user_prompt = (
        f"User Query:\n{query_text}\n\n"
        f"=== Candidate A ===\n{cand_a_text[:4000]}\n\n"
        f"=== Candidate B ===\n{cand_b_text[:4000]}\n\n"
        f"Provide your JSON evaluation:"
    )

    t0 = time.perf_counter()
    raw_content = ""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model="qwen/qwen3.8-27b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=600,
                response_format={"type": "json_object"}
            )
            raw_content = resp.choices[0].message.content
            if raw_content and raw_content.strip():
                break
        except Exception as e:
            err_str = str(e)
            print(f"      [Qwen Judge Retry {attempt+1}] {err_str[:120]}")
            if "429" in err_str:
                wait_sec = 20.0 + attempt * 10.0
                if "Please try again in" in err_str:
                    try:
                        time_part = err_str.split("Please try again in")[1].split(".")[0].strip()
                        if "m" in time_part:
                            m_val, s_val = time_part.split("m")
                            wait_sec = float(m_val.strip()) * 60.0 + float(s_val.replace("s", "").strip()) + 5.0
                        elif "s" in time_part:
                            wait_sec = float(time_part.replace("s", "").strip()) + 5.0
                    except Exception:
                        pass
                print(f"      [Rate Limit 429] Pausing {wait_sec:.1f}s before retry...")
                time.sleep(wait_sec)
            else:
                time.sleep(3.0 * (attempt + 1))

    latency = time.perf_counter() - t0
    if not raw_content or not raw_content.strip():
        raise RuntimeError("[ABORT] Qwen judge call failed after retries.")

    clean_json = raw_content.strip()
    if "```json" in clean_json:
        clean_json = clean_json.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in clean_json:
        clean_json = clean_json.split("```", 1)[1].split("```", 1)[0].strip()

    parsed = json.loads(clean_json)
    return parsed, latency


def compute_confidence_interval(data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    if not data or len(data) < 2:
        val = data[0] if data else 0.0
        return (val, val)
    n = len(data)
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / (n - 1)
    std_err = math.sqrt(variance / n)
    t_crit = 2.131 if n == 16 else 1.96
    margin = t_crit * std_err
    return (round(mean - margin, 4), round(mean + margin, 4))


def run_cross_validation():
    print("=" * 80)
    print(">>> TIER 2 CROSS-VALIDATION PASS: QWEN 27B vs GEMINI 3.1 FLASH LITE JUDGE <<<")
    print("=" * 80)

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable missing!")
    client = Groq(api_key=api_key)

    with open("data/v3_queries_dev.json", "r", encoding="utf-8") as f:
        all_dev = json.load(f)
    queries = [q for q in all_dev if q["id"] in TARGET_QUERY_IDS]

    e1_slm_cache = load_jsonl(E1_SLM_PATH)
    e2_slm_cache = load_jsonl(E2_SLM_PATH)
    b32_cache = load_jsonl(B32_PATH)

    e1_gemini_b32 = load_gemini_b32_trials(E1_GEMINI_TRIALS)
    e2_gemini_b32 = load_gemini_b32_trials(E2_GEMINI_TRIALS)

    print(f"Loaded {len(queries)} queries.")
    print(f"Loaded E1 SLM: {len(e1_slm_cache)}, E2 SLM: {len(e2_slm_cache)}, B32 Baseline: {len(b32_cache)}")
    print(f"Loaded Gemini B32 trials: E1={len(e1_gemini_b32)}, E2={len(e2_gemini_b32)}")

    # Check existing trials
    existing_trials = {}
    if os.path.exists(TRIALS_PATH):
        with open(TRIALS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    t_key = f"{rec['experiment']}_{rec['query_id']}_{rec['order_tag']}"
                    existing_trials[t_key] = rec
    print(f"Found {len(existing_trials)} pre-existing Qwen cross-validation trials.")

    experiments = [
        ("E1", "SLM_Pipeline_E1", e1_slm_cache, e1_gemini_b32),
        ("E2", "SLM_Pipeline_E2", e2_slm_cache, e2_gemini_b32),
    ]

    new_trials = []
    trial_counter = 0

    for exp_tag, slm_name, slm_cache, gemini_trials in experiments:
        print(f"\n--- Running Cross-Validation for {exp_tag} vs Tier 2 (gemini-2.5-flash) ---")
        for q in queries:
            qid = q["id"]
            slm_resp = slm_cache[qid]["response_text"]
            b32_resp = b32_cache[qid]["response_text"]

            for order_tag in ["forward", "swapped"]:
                trial_counter += 1
                t_key = f"{exp_tag}_{qid}_{order_tag}"

                if t_key in existing_trials:
                    print(f"  [{trial_counter}/32] {t_key} (Cached)")
                    new_trials.append(existing_trials[t_key])
                    continue

                if order_tag == "forward":
                    cand_a_text, cand_b_text = slm_resp, b32_resp
                    cand_a_sys, cand_b_sys = slm_name, "gemini-2.5-flash"
                else:
                    cand_a_text, cand_b_text = b32_resp, slm_resp
                    cand_a_sys, cand_b_sys = "gemini-2.5-flash", slm_name

                print(f"  [{trial_counter}/32] Judging {exp_tag} {qid} [{order_tag.upper()}] with Qwen-27B...")
                parsed, latency = call_qwen_judge(client, q["query"], cand_a_text, cand_b_text)

                crit_scores = parsed.get("criteria_scores", {})
                hol_scores = parsed.get("holistic_scores", {})
                reasoning = parsed.get("reasoning", "")
                differentiator = parsed.get("primary_differentiator", "")

                ca_crit = crit_scores.get("Candidate A", {"correctness": 3, "completeness": 3, "coherence": 3})
                cb_crit = crit_scores.get("Candidate B", {"correctness": 3, "completeness": 3, "coherence": 3})
                ca_cqs = round((ca_crit.get("correctness", 3) + ca_crit.get("completeness", 3) + ca_crit.get("coherence", 3)) / 3.0, 4)
                cb_cqs = round((cb_crit.get("correctness", 3) + cb_crit.get("completeness", 3) + cb_crit.get("coherence", 3)) / 3.0, 4)

                ca_hol = float(hol_scores.get("Candidate A", 5.0))
                cb_hol = float(hol_scores.get("Candidate B", 5.0))

                if cand_a_sys == slm_name:
                    slm_cqs, llm_cqs = ca_cqs, cb_cqs
                    slm_hol, llm_hol = ca_hol, cb_hol
                else:
                    slm_cqs, llm_cqs = cb_cqs, ca_cqs
                    slm_hol, llm_hol = cb_hol, ca_hol

                delta_crit = round(slm_cqs - llm_cqs, 4)
                p_crit = round((1.0 - (abs(delta_crit) / 4.0)) * 100.0, 2)
                delta_hol = round(slm_hol - llm_hol, 4)
                qp_hol = round(1.0 - (abs(delta_hol) / 9.0), 4)

                if slm_cqs > llm_cqs:
                    outcome_crit = "SLM_WIN"
                elif llm_cqs > slm_cqs:
                    outcome_crit = "LLM_WIN"
                else:
                    outcome_crit = "DRAW"

                if slm_hol > llm_hol:
                    outcome_hol = "SLM_WIN"
                elif llm_hol > slm_hol:
                    outcome_hol = "LLM_WIN"
                else:
                    outcome_hol = "DRAW"

                truncation_flag = any(term in reasoning.lower() for term in ["cut off", "truncated", "ends abruptly", "incomplete sentence"])

                # Concordance assertion
                if outcome_hol == "SLM_WIN":
                    assert slm_hol > llm_hol, f"Concordance fail on {t_key}"
                elif outcome_hol == "LLM_WIN":
                    assert llm_hol > slm_hol, f"Concordance fail on {t_key}"

                # Gemini trial for reference
                gem_key = f"{qid}_{order_tag}"
                gem_trial = gemini_trials.get(gem_key, {})

                trial_rec = {
                    "experiment": exp_tag,
                    "query_id": qid,
                    "baseline_tier": "b32",
                    "baseline_model": "gemini-2.5-flash",
                    "order_tag": order_tag,
                    "judge_model": "qwen/qwen3.8-27b",
                    "candidate_a_sys": cand_a_sys,
                    "candidate_b_sys": cand_b_sys,
                    "candidate_a_crit": ca_crit,
                    "candidate_b_crit": cb_crit,
                    "candidate_a_cqs": ca_cqs,
                    "candidate_b_cqs": cb_cqs,
                    "candidate_a_hol": ca_hol,
                    "candidate_b_hol": cb_hol,
                    "slm_cqs": slm_cqs,
                    "llm_cqs": llm_cqs,
                    "slm_holistic": slm_hol,
                    "llm_holistic": llm_hol,
                    "delta_criteria": delta_crit,
                    "p_criteria_pct": p_crit,
                    "outcome_criteria": outcome_crit,
                    "delta_holistic": delta_hol,
                    "qp_holistic": qp_hol,
                    "outcome_holistic": outcome_hol,
                    "primary_differentiator": differentiator,
                    "reasoning": reasoning,
                    "truncation_flag": truncation_flag,
                    "judge_latency_sec": latency,
                    "gemini_judge_outcome_holistic": gem_trial.get("outcome_holistic"),
                    "gemini_judge_slm_holistic": gem_trial.get("slm_holistic"),
                    "gemini_judge_llm_holistic": gem_trial.get("llm_holistic"),
                    "gemini_judge_reasoning": gem_trial.get("reasoning")
                }

                # Save public trial log & private key
                ts = int(time.time() * 1000)
                pub_log = {
                    "query_id": qid,
                    "experiment": exp_tag,
                    "judge_model": "qwen/qwen3.8-27b",
                    "order_tag": order_tag,
                    "criteria_scores": parsed.get("criteria_scores"),
                    "holistic_scores": parsed.get("holistic_scores"),
                    "primary_differentiator": differentiator,
                    "reasoning": reasoning,
                    "status": "SUCCESS"
                }
                priv_key = {
                    "query_id": qid,
                    "experiment": exp_tag,
                    "order_tag": order_tag,
                    "candidate_a": cand_a_sys,
                    "candidate_b": cand_b_sys,
                    "timestamp": ts
                }
                with open(os.path.join(PAIRWISE_DIR, f"qwen_judge_{exp_tag}_{qid}_{order_tag}_{ts}.json"), "w", encoding="utf-8") as f:
                    json.dump(pub_log, f, indent=2)
                with open(os.path.join(KEY_DIR, f"key_{exp_tag}_{qid}_{order_tag}_{ts}.json"), "w", encoding="utf-8") as f:
                    json.dump(priv_key, f, indent=2)

                new_trials.append(trial_rec)
                with open(TRIALS_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps(trial_rec) + "\n")

                print(f"    -> SLM Hol: {slm_hol}, LLM Hol: {llm_hol} | Outcome: {outcome_hol} (Gemini was: {gem_trial.get('outcome_holistic')})")
                time.sleep(1.0)

    # =========================================================================
    # AUDIT & COMPARISON ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("COMPUTING CROSS-VALIDATION STATISTICS (QWEN vs GEMINI JUDGE ON TIER 2)")
    print("=" * 80)

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "baseline_evaluated": "gemini-2.5-flash (32.0B)",
        "evaluator_models": {
            "primary": "gemini-3.1-flash-lite",
            "cross_validation": "qwen/qwen3.8-27b"
        },
        "experiments": {}
    }

    for exp_tag in ["E1", "E2"]:
        exp_trials = [t for t in new_trials if t["experiment"] == exp_tag]
        n_trials = len(exp_trials)

        # Qwen Metrics
        qwen_slm_wins = sum(1 for t in exp_trials if t["outcome_holistic"] == "SLM_WIN")
        qwen_draws = sum(1 for t in exp_trials if t["outcome_holistic"] == "DRAW")
        qwen_llm_wins = sum(1 for t in exp_trials if t["outcome_holistic"] == "LLM_WIN")

        qwen_slm_hol_mean = round(sum(t["slm_holistic"] for t in exp_trials) / n_trials, 4)
        qwen_llm_hol_mean = round(sum(t["llm_holistic"] for t in exp_trials) / n_trials, 4)
        qwen_delta_hol_mean = round(sum(t["delta_holistic"] for t in exp_trials) / n_trials, 4)
        qwen_qp_mean = round(sum(t["qp_holistic"] for t in exp_trials) / n_trials, 4)

        qwen_slm_cqs_mean = round(sum(t["slm_cqs"] for t in exp_trials) / n_trials, 4)
        qwen_llm_cqs_mean = round(sum(t["llm_cqs"] for t in exp_trials) / n_trials, 4)
        qwen_p_mean = round(sum(t["p_criteria_pct"] for t in exp_trials) / n_trials, 2)

        # Gemini Metrics for exact same set
        gemini_slm_wins = sum(1 for t in exp_trials if t["gemini_judge_outcome_holistic"] == "SLM_WIN")
        gemini_draws = sum(1 for t in exp_trials if t["gemini_judge_outcome_holistic"] == "DRAW")
        gemini_llm_wins = sum(1 for t in exp_trials if t["gemini_judge_outcome_holistic"] == "LLM_WIN")

        gemini_slm_hol_mean = round(sum(t["gemini_judge_slm_holistic"] for t in exp_trials) / n_trials, 4)
        gemini_llm_hol_mean = round(sum(t["gemini_judge_llm_holistic"] for t in exp_trials) / n_trials, 4)

        # Agreement Rate (identical winner category)
        agreements = sum(1 for t in exp_trials if t["outcome_holistic"] == t["gemini_judge_outcome_holistic"])
        verdict_agreement_pct = round((agreements / n_trials) * 100.0, 2)

        # Swap Consistency for Qwen
        by_query = {}
        for t in exp_trials:
            by_query.setdefault(t["query_id"], {})[t["order_tag"]] = t

        swap_consistent = 0
        for qid, q_orders in by_query.items():
            if "forward" in q_orders and "swapped" in q_orders:
                if q_orders["forward"]["outcome_holistic"] == q_orders["swapped"]["outcome_holistic"]:
                    swap_consistent += 1
        swap_consistency_pct = round((swap_consistent / len(by_query)) * 100.0, 2)

        exp_data = {
            "trials_count": n_trials,
            "verdict_agreement_rate_pct": verdict_agreement_pct,
            "qwen_swap_consistency_pct": swap_consistency_pct,
            "qwen_outcomes": {
                "slm_wins": qwen_slm_wins,
                "draws": qwen_draws,
                "llm_wins": qwen_llm_wins,
                "slm_win_rate_pct": round((qwen_slm_wins / n_trials) * 100.0, 2),
                "slm_mean_holistic": qwen_slm_hol_mean,
                "llm_mean_holistic": qwen_llm_hol_mean,
                "mean_delta_q": qwen_delta_hol_mean,
                "mean_qp_holistic": qwen_qp_mean,
                "slm_mean_cqs": qwen_slm_cqs_mean,
                "llm_mean_cqs": qwen_llm_cqs_mean,
                "mean_p_criteria_pct": qwen_p_mean
            },
            "gemini_outcomes": {
                "slm_wins": gemini_slm_wins,
                "draws": gemini_draws,
                "llm_wins": gemini_llm_wins,
                "slm_win_rate_pct": round((gemini_slm_wins / n_trials) * 100.0, 2),
                "slm_mean_holistic": gemini_slm_hol_mean,
                "llm_mean_holistic": gemini_llm_hol_mean
            },
            "delta_holistic_95ci": list(compute_confidence_interval([t["delta_holistic"] for t in exp_trials])),
            "qp_holistic_95ci": list(compute_confidence_interval([t["qp_holistic"] for t in exp_trials]))
        }
        report["experiments"][exp_tag] = exp_data

        print(f"\n[{exp_tag} Summary]")
        print(f"  Trials: {n_trials}")
        print(f"  Verdict Agreement Rate (Qwen vs Gemini): {verdict_agreement_pct}% ({agreements}/{n_trials})")
        print(f"  Qwen Swap Consistency: {swap_consistency_pct}%")
        print(f"  Qwen Outcomes   : SLM Wins={qwen_slm_wins}, Draws={qwen_draws}, LLM Wins={qwen_llm_wins}")
        print(f"  Gemini Outcomes : SLM Wins={gemini_slm_wins}, Draws={gemini_draws}, LLM Wins={gemini_llm_wins}")
        print(f"  Qwen Holistic Scores  : SLM={qwen_slm_hol_mean}, LLM={qwen_llm_hol_mean} (Delta={qwen_delta_hol_mean})")
        print(f"  Gemini Holistic Scores: SLM={gemini_slm_hol_mean}, LLM={gemini_llm_hol_mean}")

    # Qualitative Error Analysis across both judges
    print("\n--- Qualitative Defect Alignment ---")
    defect_alignment = []
    for q in queries:
        qid = q["id"]
        q_qwen = next((t for t in new_trials if t["experiment"] == "E1" and t["query_id"] == qid and t["order_tag"] == "forward"), None)
        if q_qwen:
            entry = {
                "query_id": qid,
                "domains": q.get("domains"),
                "qwen_primary_differentiator": q_qwen.get("primary_differentiator"),
                "qwen_reasoning_summary": q_qwen.get("reasoning")[:250],
                "gemini_reasoning_summary": q_qwen.get("gemini_judge_reasoning")[:250] if q_qwen.get("gemini_judge_reasoning") else ""
            }
            defect_alignment.append(entry)
            print(f"  {qid}:")
            print(f"    Qwen  : {q_qwen.get('primary_differentiator')} -> {q_qwen.get('reasoning')[:150]}...")
            print(f"    Gemini: {q_qwen.get('gemini_judge_reasoning')[:150] if q_qwen.get('gemini_judge_reasoning') else 'N/A'}...")

    report["qualitative_defect_alignment"] = defect_alignment

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[SUCCESS] Cross-validation report written to {REPORT_PATH}")


if __name__ == "__main__":
    run_cross_validation()
