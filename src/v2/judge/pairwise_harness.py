"""
True Pairwise Head-to-Head LLM Judge Harness with Position Swapping (Option 1 SLM-Focused Benchmark)
Judge Model: qwen/qwen3.8-27b on Groq (Non-reasoning dense evaluator, zero hidden tokens)
Evaluates Candidate A vs Candidate B with independent randomized orders,
strict criteria scoring (Correctness, Completeness, Coherence), and separated identity un-blinding.
"""

import json
import os
import re
import socket
import time
import urllib.request
from typing import Dict, List, Any, Optional, Tuple

socket.setdefaulttimeout(25)

PAIRWISE_JUDGE_SYSTEM_PROMPT = """You are an impartial, expert AI judge evaluating two candidate responses (Candidate A and Candidate B) to a technical user query.

Evaluation Criteria:
1. Correctness (1-5): Factual, mathematical, and algorithmic precision.
2. Completeness (1-5): Thorough fulfillment of all problem requirements and constraints.
3. Coherence (1-5): Logical structure, readability, unified authoritative voice, and seamless synthesis.

Instructions:
- Evaluate both candidates objectively.
- Evaluate both candidates objectively without position bias. Do NOT favor Candidate B simply because it appears second; evaluate Candidate A and Candidate B with equal critical rigor.
- Assign integer criteria scores (1-5) to both Candidate A and Candidate B.
- Select the winning candidate ("Candidate A", "Candidate B", or "Tie").
- Select the winning candidate ("Candidate A", "Candidate B", or "Tie"). If both candidates provide comparable quality, declare "Tie".
- State the primary differentiator and concise, rigorous reasoning.

Output strictly valid JSON matching this exact schema:
{
  "selected_candidate": "Candidate A",
  "criteria_scores": {
    "Candidate A": {"correctness": 5, "completeness": 5, "coherence": 5},
    "Candidate B": {"correctness": 4, "completeness": 4, "coherence": 4}
  },
  "primary_differentiator": "correctness",
  "reasoning": "Candidate A provided a superior derivation."
}
"""

class PairwiseLLMJudgeHarness:
    def __init__(
        self,
        judge_model_name: str = "qwen/qwen3.8-27b",
        api_key: Optional[str] = None
    ):
        self.judge_model_name = judge_model_name
        if not api_key:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key and os.path.exists(".env"):
                with open(".env", "r") as f:
                    for line in f:
                        if line.startswith("GROQ_API_KEY="):
                            api_key = line.strip().split("=", 1)[1]
                            break
        self.api_key = api_key

    def evaluate_pair(
        self,
        query_id: str,
        query_text: str,
        system_a_id: str,
        text_a: str,
        system_b_id: str,
        text_b: str,
        order_tag: str, # "forward" (A is Candidate A) or "swapped" (B is Candidate A)
        judge_log_dir: str = "logs/judge_pairwise",
        key_log_dir: str = "logs/judge_keys"
    ) -> Dict[str, Any]:
        os.makedirs(judge_log_dir, exist_ok=True)
        os.makedirs(key_log_dir, exist_ok=True)

        if order_tag == "forward":
            candidate_a_text = text_a.strip()
            candidate_b_text = text_b.strip()
            cand_a_sys = system_a_id
            cand_b_sys = system_b_id
        else:
            candidate_a_text = text_b.strip()
            candidate_b_text = text_a.strip()
            cand_a_sys = system_b_id
            cand_b_sys = system_a_id

        cand_a_trimmed = False
        cand_b_trimmed = False
        trimmed_info = {}

        def _trim_for_judge(text: str, cand_label: str, system_name: str, max_chars: int = 11000) -> Tuple[str, bool]:
            text = text.strip()
            if len(text) <= max_chars:
                return text, False
            # Find a natural section or paragraph boundary before max_chars to avoid severing code blocks
            cutoff = max_chars
            last_break = text.rfind("\n\n", max_chars - 2500, max_chars)
            if last_break != -1:
                cutoff = last_break
            print(f"[Judge Warning] Response for {system_name} ({cand_label}, {len(text)} chars) exceeded max_chars ({max_chars}). Non-destructive end-trimming applied at character {cutoff}.", flush=True)
            trimmed_text = text[:cutoff] + f"\n\n...[Remaining {len(text) - cutoff} characters trimmed at natural section boundary for context limit]..."
            return trimmed_text, True

        cand_a_formatted, cand_a_trimmed = _trim_for_judge(candidate_a_text, "Candidate A", cand_a_sys)
        cand_b_formatted, cand_b_trimmed = _trim_for_judge(candidate_b_text, "Candidate B", cand_b_sys)
        if cand_a_trimmed:
            trimmed_info["Candidate A"] = {"system": cand_a_sys, "original_length": len(candidate_a_text)}
        if cand_b_trimmed:
            trimmed_info["Candidate B"] = {"system": cand_b_sys, "original_length": len(candidate_b_text)}

        user_prompt = (
            f"User Query:\n{query_text}\n\n"
            f"=== Candidate A ===\n{cand_a_formatted}\n\n"
            f"=== Candidate B ===\n{cand_b_formatted}\n\n"
            f"Provide your JSON evaluation:"
        )

        start_t = time.perf_counter()
        raw_response, in_tok, out_tok, error_msg = self._call_judge_sync(user_prompt)
        end_t = time.perf_counter()
        latency_ms = (end_t - start_t) * 1000.0

        timestamp_ms = int(time.time() * 1000)
        pair_key = f"{system_a_id}_vs_{system_b_id}"

        if error_msg:
            public_record = {
                "query_id": query_id,
                "judge_model": self.judge_model_name,
                "order_tag": order_tag,
                "candidates_shown": ["Candidate A", "Candidate B"],
                "status": "FAILED",
                "error_detail": error_msg,
                "latency_ms": latency_ms,
                "timestamp_ms": timestamp_ms,
                "trimmed": bool(trimmed_info),
                "trim_details": trimmed_info if trimmed_info else None
            }
            public_file = os.path.join(judge_log_dir, f"judge_{query_id}_{pair_key}_{order_tag}_{timestamp_ms}.json")
            with open(public_file, "w", encoding="utf-8") as f:
                json.dump(public_record, f, indent=2)

            key_record = {
                "query_id": query_id,
                "pair_key": pair_key,
                "order_tag": order_tag,
                "candidate_a_system": cand_a_sys,
                "candidate_b_system": cand_b_sys,
                "status": "FAILED",
                "error_detail": error_msg,
                "public_log_file": public_file,
                "timestamp_ms": timestamp_ms
            }
            key_file = os.path.join(key_log_dir, f"key_{query_id}_{pair_key}_{order_tag}_{timestamp_ms}.json")
            with open(key_file, "w", encoding="utf-8") as f:
                json.dump(key_record, f, indent=2)

            return {
                "query_id": query_id,
                "system_a": system_a_id,
                "system_b": system_b_id,
                "order_tag": order_tag,
                "status": "FAILED",
                "error_detail": error_msg,
                "public_log": public_file,
                "key_log": key_file
            }

        parsed = self._parse_json_verdict(raw_response)
        selected_alias = parsed.get("selected_candidate", "Tie")

        if selected_alias == "Candidate A":
            unblinded_winner = cand_a_sys
        elif selected_alias == "Candidate B":
            unblinded_winner = cand_b_sys
        else:
            unblinded_winner = "Tie"

        crit_scores = parsed.get("criteria_scores", {})
        scores_by_system = {}
        if "Candidate A" in crit_scores:
            scores_by_system[cand_a_sys] = crit_scores["Candidate A"]
        if "Candidate B" in crit_scores:
            scores_by_system[cand_b_sys] = crit_scores["Candidate B"]

        # Compute Quality Proximity (P) and Signed Delta (ΔQ) per Three-Dimensional Framework
        cqs_a = None
        cqs_b = None
        quality_proximity = None
        quality_delta_a_minus_b = None
        if "Candidate A" in crit_scores and "Candidate B" in crit_scores:
            sc_a = crit_scores["Candidate A"]
            sc_b = crit_scores["Candidate B"]
            cqs_a = round((sc_a.get("correctness", 0) + sc_a.get("completeness", 0) + sc_a.get("coherence", 0)) / 3.0, 4)
            cqs_b = round((sc_b.get("correctness", 0) + sc_b.get("completeness", 0) + sc_b.get("coherence", 0)) / 3.0, 4)
            diff = abs(cqs_a - cqs_b)
            quality_proximity = round(max(0.0, min(1.0, 1.0 - (diff / 4.0))), 4)
            quality_delta_a_minus_b = round(cqs_a - cqs_b, 4)

        public_record = {
            "query_id": query_id,
            "judge_model": self.judge_model_name,
            "order_tag": order_tag,
            "status": "SUCCESS",
            "candidates_shown": ["Candidate A", "Candidate B"],
            "selected_candidate": selected_alias,
            "criteria_scores": crit_scores,
            "cqs_candidate_a": cqs_a,
            "cqs_candidate_b": cqs_b,
            "quality_proximity": quality_proximity,
            "quality_delta_a_minus_b": quality_delta_a_minus_b,
            "primary_differentiator": parsed.get("primary_differentiator", "correctness"),
            "reasoning": parsed.get("reasoning", ""),
            "prompt_tokens": in_tok,
            "completion_tokens": out_tok,
            "latency_ms": latency_ms,
            "timestamp_ms": timestamp_ms,
            "trimmed": bool(trimmed_info),
            "trim_details": trimmed_info if trimmed_info else None
        }
        public_file = os.path.join(judge_log_dir, f"judge_{query_id}_{pair_key}_{order_tag}_{timestamp_ms}.json")
        with open(public_file, "w", encoding="utf-8") as f:
            json.dump(public_record, f, indent=2)

        key_record = {
            "query_id": query_id,
            "pair_key": pair_key,
            "order_tag": order_tag,
            "status": "SUCCESS",
            "candidate_a_system": cand_a_sys,
            "candidate_b_system": cand_b_sys,
            "selected_alias": selected_alias,
            "unblinded_winner": unblinded_winner,
            "scores_by_system": scores_by_system,
            "cqs_candidate_a": cqs_a,
            "cqs_candidate_b": cqs_b,
            "quality_proximity": quality_proximity,
            "quality_delta_a_minus_b": quality_delta_a_minus_b,
            "trimmed": bool(trimmed_info),
            "trim_details": trimmed_info if trimmed_info else None,
            "public_log_file": public_file,
            "timestamp_ms": timestamp_ms
        }
        key_file = os.path.join(key_log_dir, f"key_{query_id}_{pair_key}_{order_tag}_{timestamp_ms}.json")
        with open(key_file, "w", encoding="utf-8") as f:
            json.dump(key_record, f, indent=2)

        return {
            "query_id": query_id,
            "system_a": system_a_id,
            "system_b": system_b_id,
            "order_tag": order_tag,
            "status": "SUCCESS",
            "selected_alias": selected_alias,
            "unblinded_winner": unblinded_winner,
            "scores_by_system": scores_by_system,
            "cqs_candidate_a": cqs_a,
            "cqs_candidate_b": cqs_b,
            "quality_proximity": quality_proximity,
            "quality_delta_a_minus_b": quality_delta_a_minus_b,
            "reasoning": parsed.get("reasoning", ""),
            "public_log": public_file,
            "key_log": key_file,
            "latency_ms": latency_ms
        }

    def _call_judge_sync(self, prompt: str) -> Tuple[str, int, int, Optional[str]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "curl/7.88.1"
        }
        body = {
            "model": self.judge_model_name,
            "messages": [
                {"role": "system", "content": PAIRWISE_JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
            "max_tokens": 2048
        }

        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST"
        )

        last_err = None
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=20) as response:
                    res_data = json.loads(response.read().decode("utf-8"))
                    text = res_data["choices"][0]["message"]["content"]
                    in_tok = res_data.get("usage", {}).get("prompt_tokens", len(prompt.split()) * 2)
                    out_tok = res_data.get("usage", {}).get("completion_tokens", len(text.split()) * 2)
                    return text, in_tok, out_tok, None
            except urllib.error.HTTPError as e:
                err_content = e.read().decode("utf-8", errors="ignore")
                last_err = f"HTTPError {e.code}: {err_content[:200]}"
                if e.code == 429:
                    wait_s = 5.0 * (attempt + 1)
                    if "try again in" in err_content:
                        try:
                            raw = err_content.split("try again in")[1].split("Need more tokens")[0].strip()
                            raw = raw.rstrip(".")
                            if "ms" in raw:
                                ms_val = float(raw.replace("ms", "").strip())
                                wait_s = (ms_val / 1000.0) + 0.5
                            elif "m" in raw:
                                parts = raw.split("m")
                                mins = float(parts[0])
                                secs_str = parts[1].rstrip("s").strip() if len(parts) > 1 else "0"
                                secs = float(secs_str) if secs_str else 0.0
                                wait_s = mins * 60.0 + secs + 1.0
                            else:
                                secs_val = float(raw.rstrip("s").strip())
                                wait_s = secs_val + 1.0
                        except Exception:
                            wait_s = 5.0 * (attempt + 1)
                    wait_s = max(0.5, min(wait_s, 30.0))
                    print(f"[{self.judge_model_name}] Groq 429 Rate Limit. Waiting {wait_s:.1f}s before retry {attempt+1}/6...", flush=True)
                    end_wait = time.time() + wait_s
                    while time.time() < end_wait:
                        time.sleep(min(1.0, max(0.05, end_wait - time.time())))
                else:
                    time.sleep(1.5)
            except Exception as e:
                last_err = str(e)
                time.sleep(1.5)

        return "", 0, 0, f"Exhausted 6 retries: {last_err}"

    def _parse_json_verdict(self, raw_text: str) -> Dict[str, Any]:
        cleaned = re.sub(r"^```json\s*", "", raw_text.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"^```\s*", "", cleaned)
        cleaned = re.sub(r"```$", "", cleaned.strip())
        try:
            return json.loads(cleaned)
        except Exception:
            return {
                "selected_candidate": "Tie",
                "criteria_scores": {},
                "primary_differentiator": "inconclusive",
                "reasoning": "Fallback on unparseable output."
            }
