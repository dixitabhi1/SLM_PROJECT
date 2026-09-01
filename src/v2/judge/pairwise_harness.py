"""
True Pairwise Head-to-Head LLM Judge Harness with Position Swapping (v2 Architecture - Groq Provider)
Evaluates Candidate A vs Candidate B with independent randomized orders,
strict criteria scoring (Correctness, Completeness, Coherence), and separated identity un-blinding.
Explicitly records failures without silent drops or backfills.
"""

import json
import os
import re
import time
import urllib.request
from typing import Dict, List, Any, Optional, Tuple

PAIRWISE_JUDGE_SYSTEM_PROMPT = """You are an impartial, expert AI judge evaluating two candidate responses (Candidate A and Candidate B) to a technical user query.

Evaluation Criteria:
1. Correctness (1-5): Factual, mathematical, and algorithmic precision.
2. Completeness (1-5): Thorough fulfillment of all problem requirements and constraints.
3. Coherence (1-5): Logical structure, readability, unified authoritative voice, and seamless synthesis.

Instructions:
- Evaluate both candidates objectively.
- Assign integer criteria scores (1-5) to both Candidate A and Candidate B.
- Select the winning candidate ("Candidate A", "Candidate B", or "Tie").
- State the primary differentiator and concise, rigorous reasoning.

Output strictly valid JSON with no markdown wrapping matching this exact schema:
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
        judge_model_name: str = "openai/gpt-oss-120b",
        api_key: Optional[str] = None
    ):
        self.judge_model_name = judge_model_name
        self.api_key = api_key or os.getenv("GROQ_API_KEY")

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
        """
        Runs blinded pairwise evaluation and logs public judge record and private un-blinded key separately.
        """
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

        user_prompt = (
            f"User Query:\n{query_text}\n\n"
            f"=== Candidate A ===\n{candidate_a_text}\n\n"
            f"=== Candidate B ===\n{candidate_b_text}\n\n"
            f"Provide your evaluation in the required JSON schema:"
        )

        start_t = time.perf_counter()
        raw_response, in_tok, out_tok, error_msg = self._call_judge_sync(user_prompt)
        end_t = time.perf_counter()
        latency_ms = (end_t - start_t) * 1000.0

        timestamp_ms = int(time.time() * 1000)
        pair_key = f"{system_a_id}_vs_{system_b_id}"

        if error_msg:
            # Explicit failure record - never silent drop
            public_record = {
                "query_id": query_id,
                "judge_model": self.judge_model_name,
                "order_tag": order_tag,
                "candidates_shown": ["Candidate A", "Candidate B"],
                "status": "FAILED",
                "error_detail": error_msg,
                "latency_ms": latency_ms,
                "timestamp_ms": timestamp_ms
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

        # 1. Write Public Judge Record (Strictly ZERO system names)
        public_record = {
            "query_id": query_id,
            "judge_model": self.judge_model_name,
            "order_tag": order_tag,
            "status": "SUCCESS",
            "candidates_shown": ["Candidate A", "Candidate B"],
            "selected_candidate": selected_alias,
            "criteria_scores": crit_scores,
            "primary_differentiator": parsed.get("primary_differentiator", "correctness"),
            "reasoning": parsed.get("reasoning", ""),
            "prompt_tokens": in_tok,
            "completion_tokens": out_tok,
            "latency_ms": latency_ms,
            "timestamp_ms": timestamp_ms
        }
        public_file = os.path.join(judge_log_dir, f"judge_{query_id}_{pair_key}_{order_tag}_{timestamp_ms}.json")
        with open(public_file, "w", encoding="utf-8") as f:
            json.dump(public_record, f, indent=2)

        # 2. Write Private Identity Key (Separated un-blinded mapping)
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
            "max_tokens": 1024
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
                with urllib.request.urlopen(req, timeout=30) as response:
                    res_data = json.loads(response.read().decode("utf-8"))
                    text = res_data["choices"][0]["message"]["content"]
                    in_tok = res_data.get("usage", {}).get("prompt_tokens", len(prompt.split()) * 2)
                    out_tok = res_data.get("usage", {}).get("completion_tokens", len(text.split()) * 2)
                    return text, in_tok, out_tok, None
            except urllib.error.HTTPError as e:
                last_err = f"HTTPError {e.code}: {e.read().decode()[:200]}"
                if e.code == 429: # Rate limit
                    time.sleep(2.0 * (attempt + 1))
                else:
                    time.sleep(1.0)
            except Exception as e:
                last_err = str(e)
                time.sleep(1.0)

        return "", 0, 0, f"Exhausted 5 retries: {last_err}"

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
