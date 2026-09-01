"""
LLM-as-Judge Blind Evaluation Harness (v2 Architecture)
Shuffles candidate responses, strips system tags, and evaluates choices across Correctness, Completeness, Coherence.
"""

import json
import random
import time
from typing import Dict, List, Any, Optional, Tuple
from ...models.base import BaseModelRunner, ModelResponse

JUDGE_SYSTEM_PROMPT = """You are an impartial, expert AI judge evaluating responses to complex, multi-domain search queries.

You will be presented with an original user query and a randomized, anonymized list of candidate responses (Candidate A, Candidate B, etc.).
Your task:
1. Objectively evaluate each candidate on:
   - Correctness (1-5): Factual, mathematical, and code accuracy.
   - Completeness (1-5): Thorough coverage of all query constraints.
   - Coherence (1-5): Logical structure, clarity, and synthesis.
2. Select the single best overall response that provides the highest quality answer.
3. State your concise, rigorous reasoning explaining why the winner was chosen.

Output strictly valid JSON with no markdown wrapping:
{
  "selected_candidate": "Candidate A",
  "criteria_scores": {
    "Candidate A": {"correctness": 5, "completeness": 5, "coherence": 5},
    "Candidate B": {"correctness": 4, "completeness": 4, "coherence": 4}
  },
  "primary_differentiator": "correctness | completeness | coherence",
  "reasoning": "Detailed explanation of why the selected candidate is superior."
}
"""

class AnonymousLLMJudge:
    def __init__(self, judge_model_runner: BaseModelRunner):
        self.runner = judge_model_runner

    async def judge_query_candidates(
        self,
        query_id: str,
        query_text: str,
        candidate_pool: Dict[str, str], # system_id -> response_text (e.g. "slm_pipeline": "...", "llama_70b": "...")
        shuffle_seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Anonymizes, randomizes, prompts judge, and outputs structured verdict.
        """
        seed = shuffle_seed if shuffle_seed is not None else int(time.time() * 1000) % 1000000
        rng = random.Random(seed)

        # 1. Shuffle systems
        system_ids = list(candidate_pool.keys())
        rng.shuffle(system_ids)

        # 2. Assign Anonymous Aliases
        alias_to_system = {}
        system_to_alias = {}
        candidate_blocks = []

        for idx, sys_id in enumerate(system_ids):
            alias = f"Candidate {chr(65 + idx)}" # Candidate A, B, C...
            alias_to_system[alias] = sys_id
            system_to_alias[sys_id] = alias
            resp_text = candidate_pool[sys_id].strip()
            candidate_blocks.append(f"### {alias}:\n{resp_text}")

        judge_prompt = (
            f"Original User Query:\n{query_text}\n\n"
            f"=== Candidate Responses ===\n\n"
            + "\n\n----------------------------------------\n\n".join(candidate_blocks)
            + "\n\nSelect the best candidate and score all candidates:"
        )

        # 3. Dispatch Judge Inference
        resp: ModelResponse = await self.runner.generate(
            prompt=judge_prompt,
            system_prompt=JUDGE_SYSTEM_PROMPT,
            temperature=0.0
        )

        parsed_verdict = self._parse_verdict(resp.text, system_ids[0])
        selected_alias = parsed_verdict.get("selected_candidate", "Candidate A")
        selected_system = alias_to_system.get(selected_alias, system_ids[0])

        return {
            "query_id": query_id,
            "judge_model": self.runner.model_name,
            "shuffle_seed": seed,
            "candidates_shown": list(alias_to_system.keys()),
            "selected_alias": selected_alias,
            "selected_system": selected_system,
            "criteria_scores": parsed_verdict.get("criteria_scores", {}),
            "primary_differentiator": parsed_verdict.get("primary_differentiator", "correctness"),
            "reasoning": parsed_verdict.get("reasoning", "Selected based on technical rigor."),
            "alias_to_system_mapping": alias_to_system,
            "raw_judge_response": resp.text
        }

    def _parse_verdict(self, raw_text: str, fallback_sys: str) -> Dict[str, Any]:
        import re
        cleaned = re.sub(r"^```json\s*", "", raw_text.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"^```\s*", "", cleaned)
        cleaned = re.sub(r"```$", "", cleaned.strip())

        try:
            return json.loads(cleaned)
        except Exception:
            return {
                "selected_candidate": "Candidate A",
                "criteria_scores": {},
                "primary_differentiator": "correctness",
                "reasoning": "Fallback selection due to unparseable judge output."
            }

