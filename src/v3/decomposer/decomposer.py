"""
Decomposer SLM (v3 Architecture - 8 Domains)
Handles initial prompt decomposition and hierarchical re-decomposition for the 8-specialist pool.
Specialist Domains:
1. coding
2. mathematics
3. formal_reasoning
4. retrieval_qa
5. science_tech
6. structured_data
7. creative_synthesis
8. systems_ops
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from ...models.base import BaseModelRunner, ModelResponse

DECOMPOSER_V3_SYSTEM_PROMPT = """You are an expert AI task decomposition model for an 8-specialist Small Language Model pipeline.
Your task is to decompose compound queries into focused, single-domain subtasks for specialized SLMs.

Atomic Stop Condition:
- If the user query is already focused and self-contained within a single domain (e.g. writing a single algorithmic module, solving a purely mathematical derivation, performing a factual retrieval, or analyzing a single logical problem), do NOT artificially fragment it.
- In such cases, emit exactly ONE root subtask ("node_1") containing the complete, unmodified original instruction.
- Only decompose queries that genuinely span multiple distinct capability domains (e.g. database schema design + Python ORM code, scientific formula derivation + software implementation, RFC retrieval + architecture reasoning).

Specialist Domain Categories (Strict 8-Domain Pool):
1. coding: Python/C++ code, algorithms, data structures, scripts, debugging, unit tests.
2. mathematics: algebraic derivations, calculus, linear algebra, proofs, probability, numerical optimization.
3. formal_reasoning: logical verification, trade-off analysis, counterfactual reasoning, constraint validation.
4. retrieval_qa: factual knowledge, RFCs, API specifications, compliance standards, technical documentation.
5. science_tech: physics, mechanics, chemistry, thermodynamics, electronics, material properties.
6. structured_data: relational schemas, SQL, table definitions, ORM data models, JSON/Protobuf formats.
7. creative_synthesis: high-level architecture overviews, executive summaries, technical reports.
8. systems_ops: shell scripting, Docker/k8s, process management, POSIX, networking, concurrency.

Output strictly valid JSON with no markdown wrapping:
{
  "subtasks": [
    {
      "id": "node_1",
      "text": "Specific domain instruction",
      "capability": "structured_data",
      "dependencies": []
    },
    {
      "id": "node_2",
      "text": "Dependent domain instruction",
      "capability": "coding",
      "dependencies": ["node_1"]
    }
  ]
}
"""

RE_DECOMPOSER_V3_SYSTEM_PROMPT = """You are an expert subtask refinement model for an 8-specialist SLM pipeline.
The given task spans multiple capability domains and must be split into 2 or 3 finer-grained, domain-isolated subtasks.

Available Domains: coding, mathematics, formal_reasoning, retrieval_qa, science_tech, structured_data, creative_synthesis, systems_ops.

Output strictly valid JSON:
{
  "subtasks": [
    {
      "id": "node_X.1",
      "text": "First domain-isolated part",
      "capability": "structured_data",
      "dependencies": []
    },
    {
      "id": "node_X.2",
      "text": "Second domain-isolated part",
      "capability": "coding",
      "dependencies": ["node_X.1"]
    }
  ]
}
"""

class DecomposerSLM_v3:
    def __init__(self, model_runner: BaseModelRunner):
        self.runner = model_runner

    async def decompose_initial(self, query_text: str) -> Dict[str, Any]:
        """Performs initial Depth=0 decomposition."""
        prompt = f"User Query: {query_text}\n\nDecompose into discrete subtasks:"
        resp = await self.runner.generate(
            prompt=prompt,
            system_prompt=DECOMPOSER_V3_SYSTEM_PROMPT,
            temperature=0.0
        )
        dag, is_valid = self._parse_dag_json(resp.text, fallback_prefix="node", default_query=query_text, depth=0)
        return {
            "subtasks": dag["subtasks"],
            "is_schema_valid": is_valid,
            "raw_response": resp.text,
            "model_response": resp
        }

    async def re_decompose_task(self, parent_task: Dict[str, Any], next_depth: int) -> Dict[str, Any]:
        """
        Re-decomposes a multi-color parent task into hierarchical child subtasks (e.g. node_1 -> node_1.1, node_1.2).
        """
        parent_id = parent_task["id"]
        parent_text = parent_task["text"]
        parent_deps = parent_task.get("dependencies", [])

        prompt = (
            f"Parent Task ID: {parent_id}\n"
            f"Parent Task Text: {parent_text}\n"
            f"Re-decompose this task into single-domain subtasks with IDs '{parent_id}.1', '{parent_id}.2':"
        )

        resp = await self.runner.generate(
            prompt=prompt,
            system_prompt=RE_DECOMPOSER_V3_SYSTEM_PROMPT,
            temperature=0.0
        )

        dag, is_valid = self._parse_dag_json(resp.text, fallback_prefix=parent_id, default_query=parent_text, depth=next_depth)

        # Ensure child IDs follow hierarchical dot notation
        child_subtasks = []
        for idx, sub in enumerate(dag["subtasks"]):
            cid = f"{parent_id}.{idx+1}"
            sub["id"] = cid
            sub["parent_id"] = parent_id
            sub["depth"] = next_depth

            # Remap internal dependencies
            remapped_deps = []
            for d in sub.get("dependencies", []):
                if d != cid:
                    remapped_deps.append(d)
            # The first child inherits the parent's external dependencies
            if idx == 0 and parent_deps:
                for pd in parent_deps:
                    if pd not in remapped_deps:
                        remapped_deps.append(pd)
            # Subsequent children depend on preceding sibling by default if not set
            elif idx > 0 and not remapped_deps:
                remapped_deps.append(f"{parent_id}.{idx}")

            sub["dependencies"] = remapped_deps
            child_subtasks.append(sub)

        return {
            "parent_id": parent_id,
            "child_subtasks": child_subtasks,
            "is_schema_valid": is_valid,
            "model_response": resp
        }

    def _parse_dag_json(self, raw_text: str, fallback_prefix: str, default_query: str, depth: int) -> Tuple[Dict[str, Any], bool]:
        cleaned = re.sub(r"^```json\s*", "", raw_text.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"^```\s*", "", cleaned)
        cleaned = re.sub(r"```$", "", cleaned.strip())

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict) and "subtasks" in parsed and isinstance(parsed["subtasks"], list) and len(parsed["subtasks"]) > 0:
                for idx, sub in enumerate(parsed["subtasks"]):
                    if not isinstance(sub, dict):
                        return self._fallback_dag(fallback_prefix, default_query, depth), False
                    if "id" not in sub:
                        sub["id"] = f"{fallback_prefix}_{idx+1}"
                    if "text" not in sub:
                        sub["text"] = default_query
                    if "capability" not in sub:
                        sub["capability"] = "coding"
                    if "dependencies" not in sub:
                        sub["dependencies"] = []
                    sub["depth"] = depth
                return parsed, True
        except Exception:
            pass

        return self._fallback_dag(fallback_prefix, default_query, depth), False

    def _fallback_dag(self, prefix: str, query: str, depth: int) -> Dict[str, Any]:
        return {
            "subtasks": [
                {
                    "id": f"{prefix}_1",
                    "text": query,
                    "capability": "coding",
                    "dependencies": [],
                    "depth": depth
                }
            ]
        }
