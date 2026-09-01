"""
Decomposer SLM (v2 Architecture)
Handles initial prompt decomposition and hierarchical re-decomposition on loop-back.
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from ...models.base import BaseModelRunner, ModelResponse

DECOMPOSER_V2_SYSTEM_PROMPT = """You are an expert AI task decomposition model.
Your task is to decompose compound queries into focused, single-domain subtasks for specialized SLMs.

Domain Categories:
- coding: algorithms, python scripts, debugging, implementations
- math: formulas, equations, calculus, proofs, numerical derivations
- reasoning: formal logic, trade-off analysis, causality, verification
- retrieval: factual specifications, RFCs, standards, law, documentation
- general: overview, background synthesis

Output strictly valid JSON with no markdown wrapping:
{
  "subtasks": [
    {
      "id": "node_1",
      "text": "Subtask instruction",
      "capability": "math",
      "dependencies": []
    }
  ]
}
"""

RE_DECOMPOSER_SYSTEM_PROMPT = """You are an expert subtask refinement model.
The given task spans multiple capability domains and must be split into 2 or 3 finer-grained, domain-isolated subtasks.

Output strictly valid JSON:
{
  "subtasks": [
    {
      "id": "node_X.1",
      "text": "First domain-specific part",
      "capability": "math",
      "dependencies": []
    },
    {
      "id": "node_X.2",
      "text": "Second domain-specific part",
      "capability": "coding",
      "dependencies": ["node_X.1"]
    }
  ]
}
"""

class DecomposerSLM_v2:
    def __init__(self, model_runner: BaseModelRunner):
        self.runner = model_runner

    async def decompose_initial(self, query_text: str) -> Dict[str, Any]:
        """Performs initial Depth=0 decomposition."""
        prompt = f"User Query: {query_text}\n\nDecompose into discrete subtasks:"
        resp = await self.runner.generate(
            prompt=prompt,
            system_prompt=DECOMPOSER_V2_SYSTEM_PROMPT,
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
            system_prompt=RE_DECOMPOSER_SYSTEM_PROMPT,
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
            data = json.loads(cleaned)
            if not isinstance(data, dict) or "subtasks" not in data or not isinstance(data["subtasks"], list) or len(data["subtasks"]) == 0:
                raise ValueError("Invalid subtasks array.")
            
            subtasks = data["subtasks"]
            for idx, task in enumerate(subtasks):
                if not isinstance(task, dict):
                    task = {"text": str(task)}
                if "id" not in task:
                    task["id"] = f"{fallback_prefix}_{idx+1}" if "_" in fallback_prefix else f"{fallback_prefix}.{idx+1}"
                if "text" not in task or not task["text"].strip():
                    task["text"] = default_query
                if "capability" not in task or task["capability"] not in ["coding", "math", "reasoning", "retrieval", "general"]:
                    task["capability"] = "general"
                if "dependencies" not in task or not isinstance(task["dependencies"], list):
                    task["dependencies"] = []
                task["depth"] = depth

            return {"subtasks": subtasks}, True
        except Exception:
            fallback = {
                "subtasks": [
                    {
                        "id": f"{fallback_prefix}_1" if "_" in fallback_prefix else f"{fallback_prefix}.1",
                        "text": default_query,
                        "capability": "general",
                        "dependencies": [],
                        "depth": depth
                    }
                ]
            }
            return fallback, False

