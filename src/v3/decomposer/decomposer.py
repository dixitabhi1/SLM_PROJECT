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
Your task is to decompose technical user queries into focused, single-domain subtasks for specialized SLMs.

Decomposition Rules:
1. Compound Multi-Domain Queries (MUST DECOMPOSE):
   - When a query contains multiple distinct objectives or clauses spanning DIFFERENT capability domains (e.g. "Retrieve RFC standards [retrieval_qa] AND evaluate vulnerabilities [formal_reasoning] AND write Python code [coding]", or "Simulate thermodynamics [science_tech] AND solve differential matrices [coding] AND export parquet datasets [structured_data]"), you MUST decompose it into 2 to 4 distinct subtask nodes.
   - Each subtask must be assigned strictly to its matching specialist domain.
   - Specify DAG dependencies so downstream subtasks depend on upstream outputs.
   - NEVER collapse a multi-domain compound query into a single monolithic node.

2. Atomic Single-Domain Queries (STRICT PROHIBITION AGAINST INTRA-DOMAIN FRAGMENTATION):
   - If a query is focused within a SINGLE specialist domain, you MUST NOT fragment it into multiple subtasks, even if it contains multiple steps, clauses, or instructions (e.g. "derive X and prove Y and show steps", or "state conditions and identify fallacies and prove theorem", or "implement function with type annotations, complexity guarantees, and unit tests").
   - ALL steps belonging to the SAME domain must be kept together in exactly ONE root subtask ("node_1").
   - NEVER emit multiple nodes with the same capability domain (e.g. do NOT emit two "mathematics" nodes, two "coding" nodes, or two "formal_reasoning" nodes). Emitting duplicate domain nodes is strictly forbidden.
   - Do NOT invent unrequested deliverables (such as separate documentation, architecture summaries, or reporting). Unit tests, complexity guarantees, and type annotations are integral parts of the "coding" domain and MUST remain inside node_1.
   - Intra-domain work is handled completely by that domain's specialist model.

Specialist Domain Categories (Strict 8-Domain Pool):
1. coding: Python/C++ code, algorithms, data structures, scripts, debugging, unit tests, type annotations, docstrings.
2. mathematics: algebraic derivations, calculus, linear algebra, proofs, probability, numerical optimization.
3. formal_reasoning: logical verification, trade-off analysis, counterfactual reasoning, constraint validation.
4. retrieval_qa: factual knowledge, RFCs, API specifications, compliance standards, technical documentation.
5. science_tech: physics, mechanics, chemistry, thermodynamics, electronics, material properties.
6. structured_data: relational schemas, SQL, table definitions, ORM data models, JSON/Protobuf formats.
7. creative_synthesis: high-level architecture overviews, executive summaries, technical reports.
8. systems_ops: shell scripting, Docker/k8s, process management, POSIX, networking, concurrency.

FEW-SHOT EXAMPLES:

Example 1 (Compound Query - Multi-Domain):
User: "Retrieve security RFC standards, evaluate Linux socket vulnerabilities, and implement a sandboxed Python runtime for multi-disciplinary challenge #21."
Output:
{
  "subtasks": [
    {
      "id": "node_1",
      "text": "Retrieve official security RFC standards and Linux socket vulnerability specifications.",
      "capability": "retrieval_qa",
      "dependencies": []
    },
    {
      "id": "node_2",
      "text": "Evaluate Linux socket vulnerabilities, security controls, and kernel-level mitigations.",
      "capability": "formal_reasoning",
      "dependencies": ["node_1"]
    },
    {
      "id": "node_3",
      "text": "Implement a sandboxed Python runtime for secure socket communication.",
      "capability": "coding",
      "dependencies": ["node_1", "node_2"]
    }
  ]
}

Example 2 (Compound Query - Multi-Domain):
User: "Simulate physical thermodynamic diffusion, solve partial differential matrices, and export relational parquet datasets for multi-disciplinary challenge #41."
Output:
{
  "subtasks": [
    {
      "id": "node_1",
      "text": "Define physical thermodynamic diffusion principles, boundary conditions, and continuous governing equations.",
      "capability": "science_tech",
      "dependencies": []
    },
    {
      "id": "node_2",
      "text": "Construct discrete Laplacian partial differential matrices and implement numerical time-stepping in Python.",
      "capability": "coding",
      "dependencies": ["node_1"]
    },
    {
      "id": "node_3",
      "text": "Export simulation trajectory to relational Parquet dataset format with schema validation.",
      "capability": "structured_data",
      "dependencies": ["node_2"]
    }
  ]
}

Example 3 (Atomic Query - Single Domain Mathematics):
User: "Derive the closed-form analytical solution and prove convergence properties for mathematical formula #1, showing all intermediate algebraic steps."
Output:
{
  "subtasks": [
    {
      "id": "node_1",
      "text": "Derive the closed-form analytical solution and prove convergence properties for mathematical formula #1, showing all intermediate algebraic steps.",
      "capability": "mathematics",
      "dependencies": []
    }
  ]
}

Example 4 (Atomic Query - Single Domain Formal Reasoning):
User: "Perform formal deductive verification of logical problem #1, state validity conditions, identify fallacies, and construct symbolic proofs."
Output:
{
  "subtasks": [
    {
      "id": "node_1",
      "text": "Perform formal deductive verification of logical problem #1, state validity conditions, identify fallacies, and construct symbolic proofs.",
      "capability": "formal_reasoning",
      "dependencies": []
    }
  ]
}

Example 5 (Atomic Query - Single Domain Coding):
User: "Implement an advanced algorithmic module #1 in Python with full type annotations, O(1)/O(log N) complexity guarantees, and comprehensive edge-case unit tests."
Output:
{
  "subtasks": [
    {
      "id": "node_1",
      "text": "Implement an advanced algorithmic module #1 in Python with full type annotations, O(1)/O(log N) complexity guarantees, and comprehensive edge-case unit tests.",
      "capability": "coding",
      "dependencies": []
    }
  ]
}

Output strictly valid JSON with no markdown wrapping:
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

        # Architectural Guard against Intra-Domain Over-Fragmentation (Fix 1 from v2):
        # If all subtasks emitted belong to the exact same capability domain,
        # collapse them into a single coherent atomic task for that specialist.
        if dag.get("subtasks"):
            caps = [s.get("capability", "coding") for s in dag["subtasks"]]
            if len(set(caps)) == 1 and len(dag["subtasks"]) > 1:
                dag["subtasks"] = [
                    {
                        "id": "node_1",
                        "text": query_text,
                        "capability": caps[0],
                        "dependencies": [],
                        "depth": 0
                    }
                ]

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
        text = raw_text.strip()
        # Check for code blocks first
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
        else:
            # Find outermost curly braces
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidate = text[start:end+1].strip()
            else:
                candidate = text

        try:
            parsed = json.loads(candidate)
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

