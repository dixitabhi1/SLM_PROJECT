"""
Decomposition SLM Engine
AI Search Framework
Calls the <=3B SLM, parses and validates DAG JSON, and enforces schema validity.
"""

import json
import re
from typing import Dict, List, Any, Optional
from ..models.base import BaseModelRunner, ModelResponse
from .prompt import DECOMPOSER_SYSTEM_PROMPT, format_decomposer_prompt

class TaskGraphDecomposer:
    def __init__(self, model_runner: BaseModelRunner):
        self.runner = model_runner

    async def decompose(self, query_text: str) -> Dict[str, Any]:
        prompt = format_decomposer_prompt(query_text)
        response = await self.runner.generate(
            prompt=prompt,
            system_prompt=DECOMPOSER_SYSTEM_PROMPT,
            temperature=0.0
        )
        
        parsed_dag, is_valid, error_msg = self._parse_and_validate_dag(response.text, query_text)
        
        return {
            "dag": parsed_dag,
            "is_schema_valid": is_valid,
            "parse_error": error_msg,
            "raw_response": response.text,
            "model_response": response
        }

    def _parse_and_validate_dag(self, raw_text: str, fallback_query: str) -> tuple[Dict[str, Any], bool, Optional[str]]:
        # Strip markdown fences if present
        cleaned = re.sub(r"^```json\s*", "", raw_text.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"^```\s*", "", cleaned)
        cleaned = re.sub(r"```$", "", cleaned.strip())

        try:
            data = json.loads(cleaned)
            if not isinstance(data, dict) or "subtasks" not in data or not isinstance(data["subtasks"], list):
                raise ValueError("JSON missing 'subtasks' array root.")
            
            subtasks = data["subtasks"]
            if len(subtasks) == 0:
                raise ValueError("Empty subtasks list.")

            node_ids = set()
            for idx, task in enumerate(subtasks):
                if not isinstance(task, dict):
                    raise ValueError(f"Subtask #{idx} is not a valid object.")
                tid = task.get("id", f"node_{idx+1}")
                task["id"] = tid
                node_ids.add(tid)
                if "text" not in task or not task["text"].strip():
                    task["text"] = fallback_query
                if "capability" not in task or task["capability"] not in ["coding", "math", "reasoning", "retrieval", "general"]:
                    task["capability"] = "general"
                if "dependencies" not in task or not isinstance(task["dependencies"], list):
                    task["dependencies"] = []

            # Validate DAG acyclicity
            for task in subtasks:
                # filter out self-dependencies or non-existent nodes
                task["dependencies"] = [d for d in task["dependencies"] if d in node_ids and d != task["id"]]

            if self._has_cycles(subtasks):
                # Break cycles by dropping dependencies
                for task in subtasks:
                    task["dependencies"] = []

            return {"subtasks": subtasks}, True, None

        except Exception as e:
            # Fallback single-node graph
            fallback_dag = {
                "subtasks": [
                    {
                        "id": "node_1",
                        "text": fallback_query,
                        "capability": "general",
                        "dependencies": []
                    }
                ]
            }
            return fallback_dag, False, str(e)

    def _has_cycles(self, subtasks: List[Dict[str, Any]]) -> bool:
        graph = {t["id"]: t["dependencies"] for t in subtasks}
        visited = set()
        rec_stack = set()

        def dfs(node):
            visited.add(node)
            rec_stack.add(node)
            for dep in graph.get(node, []):
                if dep not in visited:
                    if dfs(dep):
                        return True
                elif dep in rec_stack:
                    return True
            rec_stack.remove(node)
            return False

        for node in graph:
            if node not in visited:
                if dfs(node):
                    return True
        return False

# Type helper
Tuple_Decomposition = Dict[str, Any]
