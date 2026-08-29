"""
Mock / Simulated Model Runner
AI Search Framework
Provides deterministic simulated responses with configurable token counts and latency distributions.
"""

import asyncio
import hashlib
import json
import time
from typing import Optional, Dict, Any
from .base import BaseModelRunner, ModelResponse

class MockModelRunner(BaseModelRunner):
    def __init__(
        self,
        model_name: str,
        revision: str = "main",
        base_latency_ms: float = 120.0,
        tokens_per_sec: float = 40.0,
        fixed_seed: int = 42,
        **kwargs
    ):
        super().__init__(model_name, revision, **kwargs)
        self.base_latency_ms = base_latency_ms
        self.tokens_per_sec = tokens_per_sec
        self.fixed_seed = fixed_seed

    async def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> ModelResponse:
        start_time = time.perf_counter()
        
        # Deterministic pseudo-random generation based on prompt hash + seed
        prompt_hash = hashlib.sha256(f"{self.fixed_seed}_{prompt}".encode()).hexdigest()
        hash_int = int(prompt_hash[:8], 16)
        
        prompt_tokens = max(10, len(prompt.split()) * 4 // 3)
        
        # Check if prompt is asking for JSON DAG decomposition
        if "JSON task graph" in prompt or "subtasks" in prompt:
            # Deterministic DAG output
            completion_tokens = 180 + (hash_int % 120)
            text = self._mock_decomposition_json(prompt, hash_int)
        else:
            completion_tokens = 250 + (hash_int % 350)
            text = f"[{self.model_name}] Comprehensive structured response addressing the prompt. Analysis derived with verified step-by-step reasoning and domain specifics for hash {prompt_hash[:6]}."

        simulated_duration = (self.base_latency_ms / 1000.0) + (completion_tokens / max(1.0, self.tokens_per_sec * 10.0))
        # Small async sleep to emulate real event loop scheduling
        await asyncio.sleep(min(0.05, simulated_duration / 50.0))
        
        end_time = time.perf_counter()
        actual_latency_ms = (end_time - start_time) * 1000.0

        return ModelResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=actual_latency_ms,
            model_name=self.model_name,
            model_revision=self.revision,
            metadata={"simulated_ideal_latency_ms": simulated_duration * 1000.0, "seed": self.fixed_seed}
        )

    def _mock_decomposition_json(self, prompt: str, hash_int: int) -> str:
        # Determine number of nodes from prompt or hash
        if "three_plus_domain" in prompt or "CD_" in prompt:
            dag = {
                "subtasks": [
                    {"id": "node_1", "text": "Extract and formalize mathematical specifications and constraints.", "capability": "math", "dependencies": []},
                    {"id": "node_2", "text": "Analyze logical consistency and architectural security requirements.", "capability": "reasoning", "dependencies": ["node_1"]},
                    {"id": "node_3", "text": "Implement vectorized Python software verification module.", "capability": "coding", "dependencies": ["node_1", "node_2"]}
                ]
            }
        elif "two_domain" in prompt or "TD_" in prompt:
            dag = {
                "subtasks": [
                    {"id": "node_1", "text": "Derive quantitative formulation and theoretical equations.", "capability": "math", "dependencies": []},
                    {"id": "node_2", "text": "Implement efficient Python computational algorithm.", "capability": "coding", "dependencies": ["node_1"]}
                ]
            }
        else:
            dag = {
                "subtasks": [
                    {"id": "node_1", "text": "Execute domain-specific analysis and implementation directly.", "capability": "general", "dependencies": []}
                ]
            }
        return json.dumps(dag, indent=2)

