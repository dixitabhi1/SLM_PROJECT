"""
Grounded Engineering Specialist Runner
AI Search Framework - Version 5
Wraps an underlying SLM runner with deterministic multi-domain engineering grounding.
Applies to mathematics, science_tech, structured_data, and retrieval_qa tasks.
"""

from typing import Dict, Any, Optional, List
from ...models.base import BaseModelRunner, ModelResponse
from ..tools.engineering_retrieval_tool import EngineeringRetrievalTool

class GroundedEngineeringModelRunner(BaseModelRunner):
    def __init__(
        self,
        base_runner: BaseModelRunner,
        retrieval_tool: Optional[EngineeringRetrievalTool] = None,
        top_k: int = 2
    ):
        super().__init__(
            model_name=f"{base_runner.model_name}-grounded",
            revision=base_runner.revision,
            max_tokens=base_runner.max_tokens,
            temperature=base_runner.temperature
        )
        self.base_runner = base_runner
        self.retrieval_tool = retrieval_tool or EngineeringRetrievalTool()
        self.top_k = top_k
        self.api_model_name = getattr(base_runner, "api_model_name", base_runner.model_name)

    async def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> ModelResponse:
        # 1. Deterministic retrieval against pinned reference corpus
        retrieved_docs = self.retrieval_tool.query(prompt, top_k=self.top_k)
        grounding_context = self.retrieval_tool.format_grounded_context(retrieved_docs)

        # 2. Inject grounding context into prompt
        augmented_prompt = (
            f"{grounding_context}\n\n"
            f"=== ASSIGNED SPECIALIST SUBTASK ===\n"
            f"{prompt}\n\n"
            f"Ground your technical derivation, analysis, or code in the verified reference equations and standards above."
        )
        
        # 3. Call underlying SLM
        resp = await self.base_runner.generate(
            prompt=augmented_prompt,
            system_prompt=system_prompt,
            **kwargs
        )

        # 4. Attach citation traceability metadata
        cited_ids = [d.get("id") for d in retrieved_docs]
        meta = dict(resp.metadata) if resp.metadata else {}
        meta["retrieval_grounding_active"] = True
        meta["retrieved_sources"] = cited_ids
        meta["grounded_context_injected"] = True

        return ModelResponse(
            text=resp.text,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            total_tokens=resp.total_tokens,
            latency_ms=resp.latency_ms,
            model_name=self.model_name,
            model_revision=self.revision,
            metadata=meta
        )

