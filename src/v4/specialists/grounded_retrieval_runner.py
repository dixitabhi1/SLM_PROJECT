"""
Grounded Retrieval Specialist Runner
AI Search Framework - Version 4 Step 2
Wraps an underlying SLM runner with deterministic reference corpus grounding.
Strictly restricted to retrieval_qa specialist tasks.
"""

from typing import Dict, Any, Optional, List
from ...models.base import BaseModelRunner, ModelResponse
from ..tools.retrieval_tool import DeterministicRetrievalTool

class GroundedRetrievalModelRunner(BaseModelRunner):
    def __init__(
        self,
        base_runner: BaseModelRunner,
        retrieval_tool: Optional[DeterministicRetrievalTool] = None,
        top_k: int = 4
    ):
        super().__init__(
            model_name=f"{base_runner.model_name}-grounded",
            revision=base_runner.revision,
            max_tokens=base_runner.max_tokens,
            temperature=base_runner.temperature
        )
        self.base_runner = base_runner
        self.retrieval_tool = retrieval_tool or DeterministicRetrievalTool()
        self.top_k = top_k
        self.api_model_name = getattr(base_runner, "api_model_name", base_runner.model_name)

    async def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> ModelResponse:
        # 1. Deterministic retrieval against pinned reference corpus
        retrieved_docs = self.retrieval_tool.query(prompt, top_k=self.top_k)
        grounding_context = self.retrieval_tool.format_grounded_context(retrieved_docs)

        # 2. Inject grounding context into prompt
        augmented_prompt = f"{grounding_context}\n\nUser Task:\n{prompt}"
        
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
        meta["retrieval_mode"] = "deterministic_corpus_bm25"

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

