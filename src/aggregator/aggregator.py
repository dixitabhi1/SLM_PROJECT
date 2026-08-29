"""
Aggregator SLM Engine
AI Search Framework
Fuses subtask outputs into a coherent final response, explicitly resolving contradictions (TRD §2).
"""

import time
from typing import Dict, Any, Optional
from ..models.base import BaseModelRunner, ModelResponse
from ..instrumentation.logger import ExperimentLogger

AGGREGATOR_SYSTEM_PROMPT = """You are an expert Aggregator SLM.
Your task is to synthesize multiple subtask outputs into a unified, rigorous, and cohesive final answer to the user's original query.

Instructions:
1. Synthesize and integrate all subtask results into a unified narrative. Do NOT merely concatenate or repeat headers.
2. Explicitly identify and resolve any contradictions or discrepancies between intermediate subtask findings.
3. Ensure mathematical derivations, code blocks, and logical deductions are complete, cleanly formatted, and aligned.
4. Provide a direct, authoritative, and exhaustive response.
"""

class ResponseAggregator:
    def __init__(self, model_runner: BaseModelRunner, logger: Optional[ExperimentLogger] = None):
        self.runner = model_runner
        self.logger = logger

    async def aggregate(
        self,
        original_query: str,
        subtask_results: Dict[str, str],
        run_record: Dict[str, Any]
    ) -> str:
        """
        Synthesizes subtask outputs into the final response with full stage logging.
        """
        # Format intermediate context
        context_blocks = []
        for node_id, output_text in subtask_results.items():
            context_blocks.append(f"### Output for Subtask ({node_id}):\n{output_text.strip()}")
        
        aggregated_context = "\n\n".join(context_blocks)
        prompt = (
            f"Original User Query:\n{original_query}\n\n"
            f"Subtask Intermediate Findings:\n{aggregated_context}\n\n"
            f"Synthesize the final authoritative answer:"
        )

        start_t = time.perf_counter()
        resp: ModelResponse = await self.runner.generate(
            prompt=prompt,
            system_prompt=AGGREGATOR_SYSTEM_PROMPT,
            temperature=0.0
        )
        end_t = time.perf_counter()

        if self.logger:
            self.logger.record_stage(
                record=run_record,
                stage_name="aggregator_synthesis",
                model_name=self.runner.model_name,
                model_revision=self.runner.revision,
                start_time_s=start_t,
                end_time_s=end_t,
                prompt_tokens=resp.prompt_tokens,
                completion_tokens=resp.completion_tokens,
                input_data={"original_query": original_query, "subtask_results": subtask_results},
                output_data=resp.text,
                extra_metadata={"node_count": len(subtask_results)}
            )

        return resp.text

