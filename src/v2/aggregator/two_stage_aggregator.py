"""
Two-Stage Aggregator (v2 Architecture)
Stage 1: Local Collaboration Synthesis for multi-agent teams at depth limit.
Stage 2: Global Terminal Aggregator SLM (<= 8B) for cross-task fusion & contradiction resolution.
"""

import time
from typing import Dict, List, Any, Optional
from ...models.base import BaseModelRunner, ModelResponse
from ...instrumentation.logger import ExperimentLogger

GLOBAL_AGGREGATOR_V2_PROMPT = """You are an expert Aggregator SLM in an advanced multi-model search architecture.
Your task is to synthesize multiple subtask outputs into an authoritative, complete, and seamlessly unified final response.

Guidelines:
1. Synthesize all findings into a unified narrative. Do NOT just concatenate subtask blocks.
2. Explicitly detect and resolve any conflicting assertions or mathematical discrepancies across subtask outputs.
3. Preserve all verified code snippets, mathematical proofs, and technical citations.
4. Deliver a thorough, well-structured, and complete response to the user's original query.
"""

class TwoStageAggregator:
    def __init__(self, global_aggregator_runner: BaseModelRunner, logger: Optional[ExperimentLogger] = None):
        self.global_runner = global_aggregator_runner
        self.logger = logger

    async def aggregate_global(
        self,
        original_query: str,
        subtask_results: Dict[str, str],
        run_record: Dict[str, Any]
    ) -> str:
        """
        Global terminal fusion across all completed subtasks.
        """
        context_blocks = []
        for node_id, output_text in subtask_results.items():
            context_blocks.append(f"### Subtask Result [{node_id}]:\n{output_text.strip()}")

        context_str = "\n\n".join(context_blocks)
        prompt = (
            f"Original User Query:\n{original_query}\n\n"
            f"Completed Subtask Outputs:\n{context_str}\n\n"
            f"Synthesize the authoritative final answer:"
        )

        start_t = time.perf_counter()
        resp: ModelResponse = await self.global_runner.generate(
            prompt=prompt,
            system_prompt=GLOBAL_AGGREGATOR_V2_PROMPT,
            temperature=0.0
        )
        end_t = time.perf_counter()

        if self.logger:
            self.logger.record_stage(
                record=run_record,
                stage_name="global_aggregator_synthesis",
                model_name=self.global_runner.model_name,
                model_revision=self.global_runner.revision,
                start_time_s=start_t,
                end_time_s=end_t,
                prompt_tokens=resp.prompt_tokens,
                completion_tokens=resp.completion_tokens,
                input_data={"original_query": original_query, "subtask_results": subtask_results},
                output_data=resp.text,
                extra_metadata={"node_count": len(subtask_results), "stage": "global_terminal_aggregation"}
            )

        return resp.text

