"""
Two-Stage Aggregator with Stylistic Harmonization Pass (v2 Architecture)
Stage 1: Local Collaboration Synthesis for multi-agent teams at depth limit.
Stage 2: Global Terminal Fusion with Contradiction Resolution & Stylistic Voice Harmonization.
"""

import time
from typing import Dict, List, Any, Optional
from ...models.base import BaseModelRunner, ModelResponse
from ...instrumentation.logger import ExperimentLogger

GLOBAL_AGGREGATOR_HARMONIZATION_PROMPT = """You are an expert Aggregator SLM in an advanced multi-model search architecture.
Your task is to synthesize multiple subtask outputs into an authoritative, complete, and seamlessly unified final response.

Guidelines:
1. Unified Voice & Stylistic Harmonization:
   - Rewrite the collective findings into one single, cohesive authoritative voice.
   - Eliminate blocky transitions, disjointed bullet-point pasting, or abrupt register shifts between subtasks.
   - Ensure a continuous, polished technical narrative.

2. Contradiction Resolution & Accuracy:
   - Detect and resolve any conflicting assertions, variable notation differences, or numerical discrepancies across subtask outputs.
   - Ground all claims in the verified subtask results.

3. Complete Content Preservation (NO Content Truncation):
   - You MUST retain all verified code blocks, algorithmic implementations, mathematical derivations, equations, and technical specifics from every subtask.
   - Do NOT drop, summarize away, or omit technical details to achieve brevity.

4. Organization:
   - Present a logical structure with clear section headers, rigorous explanations, and concrete implementation artifacts.
"""

class TwoStageAggregator:
    def __init__(
        self,
        global_aggregator_runner: BaseModelRunner,
        logger: Optional[ExperimentLogger] = None,
        enable_stylistic_harmonization: bool = True
    ):
        self.global_runner = global_aggregator_runner
        self.logger = logger
        self.enable_stylistic_harmonization = enable_stylistic_harmonization

    async def aggregate_global(
        self,
        original_query: str,
        subtask_results: Dict[str, str],
        run_record: Dict[str, Any]
    ) -> str:
        """
        Global terminal fusion across all completed subtasks with stylistic harmonization.
        """
        context_blocks = []
        for node_id, output_text in subtask_results.items():
            context_blocks.append(f"### Subtask Result [{node_id}]:\n{output_text.strip()}")

        context_str = "\n\n".join(context_blocks)
        prompt = (
            f"Original User Query:\n{original_query}\n\n"
            f"Completed Subtask Outputs:\n{context_str}\n\n"
            f"Synthesize the authoritative, stylistically harmonized final answer (preserving all code, math, and data):"
        )

        start_t = time.perf_counter()
        resp: ModelResponse = await self.global_runner.generate(
            prompt=prompt,
            system_prompt=GLOBAL_AGGREGATOR_HARMONIZATION_PROMPT,
            temperature=0.0
        )
        end_t = time.perf_counter()

        if self.logger:
            self.logger.record_stage(
                record=run_record,
                stage_name="global_aggregator_harmonized_synthesis",
                model_name=self.global_runner.model_name,
                model_revision=self.global_runner.revision,
                start_time_s=start_t,
                end_time_s=end_t,
                prompt_tokens=resp.prompt_tokens,
                completion_tokens=resp.completion_tokens,
                input_data={"original_query": original_query, "subtask_results": subtask_results},
                output_data=resp.text,
                extra_metadata={
                    "node_count": len(subtask_results),
                    "stage": "global_terminal_aggregation",
                    "stylistic_harmonization_enabled": self.enable_stylistic_harmonization
                }
            )

        return resp.text
