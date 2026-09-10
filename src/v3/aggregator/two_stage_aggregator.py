"""
Two-Stage Aggregator (v3 Architecture - 8 Domains)
Synthesizes multi-specialist outputs into an authoritative, structurally complete,
and contextually rich final response matching frontier-model completeness standards.
"""

import time
from typing import Dict, List, Any, Optional
from ...models.base import BaseModelRunner, ModelResponse
from ...instrumentation.logger import ExperimentLogger

AGGREGATOR_V3_SYSTEM_PROMPT = """You are an expert Chief Synthesizer SLM in an advanced 8-specialist search architecture.
Your task is to synthesize outputs from specialized domain models (coding, mathematics, formal reasoning, retrieval, science, structured data, systems, etc.) into an authoritative, comprehensive, and seamless technical document.

Synthesis Standards:
1. Unified Technical Narrative:
   - Harmonize the findings into one cohesive, professional authoritative voice.
   - Eliminate disjointed transitions, fragmented bullet-points, or conflicting notation across subtask outputs.

2. Complete Technical Preservation:
   - You MUST retain all verified code implementations, mathematical derivations, database schemas, and equations.
   - Never replace code with placeholders, stubs, or ellipses.

3. Contextual Depth & Structural Completeness:
   - Provide comprehensive architectural context, conceptual motivation, and step-by-step walkthroughs.
   - Include operational considerations, edge-case constraints, error handling, and production trade-offs where applicable.
   - Ensure the final synthesis matches the exhaustive depth expected from frontier technical reference systems.

4. Rigorous Organization:
   - Organize logically using clear Markdown section headings (##, ###).
   - Begin with an executive problem framing, followed by the domain-specific technical deep-dives, and conclude with integration/verification guidance.
"""

class TwoStageAggregator_v3:
    def __init__(
        self,
        global_aggregator_runner: BaseModelRunner,
        logger: Optional[ExperimentLogger] = None
    ):
        self.global_runner = global_aggregator_runner
        self.logger = logger

    async def aggregate_global(
        self,
        original_query: str,
        subtask_results: Dict[str, str],
        run_record: Dict[str, Any]
    ) -> str:
        """
        Terminal aggregation across all completed subtasks.
        Bypasses aggregation (zero latency) if there is only 1 subtask.
        """
        if len(subtask_results) == 1:
            single_output = next(iter(subtask_results.values()))
            if self.logger:
                now_t = time.perf_counter()
                self.logger.record_stage(
                    record=run_record,
                    stage_name="global_aggregator_passthrough",
                    model_name="passthrough_bypass",
                    model_revision="main",
                    start_time_s=now_t,
                    end_time_s=now_t,
                    prompt_tokens=0,
                    completion_tokens=0,
                    input_data={"original_query": original_query, "subtask_results": subtask_results},
                    output_data=single_output,
                    extra_metadata={"node_count": 1, "bypassed": True, "reason": "single_subtask_direct_passthrough"}
                )
            return single_output

        context_blocks = []
        for node_id, output_text in subtask_results.items():
            context_blocks.append(f"### Subtask Result [{node_id}]:\n{output_text.strip()}")

        context_str = "\n\n".join(context_blocks)
        prompt = (
            f"Original User Query:\n{original_query}\n\n"
            f"Specialist Subtask Outputs to Synthesize:\n{context_str}\n\n"
            f"Produce the authoritative, comprehensive synthesized response (preserving all code, schemas, and mathematical details):"
        )

        start_t = time.perf_counter()
        resp: ModelResponse = await self.global_runner.generate(
            prompt=prompt,
            system_prompt=AGGREGATOR_V3_SYSTEM_PROMPT,
            temperature=0.0
        )
        end_t = time.perf_counter()

        if self.logger:
            self.logger.record_stage(
                record=run_record,
                stage_name="global_aggregator_synthesis",
                model_name=resp.model_name,
                model_revision=resp.model_revision,
                start_time_s=start_t,
                end_time_s=end_t,
                prompt_tokens=resp.prompt_tokens,
                completion_tokens=resp.completion_tokens,
                input_data={"original_query": original_query, "subtask_nodes": list(subtask_results.keys())},
                output_data=resp.text,
                extra_metadata={"node_count": len(subtask_results), "bypassed": False}
            )

        return resp.text
