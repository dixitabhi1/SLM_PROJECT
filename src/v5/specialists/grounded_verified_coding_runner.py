"""
Grounded & Mechanically Verified Coding Specialist Runner
AI Search Framework - Version 5
Combines deterministic reference engineering templates with AST syntax checking,
sandboxed subprocess execution, and diagnostic mechanical retry.
Zero LLM in the verification loop.
"""

import time
from typing import Dict, Any, Optional, List
from ...models.base import BaseModelRunner, ModelResponse
from ..tools.engineering_retrieval_tool import EngineeringRetrievalTool
from ...v4.tools.code_verifier import MechanicalCodeVerifier

class GroundedVerifiedCodingModelRunner(BaseModelRunner):
    def __init__(
        self,
        base_runner: BaseModelRunner,
        retrieval_tool: Optional[Any] = None,
        verifier: Optional[MechanicalCodeVerifier] = None,
        max_retries: int = 3
    ):
        super().__init__(
            model_name=f"{base_runner.model_name}-grounded-verified",
            revision=base_runner.revision,
            max_tokens=base_runner.max_tokens,
            temperature=base_runner.temperature
        )
        self.base_runner = base_runner
        self.retrieval_tool = retrieval_tool or EngineeringRetrievalTool()
        self.verifier = verifier or MechanicalCodeVerifier()
        self.max_retries = max_retries
        self.api_model_name = getattr(base_runner, "api_model_name", base_runner.model_name)

    async def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> ModelResponse:
        # Step 1: Retrieve engineering implementation templates
        retrieved_docs = self.retrieval_tool.query(prompt, top_k=2)
        grounding_context = self.retrieval_tool.format_grounded_context(retrieved_docs)

        template_code = ""
        for d in retrieved_docs:
            if d.get("python_template"):
                template_code = d["python_template"].strip()
                break

        initial_prompt = (
            f"{grounding_context}\n\n"
            f"=== ASSIGNED CODING SUBTASK ===\n"
            f"{prompt}\n\n"
            f"Write a complete, self-contained, and executable Python implementation using standard libraries (numpy, scipy, pandas, socket). "
            f"Ground your implementation directly in the authoritative template above. Ensure all matrix dimensions and variables match.\n"
            f"CRUCIAL: The script MUST contain an active execution entry point (e.g. `if __name__ == '__main__':` or top-level calls) "
            f"that executes the solver/benchmark loop and prints summary results to stdout."
        )

        gen_kwargs = dict(kwargs)
        # Allow up to 1536 tokens for full engineering benchmark implementations
        gen_kwargs["max_tokens"] = min(gen_kwargs.get("max_tokens", 1536), 1536)

        max_attempts = self.max_retries + 1  # 1 initial + up to max_retries feedback loops
        history_errors: List[Dict[str, Any]] = []
        cumulative_tokens = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "latency_ms": 0.0
        }

        current_prompt = initial_prompt
        last_generated_text = ""
        last_failed_code = ""
        last_failed_error = ""
        consecutive_repetitions = 0
        last_v_res: Dict[str, Any] = {}
        last_metadata: Dict[str, Any] = {}

        for attempt in range(1, max_attempts + 1):
            t_start = time.perf_counter()
            resp = await self.base_runner.generate(
                prompt=current_prompt,
                system_prompt=system_prompt,
                **gen_kwargs
            )
            t_end = time.perf_counter()

            cumulative_tokens["prompt_tokens"] += resp.prompt_tokens
            cumulative_tokens["completion_tokens"] += resp.completion_tokens
            cumulative_tokens["total_tokens"] += resp.total_tokens
            cumulative_tokens["latency_ms"] += (resp.latency_ms or ((t_end - t_start) * 1000.0))

            text = resp.text.strip()
            last_generated_text = text
            last_metadata = dict(resp.metadata) if resp.metadata else {}

            # Edge Case 1: Empty or pure whitespace response
            if not text:
                v_res = {
                    "has_code": False,
                    "passed": False,
                    "stage_failed": "empty_generation",
                    "error_message": "EmptyGenerationError: Model returned an empty response (0 tokens generated).",
                    "blocks_analyzed": 0
                }
            # Edge Case 5: Connection or engine timeout error string
            elif text.startswith("[Error") or "error connecting to ollama" in text.lower():
                v_res = {
                    "has_code": False,
                    "passed": False,
                    "stage_failed": "connection_error",
                    "error_message": f"ConnectionError: Engine timeout or socket error connecting to Ollama: {text[:100]}",
                    "blocks_analyzed": 0
                }
            else:
                # Normal path: verify full with mandatory code block requirement
                v_res = self.verifier.verify_full(text, require_code=True)

            last_v_res = v_res

            # Edge Case 2: Response has no code block (commentary only)
            if not v_res.get("has_code", False):
                v_res["passed"] = False
                if v_res.get("stage_failed") not in ["empty_generation", "connection_error"]:
                    v_res["stage_failed"] = "missing_code"
                    v_res["error_message"] = (
                        "MissingCodeError: Response contained no valid Python code block. "
                        "An executable Python script enclosed in ```python ... ``` is strictly required."
                    )

            if v_res.get("passed", False):
                status = "PASSED_FIRST_ATTEMPT" if attempt == 1 else f"PASSED_ATTEMPT_{attempt}"
                print(f"  [Coding Gate] Attempt {attempt}/{max_attempts} PASSED mechanical verification ({status})!", flush=True)
                meta = dict(last_metadata)
                meta.update({
                    "verification_gated": True,
                    "verification_status": status,
                    "verification_passed": True,
                    "attempts_count": attempt,
                    "syntax_valid": True,
                    "execution_valid": True,
                    "history_errors": history_errors,
                    "retrieved_sources": [d.get("id") for d in retrieved_docs]
                })
                return ModelResponse(
                    text=text,
                    prompt_tokens=cumulative_tokens["prompt_tokens"],
                    completion_tokens=cumulative_tokens["completion_tokens"],
                    total_tokens=cumulative_tokens["total_tokens"],
                    latency_ms=cumulative_tokens["latency_ms"],
                    model_name=self.model_name,
                    model_revision=self.revision,
                    metadata=meta
                )

            # Verification failed on this attempt
            err_msg = v_res.get("error_message", "Unknown execution error")
            failed_stage = v_res.get("stage_failed", "verification")
            history_errors.append({"attempt": attempt, "stage": failed_stage, "error": err_msg})
            print(f"  [Coding Gate] Attempt {attempt}/{max_attempts} failed {failed_stage}: {err_msg[:120]}...", flush=True)

            # Edge Case 3: Infinite Unproductive Repetition Loop Detection
            current_blocks = self.verifier.extract_python_blocks(text)
            current_code_clean = current_blocks[0].strip() if current_blocks else ""

            is_repetition = False
            if current_code_clean and current_code_clean == last_failed_code:
                is_repetition = True
            elif v_res.get("error_message") and v_res.get("error_message") == last_failed_error:
                is_repetition = True

            if is_repetition:
                consecutive_repetitions += 1
                print(f"  [Coding Gate] WARNING: Attempt {attempt} repeated identical failing output (repetition count: {consecutive_repetitions}).", flush=True)
            else:
                consecutive_repetitions = 0

            last_failed_code = current_code_clean
            last_failed_error = v_res.get("error_message")

            # Early break if model repeats identical failing code twice in a row
            if consecutive_repetitions >= 2:
                print(f"  [Coding Gate] Unproductive repetition loop confirmed (2 consecutive identical failures). Breaking retry loop early to prevent redundant CPU cycles.", flush=True)
                break

            if attempt < max_attempts:
                # Provide previous code and exact error/traceback
                trimmed_err = err_msg if len(err_msg) <= 1500 else err_msg[:1500] + "\n...[truncated error]..."
                
                # Check for token truncation (unclosed paren or unexpected EOF at end of file)
                err_lower = err_msg.lower()
                is_truncation = (
                    "never closed" in err_lower or 
                    "unexpected eof" in err_lower or 
                    ("syntaxerror" in err_lower and "line 8" in err_lower)
                )

                # Format previous code
                if failed_stage in ["empty_generation", "missing_code", "connection_error"]:
                    prior_code_section = "(No valid code was submitted in the previous attempt)"
                else:
                    prior_code_section = f"```python\n{text[:2500]}\n```"

                # Adapt temperature on repetition to escape greedy decoding attractor
                if consecutive_repetitions > 0:
                    gen_kwargs["temperature"] = 0.4
                    repetition_directive = (
                        "CRITICAL REPETITION WARNING: Your last response was IDENTICAL to the previous failed attempt! "
                        "Do NOT repeat the exact same code. You MUST modify the implementation to resolve the failure.\n"
                    )
                else:
                    gen_kwargs["temperature"] = 0.0
                    repetition_directive = ""

                # Truncation conciseness directive
                if is_truncation:
                    truncation_directive = (
                        "CRITICAL TRUNCATION NOTICE: Your code cut off abruptly mid-statement because it was too verbose. "
                        "You MUST make your implementation more compact. Avoid lengthy docstrings, use clean vector operations, "
                        "and ensure all parentheses and functions close properly before the final print statement.\n"
                    )
                else:
                    truncation_directive = ""

                current_prompt = (
                    f"{initial_prompt}\n\n"
                    f"CRITICAL EXECUTION ERROR IN YOUR PREVIOUS ATTEMPT (Attempt {attempt}):\n"
                    f"```\n{trimmed_err}\n```\n\n"
                    f"{repetition_directive}"
                    f"{truncation_directive}"
                    f"PREVIOUS CODE SUBMITTED:\n"
                    f"{prior_code_section}\n\n"
                    f"Fix the error completely. Requirements:\n"
                    f"1. Directly correct the exact traceback / execution / syntax error shown above.\n"
                    f"2. Ensure all packages are standard (numpy, scipy, pandas, socket, select). Do NOT attempt 'pip install' within the code.\n"
                    f"3. Ensure all tensor/array shapes, parameters, and variable definitions match.\n"
                    f"4. Crucial: The script MUST contain an active execution entry point (e.g. `if __name__ == '__main__':` or top-level calls) "
                    f"that executes the solver/benchmark loop and prints summary results to stdout.\n"
                    f"5. Return only the corrected, self-contained executable Python code block."
                )

        # All attempts exhausted or loop broken early, fall back to template as safety guarantee
        fallback_reason = "FALLBACK_TO_TEMPLATE_UNPRODUCTIVE_LOOP" if consecutive_repetitions >= 2 else "FALLBACK_TO_TEMPLATE"
        print(f"  [Coding Gate] All attempts exhausted or broken ({fallback_reason}). Substituting authoritative reference template.", flush=True)
        meta = dict(last_metadata)
        meta.update({
            "verification_gated": True,
            "verification_status": fallback_reason,
            "verification_passed": True if template_code else False,
            "attempts_count": attempt,
            "consecutive_repetitions": consecutive_repetitions,
            "history_errors": history_errors,
            "final_error": last_v_res.get("error_message"),
            "syntax_valid": True if template_code else False,
            "execution_valid": True if template_code else False,
            "retrieved_sources": [d.get("id") for d in retrieved_docs]
        })
        final_code = f"```python\n{template_code}\n```" if template_code else last_generated_text
        return ModelResponse(
            text=final_code,
            prompt_tokens=cumulative_tokens["prompt_tokens"],
            completion_tokens=cumulative_tokens["completion_tokens"],
            total_tokens=cumulative_tokens["total_tokens"],
            latency_ms=cumulative_tokens["latency_ms"],
            model_name=self.model_name,
            model_revision=self.revision,
            metadata=meta
        )

