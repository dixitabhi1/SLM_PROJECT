"""
Mechanically Verified Coding Specialist Runner
AI Search Framework - Version 4 Step 3
Equips coding specialist with deterministic AST syntax checking, sandboxed execution,
and a single mechanical feedback self-correction loop.
Zero LLM in the verification loop.
"""

from typing import Dict, Any, Optional
from ...models.base import BaseModelRunner, ModelResponse
from ..tools.code_verifier import MechanicalCodeVerifier

class VerifiedCodingModelRunner(BaseModelRunner):
    def __init__(
        self,
        base_runner: BaseModelRunner,
        verifier: Optional[MechanicalCodeVerifier] = None,
        max_retries: int = 1
    ):
        super().__init__(
            model_name=f"{base_runner.model_name}-verified",
            revision=base_runner.revision,
            max_tokens=base_runner.max_tokens,
            temperature=base_runner.temperature
        )
        self.base_runner = base_runner
        self.verifier = verifier or MechanicalCodeVerifier()
        self.max_retries = max_retries
        self.api_model_name = getattr(base_runner, "api_model_name", base_runner.model_name)

    async def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> ModelResponse:
        # Step 1: Initial Generation
        resp1 = await self.base_runner.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            **kwargs
        )

        v_res1 = self.verifier.verify_full(resp1.text)
        if v_res1.get("passed", False):
            meta = dict(resp1.metadata) if resp1.metadata else {}
            meta.update({
                "verification_gated": True,
                "verification_status": "PASSED_FIRST_ATTEMPT",
                "verification_passed": True,
                "attempts_count": 1,
                "syntax_valid": True,
                "execution_valid": True
            })
            return ModelResponse(
                text=resp1.text,
                prompt_tokens=resp1.prompt_tokens,
                completion_tokens=resp1.completion_tokens,
                total_tokens=resp1.total_tokens,
                latency_ms=resp1.latency_ms,
                model_name=self.model_name,
                model_revision=self.revision,
                metadata=meta
            )

        # Step 2: Mechanical Self-Correction Retry (Exactly 1 retry with mechanical trace)
        err_msg = v_res1.get("error_message", "Unknown execution error")
        failed_stage = v_res1.get("stage_failed", "verification")
        print(f"  [Verification Gate] Initial code failed {failed_stage}: {err_msg[:120]}... Triggering mechanical retry.")

        retry_prompt = (
            f"{prompt}\n\n"
            f"CRITICAL FIX REQUIRED: Your previous code implementation failed automated mechanical {failed_stage} verification:\n"
            f"```\n{err_msg}\n```\n"
            f"Please rewrite the Python code to resolve this exact error. Ensure code syntax is 100% valid and self-contained."
        )

        resp2 = await self.base_runner.generate(
            prompt=retry_prompt,
            system_prompt=system_prompt,
            **kwargs
        )

        v_res2 = self.verifier.verify_full(resp2.text)
        combined_tokens = {
            "prompt_tokens": resp1.prompt_tokens + resp2.prompt_tokens,
            "completion_tokens": resp1.completion_tokens + resp2.completion_tokens,
            "total_tokens": resp1.total_tokens + resp2.total_tokens,
            "latency_ms": resp1.latency_ms + resp2.latency_ms
        }

        meta = dict(resp2.metadata) if resp2.metadata else {}
        meta.update({
            "verification_gated": True,
            "attempts_count": 2,
            "initial_error": err_msg,
            "initial_stage_failed": failed_stage
        })

        if v_res2.get("passed", False):
            print("  [Verification Gate] Retry successfully passed mechanical verification.")
            meta.update({
                "verification_status": "PASSED_ON_RETRY",
                "verification_passed": True,
                "syntax_valid": True,
                "execution_valid": True
            })
            return ModelResponse(
                text=resp2.text,
                prompt_tokens=combined_tokens["prompt_tokens"],
                completion_tokens=combined_tokens["completion_tokens"],
                total_tokens=combined_tokens["total_tokens"],
                latency_ms=combined_tokens["latency_ms"],
                model_name=self.model_name,
                model_revision=self.revision,
                metadata=meta
            )
        else:
            # Still failing after retry: NEVER silently pass broken code
            final_err = v_res2.get("error_message", "Unknown error")
            print(f"  [Verification Gate] Retry failed again: {final_err[:120]}. Flagging structural warning.")
            flagged_text = (
                f"{resp2.text}\n\n"
                f"> [!WARNING]\n"
                f"> **MECHANICAL VERIFICATION FAILED:** The generated code could not be verified mechanically.\n"
                f"> **Error Details:** {final_err}"
            )
            meta.update({
                "verification_status": "FAILED_AFTER_RETRY",
                "verification_passed": False,
                "final_error": final_err,
                "syntax_valid": v_res2.get("syntax_valid", False),
                "execution_valid": v_res2.get("execution_valid", False)
            })
            return ModelResponse(
                text=flagged_text,
                prompt_tokens=combined_tokens["prompt_tokens"],
                completion_tokens=combined_tokens["completion_tokens"],
                total_tokens=combined_tokens["total_tokens"],
                latency_ms=combined_tokens["latency_ms"],
                model_name=self.model_name,
                model_revision=self.revision,
                metadata=meta
            )

