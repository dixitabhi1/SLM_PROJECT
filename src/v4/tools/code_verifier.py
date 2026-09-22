"""
Mechanical Code Verification Gate
AI Search Framework - Version 4 Step 3
Performs deterministic AST syntax analysis and sandboxed subprocess execution checks.
Zero LLM in the verification loop.
"""

import ast
import os
import re
import sys
import tempfile
import subprocess
from typing import Dict, Any, List, Optional, Tuple

class MechanicalCodeVerifier:
    def __init__(self, execution_timeout_sec: float = 6.0):
        self.execution_timeout_sec = execution_timeout_sec

    def extract_python_blocks(self, text: str) -> List[str]:
        """
        Extracts all fenced python code blocks from markdown text.
        Handles closed fences, unclosed fences, and raw python code.
        Guarantees that markdown fence markers (```python or ```) are never passed to ast.parse.
        """
        # 1. Closed fences
        pattern_closed = r"```(?:python|py)?\s*\n(.*?)```"
        matches = re.findall(pattern_closed, text, re.DOTALL)
        if matches:
            cleaned = []
            for m in matches:
                lines = [l for l in m.strip().splitlines() if not l.strip().startswith("```")]
                code_clean = "\n".join(lines).strip()
                if code_clean:
                    cleaned.append(code_clean)
            if cleaned:
                return cleaned

        # 2. Unclosed fence (e.g. model reached token limit before closing fence)
        pattern_unclosed = r"```(?:python|py)?\s*\n(.*)"
        match_unclosed = re.search(pattern_unclosed, text, re.DOTALL)
        if match_unclosed:
            lines = [l for l in match_unclosed.group(1).strip().splitlines() if not l.strip().startswith("```")]
            code_clean = "\n".join(lines).strip()
            if code_clean:
                return [code_clean]

        # 3. Fallback: check if text contains def/import
        if "def " in text or "import " in text:
            lines = [
                l for l in text.splitlines()
                if not l.strip().startswith("```") and not l.startswith("#") and not l.startswith("Step")
            ]
            candidate = "\n".join(lines).strip()
            if candidate:
                return [candidate]
        return []

    def verify_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Validates Python code syntax via ast.parse().
        Returns (is_valid, error_detail).
        """
        code = code.strip()
        # Defense-in-depth: strip leading/trailing markdown fence markers
        if code.startswith("```"):
            code = re.sub(r"^```(?:python|py)?\s*\n?", "", code)
        if code.endswith("```"):
            code = re.sub(r"\n?```\s*$", "", code)
        code = code.strip()

        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"SyntaxError at line {e.lineno}, offset {e.offset}: {e.msg}\n  Line: {e.text}"
        except Exception as e:
            return False, f"ParseError: {str(e)}"

    def verify_execution(self, code: str) -> Tuple[bool, Optional[str], str]:
        """
        Executes code in an isolated subprocess with strict execution timeout.
        Returns (success, error_detail, stdout).
        """
        code = code.strip()
        if code.startswith("```"):
            code = re.sub(r"^```(?:python|py)?\s*\n?", "", code)
        if code.endswith("```"):
            code = re.sub(r"\n?```\s*$", "", code)
        code = code.strip()

        # Quick syntax check first
        syntax_ok, syntax_err = self.verify_syntax(code)
        if not syntax_ok:
            return False, syntax_err, ""

        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tf:
            tf.write(code)
            tmp_path = tf.name

        try:
            res = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=self.execution_timeout_sec
            )
            if res.returncode == 0:
                stdout_text = res.stdout.strip()
                # Enhanced Verification Gate: Ensure code is not merely an un-invoked function stub
                try:
                    tree = ast.parse(code)
                    non_defs = [
                        node for node in tree.body
                        if not isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))
                    ]
                    has_main_block = any(
                        isinstance(node, ast.If) and (
                            (isinstance(node.test, ast.Compare) and any(getattr(comp, "value", "") == "__main__" or getattr(comp, "s", "") == "__main__" for comp in node.test.comparators)) or
                            "__main__" in ast.unparse(node.test)
                        )
                        for node in tree.body
                    )
                    has_top_level_call = any(
                        isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                        for node in tree.body
                    )

                    if not non_defs:
                        return False, (
                            "ExecutionIncompleteError: Code defines functions/classes but contains no active execution statements "
                            "or entry point (stdout is empty). The script must invoke its benchmark/simulation loop."
                        ), stdout_text

                    if not has_main_block and not has_top_level_call and not stdout_text:
                        return False, (
                            "ExecutionIncompleteError: Code contains no main invocation block (e.g. if __name__ == '__main__':) "
                            "and produced zero execution output. An executable benchmark/solver script is required."
                        ), stdout_text

                    if not stdout_text:
                        return False, (
                            "ExecutionIncompleteError: Code executed with return code 0 but produced zero stdout output. "
                            "Benchmark loops and solvers must run and print convergence status, timings, or summary metrics."
                        ), stdout_text

                except Exception as ex:
                    # If AST inspection fails unexpectedly, fallback to execution return code check
                    pass

                return True, None, stdout_text
            else:
                err = res.stderr.strip() or f"Process exited with return code {res.returncode}"
                return False, f"RuntimeError:\n{err}", res.stdout.strip()
        except subprocess.TimeoutExpired:
            return False, f"TimeoutExpired: Execution exceeded {self.execution_timeout_sec}s safety limit", ""
        except Exception as e:
            return False, f"ExecutionError: {str(e)}", ""
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def verify_full(self, text: str, require_code: bool = True) -> Dict[str, Any]:
        """
        Full verification pipeline across all extracted code blocks.
        If require_code is True (default), responses with no code fail verification.
        """
        blocks = self.extract_python_blocks(text)
        if not blocks:
            if require_code:
                return {
                    "has_code": False,
                    "passed": False,
                    "stage_failed": "missing_code",
                    "syntax_valid": False,
                    "execution_valid": False,
                    "error_message": (
                        "MissingCodeError: The response contained no valid Python code blocks. "
                        "An executable Python script enclosed in ```python ... ``` is strictly required."
                    ),
                    "blocks_analyzed": 0
                }
            return {
                "has_code": False,
                "passed": True, # Non-code outputs pass verification when require_code is False
                "syntax_valid": True,
                "execution_valid": True,
                "error_message": None,
                "blocks_analyzed": 0
            }

        for idx, block in enumerate(blocks, 1):
            syntax_ok, syntax_err = self.verify_syntax(block)
            if not syntax_ok:
                return {
                    "has_code": True,
                    "passed": False,
                    "stage_failed": "syntax",
                    "block_index": idx,
                    "syntax_valid": False,
                    "execution_valid": False,
                    "error_message": syntax_err,
                    "failed_code": block
                }

            exec_ok, exec_err, stdout = self.verify_execution(block)
            if not exec_ok:
                return {
                    "has_code": True,
                    "passed": False,
                    "stage_failed": "execution",
                    "block_index": idx,
                    "syntax_valid": True,
                    "execution_valid": False,
                    "error_message": exec_err,
                    "failed_code": block
                }

        return {
            "has_code": True,
            "passed": True,
            "syntax_valid": True,
            "execution_valid": True,
            "error_message": None,
            "blocks_analyzed": len(blocks)
        }

