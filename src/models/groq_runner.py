"""
Live Groq API Model Runner
Implements BaseModelRunner interface via Groq OpenAI-compatible API endpoints.
"""

import asyncio
import json
import os
import time
import urllib.request
from typing import Dict, Any, Optional
from .base import BaseModelRunner, ModelResponse

class APIGroqModelRunner(BaseModelRunner):
    def __init__(
        self,
        logical_model_name: str,
        api_model_name: str = "qwen/qwen3.8-27b",
        api_key: Optional[str] = None,
        revision: str = "main",
        max_tokens: int = 2048,
        temperature: float = 0.0
    ):
        super().__init__(model_name=logical_model_name, revision=revision, max_tokens=max_tokens, temperature=temperature)
        self.api_model_name = api_model_name
        if not api_key:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key and os.path.exists(".env"):
                with open(".env", "r") as f:
                    for line in f:
                        if line.startswith("GROQ_API_KEY="):
                            api_key = line.strip().split("=", 1)[1]
                            break
        self.api_key = api_key

    async def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> ModelResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "curl/7.88.1"
        }
        body = {
            "model": self.api_model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens)
        }

        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST"
        )

        start_t = time.perf_counter()
        last_err = None

        for attempt in range(5):
            try:
                def _do_request():
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        return json.loads(resp.read().decode("utf-8"))

                data = await asyncio.to_thread(_do_request)
                end_t = time.perf_counter()
                latency_ms = (end_t - start_t) * 1000.0

                text = data["choices"][0]["message"]["content"]
                in_tok = data.get("usage", {}).get("prompt_tokens", len(prompt.split()) * 2)
                out_tok = data.get("usage", {}).get("completion_tokens", len(text.split()) * 2)

                return ModelResponse(
                    text=text,
                    prompt_tokens=in_tok,
                    completion_tokens=out_tok,
                    total_tokens=in_tok + out_tok,
                    latency_ms=latency_ms,
                    model_name=self.model_name,
                    model_revision=self.revision,
                    metadata={"api_model": self.api_model_name, "raw_usage": data.get("usage")}
                )
            except urllib.error.HTTPError as e:
                last_err = f"HTTPError {e.code}: {e.read().decode()[:200]}"
                if e.code == 429:
                    await asyncio.sleep(2.5 * (attempt + 1))
                else:
                    await asyncio.sleep(1.0)
            except Exception as e:
                last_err = str(e)
                await asyncio.sleep(1.0)

        end_t = time.perf_counter()
        return ModelResponse(
            text=f"[Error calling {self.api_model_name}: {last_err}]",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            latency_ms=(end_t - start_t) * 1000.0,
            model_name=self.model_name,
            model_revision=self.revision,
            metadata={"error": last_err}
        )

