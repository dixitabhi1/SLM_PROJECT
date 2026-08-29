"""
vLLM / Ollama OpenAI-Compatible API Runner
AI Search Framework
Uses standard library urllib (async via asyncio.to_thread) for zero external dependencies.
"""

import asyncio
import json
import time
import urllib.request
import urllib.error
from typing import Optional, Dict, Any
from .base import BaseModelRunner, ModelResponse

class VLLMModelRunner(BaseModelRunner):
    def __init__(
        self,
        model_name: str,
        revision: str = "main",
        endpoint: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        timeout_sec: float = 60.0,
        **kwargs
    ):
        super().__init__(model_name, revision, **kwargs)
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.timeout_sec = timeout_sec

    def _sync_post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.endpoint}/chat/completions"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)

    async def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> ModelResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature)
        }

        start_time = time.perf_counter()
        try:
            data = await asyncio.to_thread(self._sync_post, payload)
        except Exception as e:
            # Fallback handling for offline / connection errors
            end_time = time.perf_counter()
            return ModelResponse(
                text=f"[Error connecting to endpoint {self.endpoint}: {e}]",
                prompt_tokens=len(prompt.split()) * 4 // 3,
                completion_tokens=10,
                total_tokens=(len(prompt.split()) * 4 // 3) + 10,
                latency_ms=(end_time - start_time) * 1000.0,
                model_name=self.model_name,
                model_revision=self.revision,
                metadata={"error": str(e)}
            )
            
        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000.0
        choice = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", len(prompt.split()) * 4 // 3)
        completion_tokens = usage.get("completion_tokens", len(choice.split()) * 4 // 3)

        return ModelResponse(
            text=choice,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=latency_ms,
            model_name=self.model_name,
            model_revision=self.revision,
            metadata={"endpoint": self.endpoint, "id": data.get("id")}
        )

