"""
Base Model Runner Interface
AI Search Framework
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class ModelResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    model_name: str
    model_revision: str = "main"
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseModelRunner(ABC):
    def __init__(self, model_name: str, revision: str = "main", max_tokens: int = 2048, temperature: float = 0.0):
        self.model_name = model_name
        self.revision = revision
        self.max_tokens = max_tokens
        self.temperature = temperature

    @abstractmethod
    async def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> ModelResponse:
        """Generate a response asynchronously given prompt and optional system prompt."""
        pass

