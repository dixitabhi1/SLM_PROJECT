"""
Agent Analyser SLM (v2 Architecture - Branch B)
Computes and maintains capability skill vectors for pool models.
"""

from typing import Dict, List, Any

# Pre-computed grounded skill vectors for the pinned candidate pool models
DEFAULT_AGENT_PROFILES = {
    "coding": {
        "model_name": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "primary_domain": "coding",
        "skill_vector": {"coding": 0.82, "math": 0.08, "reasoning": 0.05, "retrieval": 0.02, "general": 0.03}
    },
    "math": {
        "model_name": "Qwen/Qwen2.5-Math-7B-Instruct",
        "primary_domain": "math",
        "skill_vector": {"coding": 0.06, "math": 0.84, "reasoning": 0.06, "retrieval": 0.01, "general": 0.03}
    },
    "reasoning": {
        "model_name": "microsoft/Phi-3.5-mini-instruct",
        "primary_domain": "reasoning",
        "skill_vector": {"coding": 0.05, "math": 0.08, "reasoning": 0.78, "retrieval": 0.04, "general": 0.05}
    },
    "retrieval": {
        "model_name": "meta-llama/Llama-3.2-3B-Instruct",
        "primary_domain": "retrieval",
        "skill_vector": {"coding": 0.03, "math": 0.02, "reasoning": 0.05, "retrieval": 0.82, "general": 0.08}
    },
    "general": {
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "primary_domain": "general",
        "skill_vector": {"coding": 0.12, "math": 0.12, "reasoning": 0.18, "retrieval": 0.18, "general": 0.40}
    }
}

class AgentAnalyserSLM:
    def __init__(self, custom_profiles: Dict[str, Any] = None):
        self.profiles = custom_profiles or DEFAULT_AGENT_PROFILES

    def get_agent_skill_vector(self, agent_key: str) -> Dict[str, float]:
        if agent_key in self.profiles:
            return self.profiles[agent_key]["skill_vector"]
        return {"coding": 0.2, "math": 0.2, "reasoning": 0.2, "retrieval": 0.2, "general": 0.2}

    def get_all_agent_profiles(self) -> Dict[str, Any]:
        return self.profiles

