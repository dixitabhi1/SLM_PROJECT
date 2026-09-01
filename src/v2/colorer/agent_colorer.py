"""
Agent Colorer SLM (v2 Architecture - Branch B)
Assigns color profiles to pool models and identifies multi-skill bridging agents.
"""

from typing import Dict, List, Any
from .task_colorer import COLOR_TAXONOMY

class AgentColorerSLM:
    def __init__(self, bridging_threshold: float = 0.15):
        self.threshold = bridging_threshold

    def color_agent(self, agent_key: str, skill_vector: Dict[str, float]) -> Dict[str, Any]:
        primary_domain = max(skill_vector.items(), key=lambda x: x[1])[0]
        primary_color = COLOR_TAXONOMY[primary_domain]
        
        # Bridged colors (secondary skills with non-trivial support)
        bridged_domains = [d for d, w in skill_vector.items() if w >= self.threshold and d != primary_domain]
        bridged_colors = [COLOR_TAXONOMY[d] for d in bridged_domains]

        return {
            "agent_key": agent_key,
            "primary_domain": primary_domain,
            "primary_color": primary_color,
            "bridged_domains": bridged_domains,
            "bridged_colors": bridged_colors,
            "is_multi_skill_bridge": len(bridged_colors) > 0
        }

