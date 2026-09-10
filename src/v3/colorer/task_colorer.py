"""
SLM-3: Task Colorer (v3 Architecture - 8 Domains)
Colors tasks according to their 8D skill vector:
Colors: blue, green, purple, amber, teal, cyan, rose, orange.
Enforces Fix 3-narrow: creative_synthesis ('rose') is excluded from triggering multi-color split.
"""

from typing import Dict, List, Any, Tuple

COLOR_TAXONOMY_V3 = {
    "coding": "blue",
    "mathematics": "green",
    "formal_reasoning": "purple",
    "retrieval_qa": "amber",
    "science_tech": "teal",
    "structured_data": "cyan",
    "creative_synthesis": "rose",
    "systems_ops": "orange"
}

COLOR_TO_DOMAIN_V3 = {v: k for k, v in COLOR_TAXONOMY_V3.items()}

class TaskColorerSLM_v3:
    def __init__(self, multi_color_threshold: float = 0.20):
        self.threshold = multi_color_threshold

    def color_task(self, skill_vector: Dict[str, float]) -> Dict[str, Any]:
        """
        Determines the dominant color, active colors, and whether the task spans multiple specialist colors.
        """
        # Active domains exceeding threshold
        active_domains = [domain for domain, weight in skill_vector.items() if weight >= self.threshold]
        active_colors = [COLOR_TAXONOMY_V3[d] for d in active_domains]

        # Primary / dominant domain and color
        dominant_domain = max(skill_vector.items(), key=lambda x: x[1])[0]
        dominant_color = COLOR_TAXONOMY_V3[dominant_domain]

        # Fix 3-narrow: creative_synthesis ('rose') conversational text is excluded from triggering multi-color decomposition
        non_synthesis_active_colors = [c for c in active_colors if c != "rose"]
        spans_multiple = len(non_synthesis_active_colors) > 1

        return {
            "dominant_color": dominant_color,
            "dominant_domain": dominant_domain,
            "active_colors": active_colors if active_colors else [dominant_color],
            "active_domains": active_domains if active_domains else [dominant_domain],
            "spans_multiple_colors": spans_multiple,
            "color_count": len(active_colors) if active_colors else 1
        }

