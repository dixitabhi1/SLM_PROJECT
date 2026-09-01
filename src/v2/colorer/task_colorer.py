"""
SLM-3: Task Colorer (v2 Architecture - Branch A)
Colors tasks by similarity according to their 5D skill vector.
Colors: blue (coding), green (math), purple (reasoning), amber (retrieval), slate (general)
"""

from typing import Dict, List, Any, Tuple

COLOR_TAXONOMY = {
    "coding": "blue",
    "math": "green",
    "reasoning": "purple",
    "retrieval": "amber",
    "general": "slate"
}

COLOR_TO_DOMAIN = {v: k for k, v in COLOR_TAXONOMY.items()}

class TaskColorerSLM:
    def __init__(self, multi_color_threshold: float = 0.22):
        self.threshold = multi_color_threshold

    def color_task(self, skill_vector: Dict[str, float]) -> Dict[str, Any]:
        """
        Determines the dominant color, active colors, and whether the task spans multiple colors.
        """
        # Active colors exceeding threshold
        active_domains = [domain for domain, weight in skill_vector.items() if weight >= self.threshold]
        active_colors = [COLOR_TAXONOMY[d] for d in active_domains]

        # Primary / dominant color
        dominant_domain = max(skill_vector.items(), key=lambda x: x[1])[0]
        dominant_color = COLOR_TAXONOMY[dominant_domain]

        # Multi-color check: spans > 1 distinct color class
        spans_multiple = len(active_colors) > 1

        return {
            "dominant_color": dominant_color,
            "dominant_domain": dominant_domain,
            "active_colors": active_colors if active_colors else [dominant_color],
            "active_domains": active_domains if active_domains else [dominant_domain],
            "spans_multiple_colors": spans_multiple,
            "color_count": len(active_colors) if active_colors else 1
        }

