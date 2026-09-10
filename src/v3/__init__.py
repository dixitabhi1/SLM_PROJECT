"""
AI Search Framework v3
All-SLM Decomposed Pipeline with 8 Specialized Domain Models (all <=5B)
"""

from .analyser.task_analyser import TaskAnalyserSLM_v3
from .colorer.task_colorer import TaskColorerSLM_v3
from .matching.matching_slm import MatchingSLM_v3
from .pipeline import SLMPipeline_v3

__all__ = [
    "TaskAnalyserSLM_v3",
    "TaskColorerSLM_v3",
    "MatchingSLM_v3",
    "SLMPipeline_v3"
]

