"""
SLM-2: Task Analyser (v2 Architecture)
Computes a 5-dimensional normalized skill requirement vector for each task.
Skill vector: <coding, math, logic/reasoning, retrieval, general>
"""

import math
import re
from typing import Dict, List, Any, Optional
from ...models.base import BaseModelRunner, ModelResponse

SKILL_CATEGORIES = ["coding", "math", "reasoning", "retrieval", "general"]

LEXICAL_PATTERNS = {
    "coding": [
        r"\bpython\b", r"\bcode\b", r"\bfunction\b", r"\bclass\b", r"\balgorithm\b",
        r"\bimplement\b", r"\bscript\b", r"\bdebug\b", r"\brefactor\b", r"\bconcurrency\b",
        r"\basync\b", r"\bapi\b", r"\bmiddleware\b", r"\bdata structure\b", r"\bstack\b",
        r"\bqueue\b", r"\btree\b", r"\bgraph\b", r"\bhash\b", r"\bpointer\b", r"\bthread\b",
        r"\bparser\b", r"\bdatabase\b", r"\bsql\b", r"\brust\b", r"\bunit test\b", r"\bsimulation\b"
    ],
    "math": [
        r"\bmath\b", r"\bderivative\b", r"\bderive\b", r"\bintegral\b", r"\bcalculus\b", r"\bequation\b",
        r"\bformula\b", r"\beigenvalue\b", r"\beigenvector\b", r"\bmatrix\b", r"\bprobability\b",
        r"\bvariance\b", r"\bdistribution\b", r"\btheorem\b", r"\bproof\b", r"\bseries\b",
        r"\bconvergence\b", r"\blinear algebra\b", r"\boptimization\b", r"\bhessian\b",
        r"\bgradient\b", r"\bmonte carlo\b", r"\bmcmc\b", r"\bkalman\b", r"\bfilter\b", r"\bstatistics\b"
    ],
    "reasoning": [
        r"\breasoning\b", r"\blogic\b", r"\bsyllogism\b", r"\bvalidity\b", r"\bfallacy\b",
        r"\bdeduction\b", r"\binduction\b", r"\bparadox\b", r"\bcausal\b", r"\btrade-off\b",
        r"\bevaluate\b", r"\banalyze\b", r"\bwhy\b", r"\bcompare\b", r"\bcontrast\b",
        r"\binvariant\b", r"\bgame-theoretic\b", r"\bequilibrium\b", r"\bformal proof\b", r"\bverify\b"
    ],
    "retrieval": [
        r"\bretrieve\b", r"\bsearch\b", r"\bextract\b", r"\bstandard\b", r"\bspecification\b",
        r"\brfc\b", r"\bregulation\b", r"\bstatute\b", r"\bframework\b", r"\bguideline\b",
        r"\bliterature\b", r"\bhistory\b", r"\bchronology\b", r"\bfactual\b", r"\bprotocol\b"
    ],
    "general": [
        r"\bsummarize\b", r"\bexplain\b", r"\boverview\b", r"\bdescribe\b", r"\bgeneral\b",
        r"\bpolicy\b", r"\bintroduction\b", r"\bbackground\b", r"\bsynthesis\b", r"\bcontext\b"
    ]
}

COMPILED_PATTERNS = {
    cat: [re.compile(p, re.IGNORECASE) for p in pats]
    for cat, pats in LEXICAL_PATTERNS.items()
}

class TaskAnalyserSLM:
    def __init__(self, model_runner: Optional[BaseModelRunner] = None):
        self.runner = model_runner

    def analyse_skill_vector(self, task_text: str, prior_capability_tag: str = "") -> Dict[str, float]:
        """
        Computes 5D normalized skill vector: s = <code, math, reasoning, retrieval, general>.
        Uses multi-token semantic keyword matching + temperature-scaled softmax.
        """
        raw_scores: Dict[str, float] = {cat: 0.2 for cat in SKILL_CATEGORIES}

        # Match lexical pattern hits
        for cat, pats in COMPILED_PATTERNS.items():
            for pat in pats:
                matches = len(pat.findall(task_text))
                if matches > 0:
                    raw_scores[cat] += matches * 2.0

        if prior_capability_tag and prior_capability_tag in raw_scores:
            raw_scores[prior_capability_tag] += 3.5

        # Normalize with temperature-scaled softmax
        temperature = 1.0
        scaled = {k: v / temperature for k, v in raw_scores.items()}
        max_s = max(scaled.values())
        exp_s = {k: math.exp(v - max_s) for k, v in scaled.items()}
        total_exp = sum(exp_s.values())

        normalized_vector = {k: round(exp_s[k] / total_exp, 4) for k in SKILL_CATEGORIES}
        return normalized_vector
