"""
SLM-2: Task Analyser (v3 Architecture - 8 Domains)
Computes an 8-dimensional normalized skill requirement vector for each task.
Skill vector: <coding, mathematics, formal_reasoning, retrieval_qa, science_tech, structured_data, creative_synthesis, systems_ops>
"""

import math
import re
from typing import Dict, List, Any, Optional
from ...models.base import BaseModelRunner

SKILL_CATEGORIES_V3 = [
    "coding",
    "mathematics",
    "formal_reasoning",
    "retrieval_qa",
    "science_tech",
    "structured_data",
    "creative_synthesis",
    "systems_ops"
]

LEXICAL_PATTERNS_V3 = {
    "coding": [
        r"\bpython\b", r"\bcode\b", r"\bfunction\b", r"\bclass\b", r"\balgorithm\b",
        r"\bimplement\b", r"\bscript\b", r"\bdebug\b", r"\brefactor\b", r"\bdata structure\b",
        r"\bheap\b", r"\btree\b", r"\bgraph\b", r"\btrie\b", r"\bhash\b", r"\bunit test\b",
        r"\borm\b", r"\bapi\b", r"\bbackend\b",
        r"\btype annotations\b", r"\bcomplexity\b", r"\btime complexity\b", r"\bspace complexity\b"
    ],
    "mathematics": [
        r"\bmath\b", r"\bderivative\b", r"\bderive\b", r"\bintegral\b", r"\bcalculus\b", r"\bequation\b",
        r"\bformula\b", r"\beigenvalue\b", r"\beigenvector\b", r"\bmatrix\b", r"\bprobability\b",
        r"\bvariance\b", r"\bdistribution\b", r"\btheorem\b", r"\bproof\b", r"\bseries\b",
        r"\bconvergence\b", r"\blinear algebra\b", r"\boptimization\b", r"\bhessian\b",
        r"\bgradient\b", r"\bmonte carlo\b", r"\bmcmc\b", r"\bstatistics\b"
    ],
    "formal_reasoning": [
        r"\breasoning\b", r"\blogic\b", r"\bsyllogism\b", r"\bvalidity\b", r"\bfallacy\b",
        r"\bdeduction\b", r"\binduction\b", r"\bparadox\b", r"\bcausal\b", r"\bcounterfactual\b",
        r"\btrade-off\b", r"\bevaluate\b", r"\banalyze\b", r"\bwhy\b", r"\binvariant\b",
        r"\bgame-theoretic\b", r"\bequilibrium\b", r"\bformal proof\b", r"\bverify\b"
    ],
    "retrieval_qa": [
        r"\bretrieve\b", r"\bsearch\b", r"\bextract\b", r"\bstandard\b", r"\bspecification\b",
        r"\brfc\b", r"\bregulation\b", r"\bstatute\b", r"\bframework\b", r"\bguideline\b",
        r"\bliterature\b", r"\bhistory\b", r"\bchronology\b", r"\bfactual\b", r"\bprotocol\b",
        r"\bwho is\b", r"\bwhat is\b", r"\bwhen did\b"
    ],
    "science_tech": [
        r"\bphysics\b", r"\bchemistry\b", r"\bbiology\b", r"\bthermodynamics\b", r"\belectromagnetism\b",
        r"\bquantum\b", r"\bgenetics\b", r"\bcircuit\b", r"\bsemiconductor\b", r"\bkinetic\b",
        r"\benergy\b", r"\bchemical\b", r"\bmolecular\b", r"\bcellular\b", r"\boptics\b"
    ],
    "structured_data": [
        r"\bsql\b", r"\bdatabase\b", r"\btable\b", r"\bschema\b", r"\brelational\b",
        r"\bjson\b", r"\byaml\b", r"\bprotobuf\b", r"\bast\b", r"\bparquet\b",
        r"\borm\b", r"\bentity\b", r"\bdata model\b",
        r"\bjoin\b", r"\bindex\b", r"\bforeign key\b", r"\bquery\b", r"\bdata transformation\b"
    ],
    "creative_synthesis": [
        r"\bsummarize\b", r"\boverview\b", r"\bdescribe\b", r"\bsynthesis\b", r"\breport\b",
        r"\bexecutive summary\b", r"\bharmonize\b", r"\bperspective\b", r"\bessay\b",
        r"\bprose\b", r"\bpresentation\b", r"\bintroduction\b", r"\bconclusion\b"
    ],
    "systems_ops": [
        r"\bbash\b", r"\bshell\b", r"\blinux\b", r"\bdocker\b", r"\bkubernetes\b",
        r"\bdevops\b", r"\bposix\b", r"\bprocess\b", r"\bsocket\b", r"\bconcurrency\b",
        r"\bthread\b", r"\bmemory management\b", r"\bkernel\b", r"\bnetwork\b", r"\bport\b"
    ]
}

COMPILED_PATTERNS_V3 = {
    cat: [re.compile(p, re.IGNORECASE) for p in pats]
    for cat, pats in LEXICAL_PATTERNS_V3.items()
}

class TaskAnalyserSLM_v3:
    def __init__(self, model_runner: Optional[BaseModelRunner] = None):
        self.runner = model_runner

    def analyse_skill_vector(self, task_text: str, prior_capability_tag: str = "", temperature: float = 1.8) -> Dict[str, float]:
        """
        Computes 8D normalized skill vector:
        s = <coding, mathematics, formal_reasoning, retrieval_qa, science_tech, structured_data, creative_synthesis, systems_ops>.
        Uses multi-token semantic keyword matching + temperature-scaled softmax.
        """
        raw_scores: Dict[str, float] = {cat: 0.1 for cat in SKILL_CATEGORIES_V3}

        # Match lexical pattern hits
        for cat, pats in COMPILED_PATTERNS_V3.items():
            for pat in pats:
                matches = len(pat.findall(task_text))
                if matches > 0:
                    raw_scores[cat] += matches * 2.0

        if prior_capability_tag and prior_capability_tag in raw_scores:
            raw_scores[prior_capability_tag] += 3.5

        # Normalize with temperature-scaled softmax
        scaled = {k: v / temperature for k, v in raw_scores.items()}
        max_s = max(scaled.values())
        exp_s = {k: math.exp(v - max_s) for k, v in scaled.items()}
        total_exp = sum(exp_s.values())

        normalized_vector = {k: round(exp_s[k] / total_exp, 4) for k in SKILL_CATEGORIES_V3}
        return normalized_vector

