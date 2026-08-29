"""
Capability Router
AI Search Framework
Rule-based & embedding-similarity router (Non-generative, near-zero overhead per TRD §2).
"""

import re
import math
from typing import Dict, List, Any, Tuple

class CapabilityRouter:
    """
    Non-generative router mapping subtask text to specialized SLMs.
    Uses multi-token keyword taxonomy and lexical heuristic vectors with confidence scoring.
    """
    
    CAPABILITY_TAXONOMY = {
        "coding": [
            r"\bpython\b", r"\bcode\b", r"\bfunction\b", r"\bclass\b", r"\balgorithm\b",
            r"\bimplement\b", r"\bscript\b", r"\bdebug\b", r"\brefactor\b", r"\bconcurrency\b",
            r"\basync\b", r"\bapi\b", r"\bmiddleware\b", r"\bdata structure\b", r"\bstack\b",
            r"\bqueue\b", r"\btree\b", r"\bgraph\b", r"\bhash\b", r"\bpointer\b", r"\bthread\b",
            r"\bcompiler\b", r"\bparser\b", r"\bdatabase\b", r"\bquery\b", r"\bsql\b", r"\brust\b"
        ],
        "math": [
            r"\bmath\b", r"\bderivative\b", r"\bintegral\b", r"\bcalculus\b", r"\bequation\b",
            r"\bformula\b", r"\beigenvalue\b", r"\beigenvector\b", r"\bmatrix\b", r"\bprobability\b",
            r"\bvariance\b", r"\bdistribution\b", r"\btheorem\b", r"\bproof\b", r"\bseries\b",
            r"\bconvergence\b", r"\blinear algebra\b", r"\boptimization\b", r"\bhessian\b",
            r"\bgradient\b", r"\bconvex\b", r"\bmonte carlo\b", r"\bmcmc\b", r"\bkalman\b"
        ],
        "reasoning": [
            r"\breasoning\b", r"\blogic\b", r"\bsyllogism\b", r"\bvalidity\b", r"\bfallacy\b",
            r"\bdeduction\b", r"\binduction\b", r"\bparadox\b", r"\bcausal\b", r"\btrade-off\b",
            r"\bevaluate\b", r"\banalyze\b", r"\bwhy\b", r"\bcompare\b", r"\bcontrast\b",
            r"\binvariant\b", r"\bgame-theoretic\b", r"\bequilibrium\b", r"\bformal proof\b"
        ],
        "retrieval": [
            r"\bretrieve\b", r"\bsearch\b", r"\bextract\b", r"\bstandard\b", r"\bspecification\b",
            r"\brfc\b", r"\bregulation\b", r"\bstatute\b", r"\bframework\b", r"\bguideline\b",
            r"\bliterature\b", r"\bhistory\b", r"\bchronology\b", r"\bfactual\b", r"\bprotocol\b"
        ],
        "general": [
            r"\bsummarize\b", r"\bexplain\b", r"\boverview\b", r"\bdescribe\b", r"\bgeneral\b",
            r"\bpolicy\b", r"\bintroduction\b", r"\bbackground\b", r"\bsynthesis\b"
        ]
    }

    def __init__(self, confidence_threshold: float = 0.70, replication_cutoff: float = 0.65):
        self.confidence_threshold = confidence_threshold
        self.replication_cutoff = replication_cutoff
        # Precompile regex patterns
        self.patterns = {
            cap: [re.compile(p, re.IGNORECASE) for p in pats]
            for cap, pats in self.CAPABILITY_TAXONOMY.items()
        }

    def route(self, subtask_text: str, explicit_capability_tag: str = "") -> Dict[str, Any]:
        """
        Routes a subtask to primary (and optionally replicated secondary) pool capabilities.
        Returns routing decision, confidence score, and replication recommendation.
        """
        scores: Dict[str, float] = {cap: 0.1 for cap in self.CAPABILITY_TAXONOMY.keys()}
        
        # 1. Match lexical features
        for cap, compiled_pats in self.patterns.items():
            for pat in compiled_pats:
                matches = len(pat.findall(subtask_text))
                if matches > 0:
                    scores[cap] += matches * 1.5

        # 2. Prior from explicit decomposer tag if present
        if explicit_capability_tag and explicit_capability_tag in scores:
            scores[explicit_capability_tag] += 3.0

        # 3. Softmax / normalization for calibrated confidence with temperature scaling
        temperature = 1.2
        scaled_scores = {cap: s / temperature for cap, s in scores.items()}
        max_s = max(scaled_scores.values())
        exp_scores = {cap: math.exp(s - max_s) for cap, s in scaled_scores.items()}
        total_exp = sum(exp_scores.values())
        probs = {cap: exp_scores[cap] / total_exp for cap in scores.keys()}
        
        sorted_caps = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        primary_cap, primary_conf = sorted_caps[0]
        secondary_cap, secondary_conf = sorted_caps[1]

        # Replicate if primary confidence is low and second candidate has meaningful support
        needs_replication = (primary_conf < self.replication_cutoff) and (secondary_conf >= 0.15)

        return {
            "primary_capability": primary_cap,
            "confidence": round(primary_conf, 4),
            "secondary_capability": secondary_cap if needs_replication else None,
            "secondary_confidence": round(secondary_conf, 4) if needs_replication else None,
            "replicate": needs_replication,
            "all_probabilities": {k: round(v, 4) for k, v in probs.items()}
        }
