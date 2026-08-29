"""
Statistical Analysis & Metrics Engine
AI Search Framework
Computes: Latency/Cost ratios, 95% CIs, paired non-inferiority test, Graph Edit Distance (GED), and crossover metrics.
Pure Python standard library implementation (no numpy/scipy dependency required).
"""

import math
import statistics
from typing import Dict, List, Any, Tuple, Optional

class StatisticalAnalyzer:
    def __init__(self, non_inferiority_margin: float = 0.20, alpha: float = 0.025, confidence: float = 0.95):
        self.non_inferiority_margin = non_inferiority_margin
        self.alpha = alpha
        self.confidence = confidence

    # -------------------------------------------------------------
    # 1. Confidence Intervals & Paired Ratio Analysis (RQ1 & RQ2)
    # -------------------------------------------------------------
    def compute_paired_ratio_stats(self, slm_values: List[float], baseline_values: List[float]) -> Dict[str, Any]:
        """
        Computes geometric and arithmetic ratios with 95% Confidence Intervals using standard log-normal methodology.
        """
        if len(slm_values) != len(baseline_values) or len(slm_values) == 0:
            return {"error": "Invalid sample sizes for paired ratio computation."}

        n = len(slm_values)
        ratios = [s / max(1e-6, b) for s, b in zip(slm_values, baseline_values)]
        log_ratios = [math.log(max(1e-6, r)) for r in ratios]

        mean_ratio = statistics.mean(ratios)
        median_ratio = statistics.median(ratios)
        
        mean_log = statistics.mean(log_ratios)
        std_log = statistics.stdev(log_ratios) if n > 1 else 0.0
        se_log = std_log / math.sqrt(n)
        
        # 95% critical value
        z_crit = 1.96 if n >= 30 else 2.05
        ci_lower = math.exp(mean_log - z_crit * se_log)
        ci_upper = math.exp(mean_log + z_crit * se_log)

        return {
            "n": n,
            "mean_ratio": round(mean_ratio, 4),
            "median_ratio": round(median_ratio, 4),
            "geometric_mean_ratio": round(math.exp(mean_log), 4),
            "ci_95_lower": round(ci_lower, 4),
            "ci_95_upper": round(ci_upper, 4),
            "std_log": round(std_log, 4)
        }

    # -------------------------------------------------------------
    # 2. Quality Non-Inferiority Testing (RQ3)
    # -------------------------------------------------------------
    def compute_non_inferiority_test(
        self,
        slm_scores: List[float],
        baseline_scores: List[float],
        margin: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Paired one-sided non-inferiority test: H0: mu(S - B) <= -delta vs H1: mu(S - B) > -delta.
        """
        delta = margin if margin is not None else self.non_inferiority_margin
        n = len(slm_scores)
        if n != len(baseline_scores) or n == 0:
            return {"error": "Invalid sample sizes."}

        diffs = [s - b for s, b in zip(slm_scores, baseline_scores)]
        mean_diff = statistics.mean(diffs)
        std_diff = statistics.stdev(diffs) if n > 1 else 0.0
        se_diff = std_diff / math.sqrt(n)

        # Non-inferiority test statistic: t = (mean_diff - (-delta)) / se_diff
        t_stat = (mean_diff + delta) / se_diff if se_diff > 0 else 0.0
        
        # Normal approximation for one-sided p-value
        p_val = 0.5 * math.erfc(t_stat / math.sqrt(2))

        # 95% Two-sided (or 97.5% one-sided) lower bound:
        lower_bound_95 = mean_diff - 1.96 * se_diff
        is_non_inferior = lower_bound_95 > -delta

        return {
            "n": n,
            "mean_quality_diff": round(mean_diff, 4),
            "std_diff": round(std_diff, 4),
            "se_diff": round(se_diff, 4),
            "non_inferiority_margin": delta,
            "lower_bound_95_ci": round(lower_bound_95, 4),
            "t_statistic": round(t_stat, 4),
            "p_value": round(p_val, 6),
            "non_inferiority_demonstrated": is_non_inferior
        }

    # -------------------------------------------------------------
    # 3. Decomposition Accuracy: Graph Edit Distance (RQ5)
    # -------------------------------------------------------------
    def compute_graph_edit_distance(self, gen_dag: Dict[str, Any], gold_dag: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculates Graph Edit Distance (GED) and Structural Jaccard Similarity between generated and gold DAGs.
        """
        gen_nodes = gen_dag.get("subtasks", [])
        gold_nodes = gold_dag.get("subtasks", [])

        n_gen = len(gen_nodes)
        n_gold = len(gold_nodes)

        # Node capability matching
        gen_caps = [n.get("capability", "general") for n in gen_nodes]
        gold_caps = [n.get("capability", "general") for n in gold_nodes]

        # Greedy capability alignment
        matched_nodes = 0
        temp_gold = list(gold_caps)
        for c in gen_caps:
            if c in temp_gold:
                matched_nodes += 1
                temp_gold.remove(c)

        node_insertions = max(0, n_gen - n_gold)
        node_deletions = max(0, n_gold - n_gen)
        node_substitutions = max(n_gen, n_gold) - matched_nodes - (node_insertions + node_deletions)

        # Edge matching
        def extract_edge_pairs(tasks):
            edges = set()
            id_to_idx = {t["id"]: idx for idx, t in enumerate(tasks)}
            for t in tasks:
                v = id_to_idx.get(t["id"])
                for u_id in t.get("dependencies", []):
                    u = id_to_idx.get(u_id)
                    if u is not None and v is not None:
                        edges.add((u, v))
            return edges

        gen_edges = extract_edge_pairs(gen_nodes)
        gold_edges = extract_edge_pairs(gold_nodes)

        edge_diff = len(gen_edges.symmetric_difference(gold_edges))
        total_ged = node_insertions + node_deletions + node_substitutions + edge_diff

        max_possible_edits = max(1, n_gen + n_gold + len(gen_edges) + len(gold_edges))
        similarity = max(0.0, 1.0 - (total_ged / max_possible_edits))

        return {
            "ged": total_ged,
            "node_count_gen": n_gen,
            "node_count_gold": n_gold,
            "matched_capabilities": matched_nodes,
            "edge_diff": edge_diff,
            "structural_similarity": round(similarity, 4),
            "exact_match": total_ged == 0
        }

