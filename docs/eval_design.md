# Evaluation Design, Hypotheses & Statistical Protocol

**Project:** AI Search Framework (All-SLM Decomposed Pipeline vs. Monolithic LLM Baseline)  
**Phase:** 2 (Hypotheses & Eval Design)  
**Status:** Complete  
**Date:** 2026-08-27  

---

## 1. Research Questions & Formal Hypotheses

The evaluation follows a **within-subject, paired experimental design** (TRD §7): every evaluation query $q_i \in Q$ is executed through both the **Proposed All-SLM Pipeline** ($S$) and the **Fixed Monolithic Large-LLM Baseline** ($B$), ensuring identical query inputs across conditions.

### RQ1: Latency Reduction across Complexity
* **Question:** Does the all-SLM decomposed pipeline reduce end-to-end wall-clock latency compared to a single large-LLM baseline, and by how much across query complexity tiers?
* **Metric:** Latency Ratio $R_{lat}(q_i) = \frac{\text{Latency}_S(q_i)}{\text{Latency}_B(q_i)}$.
* **Hypotheses (per complexity tier $k$):**
  * $H_{0, \text{RQ1}}^{(k)}: \mu(\ln R_{lat}^{(k)}) \ge 0$ (No latency reduction in tier $k$).
  * $H_{1, \text{RQ1}}^{(k)}: \mu(\ln R_{lat}^{(k)}) < 0$ (Statistically significant latency reduction in tier $k$).
* **Statistical Test:** Paired Student's $t$-test (or Wilcoxon signed-rank test for non-normal distributions) on paired wall-clock latencies, with 95% Confidence Intervals per complexity bucket.

### RQ2: Compute / Token Cost Reduction
* **Question:** Does the all-SLM pipeline reduce compute and token cost compared to the LLM baseline?
* **Metric:** Cost Ratio $R_{cost}(q_i) = \frac{\text{Cost}_S(q_i)}{\text{Cost}_B(q_i)}$ computed using a versioned pricing/compute-cost table (`pricing_table.json`).
* **Hypotheses (per complexity tier $k$):**
  * $H_{0, \text{RQ2}}^{(k)}: \mu(\ln R_{cost}^{(k)}) \ge 0$ (No cost reduction in tier $k$).
  * $H_{1, \text{RQ2}}^{(k)}: \mu(\ln R_{cost}^{(k)}) < 0$ (Statistically significant cost reduction in tier $k$).
* **Statistical Test:** Paired $t$-test on log-transformed costs with 95% CIs per complexity tier.

### RQ3: Quality Non-Inferiority
* **Question:** Does the all-SLM pipeline's answer quality match the LLM baseline (non-inferiority), or does the quality gap between SLMs and an LLM outweigh the latency/cost savings?
* **Metric:** Mean paired quality difference $\Delta Q(q_i) = Q_S(q_i) - Q_B(q_i)$ on a standardized 1–5 blind rubric.
* **Pre-Registered Non-Inferiority Margin:** $\delta = 0.20$ points on the 1–5 scale.
* **Hypotheses:**
  * $H_{0, \text{RQ3}}: \mu(\Delta Q) \le -\delta$ (The SLM pipeline is inferior to the LLM baseline by $\ge \delta$).
  * $H_{1, \text{RQ3}}: \mu(\Delta Q) > -\delta$ (The SLM pipeline is non-inferior to the LLM baseline within margin $\delta$).
* **Statistical Test:** Paired one-sided non-inferiority $t$-test at $\alpha = 0.025$ (equivalent to lower bound of 95% two-sided CI $> -\delta$).

### RQ4: Complexity Crossover Point
* **Question:** At what query complexity does the pipeline's coordination overhead (decomposition latency, routing overhead, aggregator latency) stop paying for itself relative to the LLM baseline?
* **Metric:** Crossover identification where $R_{lat}(k) \ge 1.0$ or where $\Delta Q(k) \le -\delta$.
* **Analysis:** Empirical curve estimation plotting Latency Ratio and Quality Delta against DAG node count, dependency depth, and domain count.

### RQ5: Decomposition Accuracy & Downstream Error Attribution
* **Question:** How much does decomposition accuracy alone explain end-to-end quality gaps versus the LLM baseline?
* **Metric:** Graph Edit Distance (GED) and Structural Jaccard Similarity between generated DAGs and hand-labeled Gold DAGs on the gold subset ($n \ge 50$).
* **Statistical Test:** Pearson/Spearman correlation $r(\text{GED}(G_{\text{gen}}, G_{\text{gold}}), \Delta Q)$ to quantify the variance in quality delta explained by structural decomposition errors.

---

## 2. Evaluation Dataset Stratification

The evaluation dataset consists of **$N = 180$ queries**, partitioned into three complexity tiers:

```
                            [ Total Dataset: N = 180 ]
                                       |
         +-----------------------------+-----------------------------+
         |                             |                             |
  Tier 1: Single-Domain         Tier 2: 2-Domain Compound     Tier 3: 3+-Domain Compound
     (Control, n = 50)               (n = 70)                      (n = 60)
         |                             |                             |
  - Coding (15)                 - Code + Math (20)            - Code + Math + Reasoning (20)
  - Math (15)                   - Code + Reasoning (20)       - Retrieval + Code + Reasoning (20)
  - Reasoning / Logic (10)      - Retrieval + Reasoning (15)  - Retrieval + Math + Policy (20)
  - Retrieval / General (10)    - Math + General (15)
```

### Stratification Criteria
1. **Tier 1: Single-Domain (Control) ($n = 50$):**  
   Atomic queries that require exactly one specialized capability (e.g., standard algorithmic implementation, single-variable calculus problem, formal logical deduction, factual recall). Serves as a baseline to measure the pipeline's fixed decomposition/aggregation overhead.
2. **Tier 2: 2-Domain Compound ($n = 70$):**  
   Queries requiring two distinct capabilities with clear sub-problem boundaries (e.g., writing a Python simulation script for a stochastic math model; retrieving historical financial data and formulating an analytical investment rationale).
3. **Tier 3: 3+-Domain Compound ($n = 60$):**  
   Highly complex queries requiring decomposition across $\ge 3$ capabilities with dependency edges (e.g., retrieving regulatory tax documentation, deriving mathematical amortization formulas, implementing Python verification routines, and synthesizing a structured executive policy comparison).

---

## 3. Sample Size & Power Calculation

To ensure statistical validity and avoid underpowered comparisons (TRD §8), we compute the required sample size:

### 3.1 Quality Non-Inferiority Power Analysis
* **Parameters:**
  * Non-inferiority margin: $\delta = 0.20$ points (on 1–5 scale).
  * True mean difference under $H_1$: $\mu_0 = 0.0$ (assuming true quality parity).
  * Estimated standard deviation of paired differences: $\sigma_d = 0.65$.
  * Significance level: $\alpha = 0.025$ (one-sided).
  * Target statistical power: $1 - \beta = 0.80$.
* **Formula:**
  $$n = \frac{(Z_{1-\alpha} + Z_{1-\beta})^2 \cdot \sigma_d^2}{\delta^2} = \frac{(1.96 + 0.8416)^2 \cdot (0.65)^2}{(0.20)^2} = \frac{7.8489 \cdot 0.4225}{0.04} \approx 82.9 \implies n \ge 83 \text{ pairs}$$
* **Adequacy:** $N = 180$ total queries comfortably exceeds this threshold ($n = 83$), providing $> 95\%$ power for the aggregate test and adequate power ($\approx 75\text{--}80\%$) within individual complexity strata ($n \ge 50\text{--}70$).

### 3.2 Latency Ratio Power Analysis
* **Parameters:**
  * Minimum detectable effect size: Cohen's $d = 0.35$ (detecting moderate latency shift $\ge 25\%$).
  * Significance level: $\alpha = 0.05$ (two-sided).
  * Target power: $1 - \beta = 0.80$.
* **Formula:**
  $$n = \frac{(Z_{1-\alpha/2} + Z_{1-\beta})^2}{d^2} = \frac{(1.96 + 0.8416)^2}{(0.35)^2} = \frac{7.8489}{0.1225} \approx 64.1 \implies n \ge 65 \text{ pairs}$$
* **Adequacy:** With $n = 50\text{--}70$ per tier and $N = 180$ in total, each tier is powered to detect meaningful latency differences.

---

## 4. Pre-Registered Blind Quality Rubric

Responses are evaluated across three dimensions on an integer scale from 1 to 5.

```
+-------------------+---------------------------------------------------------+
| Dimension         | Description & Criteria                                  |
+-------------------+---------------------------------------------------------+
| 1. Correctness    | Factual truth, mathematical rigor, code validity, and   |
|                   | logical soundness across all sub-facets.                |
|                   |  5: Flawless logic, fully correct code/math.            |
|                   |  3: Minor non-critical inaccuracy or syntax bug.        |
|                   |  1: Critical logical fallacy, hallucination, or wrong.  |
+-------------------+---------------------------------------------------------+
| 2. Completeness   | Thoroughness in addressing every explicit and implicit  |
|                   | constraint in compound multi-domain requests.           |
|                   |  5: Every domain facet completely addressed.            |
|                   |  3: Primary facet covered, secondary sub-task shallow.  |
|                   |  1: Substantial parts of the user prompt omitted.       |
+-------------------+---------------------------------------------------------+
| 3. Coherence &    | Logical flow, integration of multi-domain findings,     |
|    Synthesis      | contradiction resolution, and absence of disjoint       |
|                   | copy-pasted blocks.                                     |
|                   |  5: Seamlessly unified synthesis, clear structure.      |
|                   |  3: Understandable but disjoint subtask concatenation.  |
|                   |  1: Incoherent, contradictory, or disjoint output.      |
+-------------------+---------------------------------------------------------+
```

### Double-Blind Presentation Protocol
1. **Sanitization:** System identifiers, headers, and metadata are stripped.
2. **Randomization:** Responses from $S$ (SLM Pipeline) and $B$ (LLM Baseline) are randomly assigned anonymous aliases (`Candidate A` vs `Candidate B`).
3. **Independent Key:** The unblinding key is written to a segregated mapping file (`logs/blind_eval_key_<timestamp>.json`) and withheld from human/model raters until all scores are committed.
4. **Inter-Rater Reliability:** When multiple raters/judges score queries, Cohen's $\kappa$ or Intraclass Correlation Coefficient (ICC) is computed and reported.

---

## 5. Held-Out Split Discipline

To prevent data contamination and prompt overfitting (TRD §5, AGENTS.md Rule 5):
* **Development Split ($n = 60$):** Used for initial decomposer prompt engineering, router calibration, and aggregator prompt tuning.
* **Held-Out Split ($n = 120$):** Strictly isolated and locked. **No agent code or prompt iteration may read, print, or evaluate against this split until Phase 7/8.**
* **Gold DAG Subset ($n = 60$):** Hand-annotated task graphs (subtasks, capability tags, dependencies) split across Dev ($n = 20$) and Held-Out ($n = 40$) for RQ5 decomposition accuracy evaluation.

