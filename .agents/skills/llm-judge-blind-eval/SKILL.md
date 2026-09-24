---
name: llm-judge-blind-eval
description: Symmetrical double-blind evaluation under dual judging modes (1-5 3-criteria and 1-10 holistic), dual Quality Proximity metric computation, and first-class draw accounting.
---

# LLM Judge Blind Evaluation Framework

Source of truth: `.agents/knowledge/mentor_experiment_protocol_source.txt` §4–5 and `.agents/knowledge/TRD_source.txt` §7.

## Dual Parallel Judging Frameworks

Every evaluation trial under the Mentor Experiment Protocol must run both judging modes independently:

### Mode A: Established 1–5 Three-Criteria Protocol
- **Criteria Evaluated:** Correctness ($S_{\text{corr}} \in [1, 5]$), Completeness ($S_{\text{comp}} \in [1, 5]$), Coherence ($S_{\text{cohe}} \in [1, 5]$).
- **Composite Quality Score (CQS):** $\text{CQS} = \frac{S_{\text{corr}} + S_{\text{comp}} + S_{\text{cohe}}}{3.0} \in [1.00, 5.00]$.
- **Criteria Quality Proximity:**
  $$P_{\text{criteria}} = \left( 1 - \frac{|Q_S - Q_L|}{4.0} \right) \times 100\%$$
- **Outcome Classification:**
  - SLM Win: $Q_S > Q_L$
  - Baseline Win: $Q_L > Q_S$
  - Draw: $Q_S = Q_L$

### Mode B: New 1–10 Holistic Protocol
- **Score Scale:** Single holistic integer score $Q \in [1, 10]$ assigned blindly to each response.
- **Holistic Quality Proximity:**
  $$QP = \left( 1 - \frac{|Q_S - Q_L|}{9.0} \right)$$
- **Outcome Classification:**
  - SLM Win: $Q_S > Q_L$
  - LLM Win: $Q_L > Q_S$
  - Draw: $Q_S = Q_L$

## First-Class Draw Accounting

- Draws are a distinct, first-class logged outcome category.
- Never fold draws into wins or losses.
- Report all three rates independently:
  $$\text{SLM Win \%}, \quad \text{Draw \%}, \quad \text{LLM Win \%}$$

## Blinding and Positional Symmetry

1. Anonymize systems as Candidate A and Candidate B.
2. Evaluate both presentation orders for every query:
   - Forward: Candidate A = SLM, Candidate B = Baseline
   - Swapped: Candidate A = Baseline, Candidate B = SLM
3. Record cryptographic unblinding keys separately in `logs/*_judge_keys/`.
4. Enforce raw-file score/label concordance check on every trial.

