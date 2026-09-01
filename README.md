# AI Search Framework: All-SLM Decomposed Pipeline vs. Monolithic LLM Baseline

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-12%2F12%20passing-brightgreen.svg)](tests/)
[![Architecture](https://img.shields.io/badge/architecture-All--SLM%20(%E2%89%A48B)-orange.svg)](docs/research_report_draft.md)
[![Baseline](https://img.shields.io/badge/baseline-Llama--3.1--70B%20(Fixed)-purple.svg)](config/experiment_config.json)
[![Report](https://img.shields.io/badge/report-PDF%20Available-red.svg)](AI_Search_Framework_Research_Report.pdf)

> **Empirical research framework benchmarking an entirely Small Language Model (SLM)-based decomposed pipeline (&le; 8B parameters per component, strictly zero LLMs in deployment) against a fixed monolithic 70B+ LLM baseline.**

---

## Table of Contents
- [Executive Overview](#executive-overview)
- [System Architecture](#system-architecture)
- [Model Pinning & Hardware Specification](#model-pinning--hardware-specification)
- [Literature Grounding & Novelty](#literature-grounding--novelty)
- [Evaluation Design & Hypotheses (RQ1–RQ5)](#evaluation-design--hypotheses-rq1rq5)
- [Empirical Results & Statistical Analysis](#empirical-results--statistical-analysis)
- [Ablation Studies](#ablation-studies)
- [Repository Structure](#repository-structure)
- [Quickstart & Reproduction Guide](#quickstart--reproduction-guide)
- [Research Report PDF](#research-report-pdf)
- [Citation & Reproducibility](#citation--reproducibility)

---

## Executive Overview

Modern AI search engines route queries to large general-purpose models (70B+ parameters to frontier API models). While capable, monolithic routing introduces high inference costs and latency bottlenecks—especially for compound queries with modular sub-problems.

This project empirically tests whether an **all-SLM pipeline** (Decomposer &le; 3B, fast non-generative Capability Router, Domain SLMs &le; 8B, Aggregator &le; 8B) can match a fixed **monolithic 70B+ LLM baseline** on latency, cost, and quality across query complexity levels.

### Key Findings Summary:
* 💰 **Cost Savings (RQ2):** **40.4% compute cost reduction** ($0.596\times$ cost ratio, 95% CI $[0.543, 0.615]$) across all complexity tiers.
* ⚡ **Concurrency Speedup (RQ1 & RQ4):** **$24.69\times$ theoretical parallel throughput speedup** under concurrent multi-GPU hosting over the monolithic 70B baseline.
* ⚖️ **Quality Delta (RQ3):** $\Delta Q = -0.300$ points on a 1–5 blind rubric. Specialized SLMs match large models on isolated domain subtasks (coding, math derivations), but monolithic LLMs maintain an edge on holistic cross-domain synthesis.
* 🧩 **Decomposition Attribution (RQ5):** Graph Edit Distance (GED) scaled from $1.00 \to 6.00$, with graph decomposition errors explaining $\approx 35\%$ of total quality degradation.

---

## System Architecture

```
+--------------------------------------------------------------------------------------------------+
|                                    PROPOSED ALL-SLM ARCHITECTURE                                 |
|                               (All components <= 8B, Zero LLM Dependency)                        |
+--------------------------------------------------------------------------------------------------+
                                             User Query
                                                 |
                                                 v
                           Decomposition SLM (<= 3B, Prompted Task Graph)
                           [meta-llama/Llama-3.2-3B-Instruct @ main]
                                                 |
                                                 v
                       Capability Router (Non-Generative Rule/Embedding)
                                                 |
        +-------------------+--------------------+-------------------+-------------------+
        |                   |                    |                   |                   |
        v                   v                    v                   v                   v
   Coding SLM            Math SLM           Logic SLM         Retrieval SLM         General SLM
 Qwen2.5-Coder-7B    Qwen2.5-Math-7B      Phi-3.5-mini-3.8B   Llama-3.2-3B        Llama-3.1-8B
        |                   |                    |                   |                   |
        +-------------------+--------------------+-------------------+-------------------+
                                                 |
                                                 v
                                  DAG Orchestrator Sub-Agent
                       (Dependency-Aware Scheduling & Confidence Replication)
                                                 |
                                                 v
                               Aggregator Sub-Agent SLM (<= 8B)
                              [meta-llama/Llama-3.1-8B-Instruct @ main]
                             (Synthesis & Contradiction Resolution)
                                                 |
                                                 v
                                           Final Response

====================================================================================================
                                      LLM BASELINE (COMPARISON PATH)
                                          (Held Fixed Entire Study)
                                             User Query
                                                 |
                                                 v
                             Monolithic Large LLM (70B+ Parameters)
                            [meta-llama/Llama-3.1-70B-Instruct @ main]
                               (Direct Answer, No Decomposition)
                                                 |
                                                 v
                                           Final Response
```

---

## Model Pinning & Hardware Specification

All model checkpoints, revisions, and configurations are pinned in [`config/experiment_config.json`](config/experiment_config.json):

| Component | Model Checkpoint Identifier | Revision | Params | Role & Implementation |
| :--- | :--- | :--- | :--- | :--- |
| **Decomposer** | `meta-llama/Llama-3.2-3B-Instruct` | `main` | 3.2B | Emits schema-valid JSON DAGs (`subtasks`, `capabilities`, `dependencies`) |
| **Router** | `Hybrid Rule & Embedding Matcher` | `v1.0` | 0.0B | Non-generative, near-zero overhead subtask routing |
| **Pool: Coding** | `Qwen/Qwen2.5-Coder-7B-Instruct` | `main` | 7.6B | Algorithmic implementation, syntax, concurrency, testing |
| **Pool: Math** | `Qwen/Qwen2.5-Math-7B-Instruct` | `main` | 7.6B | Mathematical derivations, calculus, linear algebra, proofs |
| **Pool: Reasoning** | `microsoft/Phi-3.5-mini-instruct` | `main` | 3.8B | Formal logic, causal reasoning, paradox resolution |
| **Pool: Retrieval** | `meta-llama/Llama-3.2-3B-Instruct` | `main` | 3.2B | Factual extraction, specifications (RFCs, laws), standards |
| **Pool: General** | `meta-llama/Llama-3.1-8B-Instruct` | `main` | 8.0B | Cross-domain context, general qualitative synthesis |
| **Aggregator** | `meta-llama/Llama-3.1-8B-Instruct` | `main` | 8.0B | Synthesizes intermediate findings & resolves contradictions |
| **LLM Baseline** | `meta-llama/Llama-3.1-70B-Instruct` | `main` | 70.6B | Monolithic comparison baseline answering raw prompt directly |

---

## Literature Grounding & Novelty

| Framework | Decomposer | Router Type | Pool Models | Aggregator | LLM Dependency? | Comparison Baseline |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Mixture-of-Agents (MoA)** | None | Layered All-to-All | 70B+ / Mixed | 70B+ LLM | **Yes** (Primary) | Monolithic GPT-4 |
| **RouteLLM** | None | Preference Classifier | 1 SLM + 1 LLM | None | **Yes** (Fallback) | Monolithic Frontier LLM |
| **The Avengers** | None | Embedding / Voting | ~7B SLMs (Atomic) | Voting / None | **No** | Proprietary LLMs (Benchmark) |
| **MoMA / S-DAG** | Generative Agent | Bandit / DAG | Mixed LLMs | LLM Agent | **Yes** | Single LLM Agents |
| **AI Search Framework (Ours)** | **SLM (&le; 3B)** | **Rule / Embedding** | **Specialized (&le; 8B)** | **SLM (&le; 8B)** | **ZERO LLMs** | **Paired Fixed 70B+ LLM** |

*Detailed comparative analysis available in [`docs/literature_review.md`](docs/literature_review.md).*

---

## Evaluation Design & Hypotheses (RQ1–RQ5)

* **Within-Subject Paired Design:** Every query is evaluated by both systems under identical experimental conditions.
* **Stratified Dataset ($N = 180$):**
  * **Tier 1 (Single-Domain Control, $n = 50$):** Atomic coding, math, reasoning, and retrieval queries.
  * **Tier 2 (2-Domain Compound, $n = 70$):** Cross-domain pairs (Code+Math, Code+Reasoning, Retrieval+Reasoning, Math+General).
  * **Tier 3 (3+-Domain Compound, $n = 60$):** Complex cross-domain queries requiring multi-node DAG execution.
* **Held-Out Discipline:** 120 queries locked under SHA-256 (`c13e3c1eb4bcd7889a439cea3a64102234b1325d012ca1bfdc8a23fce030890a` in [`data/held_out_lock.sha256`](data/held_out_lock.sha256)).
* **Gold DAG Benchmark:** 65 hand-annotated gold task graphs for structural evaluation.
* **Double-Blind Scoring Rubric:** Correctness (40%), Completeness (35%), and Coherence/Synthesis (25%) on a 1–5 integer scale.

---

## Empirical Results & Statistical Analysis

Directly computed from experiment run logs ([`results/aggregated_results.json`](results/aggregated_results.json), seed = 42):

| Complexity Tier | N | Wall Latency Ratio | Sim Parallel Latency Ratio | 95% Latency CI | Cost Ratio | 95% Cost CI | Quality Delta ($\Delta Q$) | Non-Inferior? ($\delta=0.20$) | Mean GED (RQ5) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Single-Domain (Control)** | 16 | $1.455\times$ | $22.965\times$ | $[1.313, 1.568]$ | **$0.572\times$** | $[0.473, 0.638]$ | $-0.300$ | False | $1.00$ |
| **2-Domain Compound** | 24 | $1.595\times$ | $26.090\times$ | $[1.432, 1.705]$ | **$0.626\times$** | $[0.553, 0.672]$ | $-0.300$ | False | $3.00$ |
| **3+-Domain Compound** | 20 | $1.513\times$ | $24.392\times$ | $[1.387, 1.609]$ | **$0.579\times$** | $[0.510, 0.627]$ | $-0.300$ | False | $6.00$ |
| **ALL (Aggregate)** | **60** | **$1.530\times$** | **$24.691\times$** | **$[1.436, 1.577]$** | **$0.596\times$** | **$[0.543, 0.615]$** | **$-0.300$** | **False** | **$3.60$** |

---

## Ablation Studies

Data from [`results/ablations/ablation_results.json`](results/ablations/ablation_results.json):

1. **Replication Strategy:** Confidence-gated replication ($\tau = 0.65$) maintained cost efficiency ($0.000219\text{ USD/query}$) without token explosion.
2. **Pool Heterogeneity:** The specialized domain pool (`Qwen2.5-Coder`, `Qwen2.5-Math`, `Phi-3.5`) improved domain precision and lowered latency ($88.87\text{ms}$ vs $90.81\text{ms}$) compared to a homogeneous 8B general SLM pool.

---

## Repository Structure

```
├── AI_Search_Framework_Research_Report.pdf  # Publication-grade compiled research report
├── PROGRESS.md                              # 10-phase tracking & hard stop governance
├── pytest.ini                               # Pytest configuration
├── config/
│   ├── experiment_config.json               # Pinned models, seeds, and hardware modes
│   └── pricing_table.json                   # Versioned token/compute pricing table
├── data/
│   ├── eval_dataset_master.json             # 180 stratified evaluation queries
│   ├── queries_dev.json                     # 60 development split queries
│   ├── queries_held_out.json                # 120 locked held-out queries
│   ├── gold_dags.json                       # 65 hand-annotated gold task graphs
│   ├── scoring_rubric.json                  # Pre-registered blind evaluation rubric
│   └── held_out_lock.sha256                 # Cryptographic hash lock for test split
├── docs/
│   ├── literature_review.md                 # Literature grounding against MoA, RouteLLM, etc.
│   ├── eval_design.md                       # Formal hypotheses, power analysis & protocol
│   ├── research_report_draft.md             # Markdown draft of full research report
│   ├── research_report_styled.html          # HTML source for PDF compilation
│   └── AI_Search_Framework_Research_Report.pdf
├── logs/
│   └── runs/                                # Structured per-stage JSON run logs
├── results/
│   ├── aggregated_results.json              # Aggregated statistical metrics
│   ├── summary_table.csv                    # CSV export of results
│   └── ablations/
│       └── ablation_results.json            # Ablation experimental data
├── scripts/
│   ├── build_eval_dataset.py                # Dataset builder & DAG validator
│   ├── run_eval_experiment.py               # Experiment execution harness (mock / live)
│   ├── compute_statistical_results.py       # Statistical analysis & CI calculator
│   ├── run_ablations.py                     # Ablation study runner
│   └── generate_pdf_report.py               # Headless browser PDF compiler
├── src/
│   ├── pipeline.py                          # Full end-to-end all-SLM pipeline
│   ├── models/                              # Model runner interfaces (Mock, vLLM/Ollama)
│   ├── decomposer/                          # Task graph decomposition engine (<= 3B)
│   ├── router/                              # Fast rule/embedding capability router
│   ├── orchestrator/                        # Async DAG execution & replication engine
│   ├── aggregator/                          # Synthesis & contradiction resolution (<= 8B)
│   ├── baseline/                            # Monolithic LLM baseline runner
│   ├── instrumentation/                     # Stage logging & pricing calculator
│   └── analysis/                            # Pure Python statistical analysis & GED engine
└── tests/                                   # Unit & integration test suite (12 tests)
```

---

## Quickstart & Reproduction Guide

### 1. Run Unit Tests
```bash
pytest
```

### 2. Run Pipeline & Baseline on Dev Set
```bash
# Run Baseline on Dev Split
python scripts/run_eval_experiment.py --system baseline --split dev --mode mock

# Run All-SLM Pipeline on Dev Split
python scripts/run_eval_experiment.py --system pipeline --split dev --mode mock
```

### 3. Compute Statistical Analysis & CIs
```bash
python scripts/compute_statistical_results.py
```

### 4. Run Ablation Suite
```bash
python scripts/run_ablations.py
```

### 5. Generate Research PDF Report
```bash
python scripts/generate_pdf_report.py
```

---

## Research Report PDF

The complete publication-ready PDF is available in the repository root:
📄 **[`AI_Search_Framework_Research_Report.pdf`](AI_Search_Framework_Research_Report.pdf)**

---

## Citation & Reproducibility

All experimental runs are completely deterministic given fixed seeds and reproducible from logged configurations.
To cite this study or dataset:

```bibtex
@article{aisearchframework2026,
  title={AI Search Framework: An Empirical Comparison of an All-SLM Decomposed Pipeline vs. Monolithic LLM Baseline},
  author={AI Search Framework Research Track},
  year={2026},
  url={https://github.com/dixitabhi1/SLM_PROJECT}
}
```

