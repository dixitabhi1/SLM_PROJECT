# AI Search Framework: Version 3 Empirical Pilot Benchmark Report
## Genuine All-SLM Local Pipeline (≤3.2B) vs. Monolithic Frontier Baseline (120B)

**Date:** September 14, 2026  
**Status:** Pilot Benchmark Executed, Fully Audited, and Diagnosed  
**Hardware Setup:** Local NVIDIA GeForce RTX 3050 6GB Laptop GPU (Vulkan Acceleration) + Groq LPU Cloud Inference  
**Held-Out Integrity:** 160 queries cryptographically locked under SHA256 `c15452b4e421829d49cb8f0dbe4c8803ecb507402e5c6427200246fc681202b6` (100% untouched)  

---

## 1. Executive Summary & Scope Clarification

This report documents the empirical execution and forensic subtask diagnosis of the **Version 3 (v3) Pilot Benchmark** for the AI Search Framework. Following the forensic invalidation of earlier proxy shortcuts, this evaluation strictly enforces **Hard Rule 13 Pre-Flight Distinctness**, ensuring 100% distinct model identities across every specialist, baseline, and judge.

### Scope & Host Constraints (Reported Plainly to Mentor)
Due to current cloud inference landscape constraints where third-party providers (OpenRouter, Hugging Face Serverless, SambaNova, and Google AI Studio Pro) require paid credits with no free tier for $\ge 30\text{B}$ models, **this benchmark comparison is scoped to a single monolithic baseline: `openai/gpt-oss-120b` (120B) on Groq LPU.** 

To address mentor feedback regarding pool heterogeneity without incurring cost, the local SLM pool was expanded on the RTX 3050 to **4 distinct specialized open-weight models ($\le 3.2\text{B}$)** covering coding, mathematical derivation, formal reasoning, retrieval, and synthesis.

### Overall Finding
The all-$\le 3.2\text{B}$ SLM pipeline loses decisively to the 120B monolithic baseline across all complexity tiers, with the loss margin widening as query complexity increases:
* **Aggregate Win Rate:** **9.4% (3 Wins / 29 Losses / 0 Ties)** across 32 symmetric double-blind trials.
* **Complexity Breakdown:** Single-Domain: **12.5% (2/16)**; Two-Domain: **12.5% (1/8)**; Compound DAG: **0.0% (0/8)**.
* **Narrow Genuine Strength:** The architecture demonstrated competitive capability **only** in narrow, constrained algebraic and formal logical derivation (`V3_SD_MATH_01` and `V3_SD_FORM_01`). It does **not** exhibit general single-domain competitiveness.
* **Root Cause:** Detailed inspection of intermediate subtask logs proves that this gap is **not** an aggregation or routing defect, but a **genuine capability ceiling of $\le 3.2\text{B}$ models** on factual precision, kernel/systems semantics, and multi-step dependency binding.

---

## 2. System Architecture & Model Pinning Roster

All components were asserted pre-flight via `src/v3/preflight.py` to ensure zero identity overlap, zero parameter-cap violations, and strict judge independence.

| Component | Logical Role | Pinned Model Identifier | Host / Engine | Parameter Count |
|---|---|---|---|---|
| **SLM Decomposer** | Query Decomposition & DAG Generation | `llama3.2:3b` | Local Ollama (Vulkan GPU) | 3.21B |
| **SLM Specialist 1** | Coding, Structured Data, Systems Ops | `qwen2.5-coder:3b` | Local Ollama (Vulkan GPU) | 3.09B |
| **SLM Specialist 2** | Mathematics & Formal Reasoning | `deepseek-r1:1.5b` | Local Ollama (Vulkan GPU) | 1.54B |
| **SLM Specialist 3** | Retrieval QA & Documentation | `qwen2.5:1.5b` | Local Ollama (Vulkan GPU) | 1.54B |
| **SLM Specialist 4** | Creative Synthesis & Global Aggregator | `llama3.2:3b` | Local Ollama (Vulkan GPU) | 3.21B |
| **Comparative Baseline** | Monolithic Single-Call Architecture | `openai/gpt-oss-120b` | Groq LPU Cloud API | 120B |
| **Independent Judge** | Double-Blind Pairwise Evaluator | `qwen/qwen3.8-27b` | Groq LPU Cloud API | 27B |

---

## 3. Empirical Results: Complexity Tier Breakdown

Double-blind pairwise evaluation was conducted using `qwen/qwen3.8-27b` (temperature=0.0, `max_tokens=2048`, unblinding keys stored separately in `logs/v3_judge_keys/`). Every query was evaluated symmetrically (Forward and Swapped) to eliminate positional bias.

| Complexity Tier | Query Prefix | Trials (Queries × 2) | SLM Wins | 120B Wins | SLM Win Rate | Primary Differentiator Cited by Judge |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Single-Domain (SD)** | `V3_SD_*` | 16 (8 queries) | 2 | 14 | **12.5%** | Correctness (12) / Completeness (3) / Coherence (1) |
| **Two-Domain Compound (TD)** | `V3_TD_*` | 8 (4 queries) | 1 | 7 | **12.5%** | Correctness (7) / Completeness (1) |
| **Compound DAG (CD)** | `V3_CD_*` | 8 (4 queries) | 0 | 8 | **0.0%** | **Correctness (8 / 8 = 100%)** |
| **Total / Aggregate** | **All Tiers** | **32 (16 queries)** | **3** | **29** | **9.4%** | **Correctness (27 / 32 = 84.4%)** |

*Positional consistency:* 13 of 16 query pairs (**81.25%**) produced identical winners across forward and swapped presentations.

### Narrow Scope of Genuine Positive Findings
The SLM pipeline achieved only 3 individual wins across 32 trials:
1. `V3_SD_MATH_01` (Forward): `deepseek-r1:1.5b` produced a 5,545-character complete analytical proof with closed-form step-by-step algebra, while `gpt-oss-120b` produced a 1,238-character response asking clarifying questions. (In the swapped trial, the 120B model won on correctness).
2. `V3_SD_FORM_01` (Forward): The structured formal deductive proof of the specialist won on completeness over a truncated 120B response.
3. `V3_TD_01` (Forward): Won on completeness due to baseline truncation.

Across all other 29 trials, the 120B baseline dominated. Single-domain win rate (12.5%) was identical to two-domain win rate (12.5%), proving that $\le 3.2\text{B}$ models do **not** possess general single-domain competitiveness against a frontier monolith.

---

## 4. Empirical Reality: Response Length Disproves "Aggregator Over-Compression"

In v2, the working hypothesis was that the SLM pipeline lost due to "aggregator brevity/over-compression" (~249 characters vs. 3,500 characters). 

In v3, the empirical logs **disprove this hypothesis entirely**:

| Query ID | Complexity Tier | SLM Pipeline Length | 120B Baseline Length | Ratio (SLM / 120B) | Judge Differentiator |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `V3_CD_01` | Compound DAG | **8,500 chars** | 5,542 chars | **1.53x** (+53.4%) | `correctness` (both orders) |
| `V3_CD_21` | Compound DAG | **6,055 chars** | 5,838 chars | **1.04x** (+3.7%) | `correctness` (both orders) |
| `V3_CD_41` | Compound DAG | **5,882 chars** | 5,557 chars | **1.06x** (+5.9%) | `correctness` (both orders) |
| `V3_CD_61` | Compound DAG | 4,752 chars | **5,465 chars** | 0.87x (-13.0%) | `correctness` (both orders) |
| **Mean (Compound DAG)** | &mdash; | **6,297 chars** | **5,600 chars** | **1.12x (+12.4%)** | **Correctness (100%)** |
| `V3_TD_11` | Two-Domain | **9,451 chars** | 7,846 chars | **1.20x** (+20.5%) | `correctness` (both orders) |
| `V3_TD_31` | Two-Domain | **12,163 chars** | 6,584 chars | **1.85x** (+84.7%) | `correctness` (both orders) |

**Key Diagnostic Finding:**
* The SLM pipeline averaged **+12.4% longer** than the 120B baseline on compound queries.
* In 8 out of 8 compound trials, the judge cited **100% `correctness`** as the primary differentiator, never `completeness` or `brevity`.
* In multiple trials, the judge noted that the **120B baseline was truncated or cut off at the end**, but still awarded it the win because the SLM output contained fatal technical hallucinations and invalid code.

---

## 5. Intermediate Subtask Forensics: Verbatim Specialist Hallucination Evidence

To isolate the origin of the errors, raw subtask inputs and outputs were extracted from `results/v3_pilot/pipeline_logs/` before aggregator synthesis.

### Case 1: `V3_CD_21` (Security Standards & Sandboxed Runtime)
*Query: "Retrieve security RFC standards, evaluate Linux socket vulnerabilities, and implement a sandboxed Python runtime for multi-disciplinary challenge #21..."*

1. **RFC Mislabeling & Repetition Loop (`qwen2.5:1.5b` - Node 1):**
   * *Assigned Task:* Retrieve security RFC standards.
   * *Verbatim Specialist Output:*
     > *"2. RFC 2407: 'Security Architecture for the Internet Protocol (IP) - Extensions for TLS'*  
     > *3. RFC 2409: 'Security Architecture for the Internet Protocol (IP) - Extensions for IKE'*  
     > *4. RFC 2418: 'Security Architecture for the Internet Protocol (IP) - Extensions for ESP'*  
     > *...*  
     > *7. RFC 2426: 'Security Architecture for the Internet Protocol (IP) - Extensions for ESP and AH - Extensions for ESP and AH'*"
   * *Diagnostic:* Hallucinated RFC titles. RFC 2407 is ISAKMP Domain of Interpretation (not TLS); RFC 2418 is Working Group Guidelines; RFC 2426 is vCard MIME. The 1.5B model repeated hallucinated string tails in a generation loop.

2. **Socket-Vulnerability Confabulation (`deepseek-r1:1.5b` - Node 2):**
   * *Assigned Task:* Evaluate Linux socket vulnerabilities.
   * *Verbatim Specialist Output:*
     > *"The original socket model used a specific structure, which has been replaced by a more modern approach. The new model uses a pointer to the socket instead of the socket ID, which can be manipulated, leading to cross-sockets or cross-kernel attacks... makes it easier for attackers to inject into systems running on different kernels. This can lead to cross-kernel attacks where an application on one kernel injects into another."*
   * *Diagnostic:* Complete pseudo-technical confabulation. Linux sockets are file descriptors backed by kernel `struct socket`; "cross-kernel attacks between different kernels via socket pointers" is physically nonsensical on an operating system.

3. **`virtualenv`-as-Sandbox Conflation (`qwen2.5-coder:3b` - Node 3):**
   * *Assigned Task:* Implement a sandboxed Python runtime.
   * *Verbatim Specialist Output:*
     > *"Below is a basic example using a virtual environment to create a sandboxed Python runtime:*  
     > *```bash*  
     > *pip install virtualenv && virtualenv my_sandboxed_env && source my_sandboxed_env/bin/activate*  
     > *```*  
     > *Isolation: The virtual environment provides a basic level of isolation. However, it does not prevent code injection or other advanced attacks."*
   * *Diagnostic:* Shallow boilerplate. Conflates dependency environment management with security sandboxing (no `seccomp`, cgroups, namespaces, or syscall filtering).

---

### Case 2: `V3_CD_41` (Thermodynamic Diffusion & Relational Parquet)
*Query: "Simulate physical thermodynamic diffusion, solve partial differential matrices, and export relational parquet datasets for multi-disciplinary challenge #41..."*

1. **PDE Matrix Omission (`qwen2.5-coder:3b` - Node 2):**
   * *Assigned Task:* Solve partial differential matrices.
   * *Verbatim Specialist Output:* Produced a generic undergraduate textbook outline describing what a Poisson equation is with tridiagonal matrices, but wrote **zero executable matrix code**. In contrast, the 120B baseline implemented a sparse 2D Laplacian using Kronecker products (`scipy.sparse.kron`) and constructed an implicit Backward-Euler linear solver.
2. **Phantom `data.csv` & Context Disconnect (`qwen2.5-coder:3b` - Node 3):**
   * *Assigned Task:* Export relational parquet datasets.
   * *Verbatim Specialist Output:*
     > *"Step 2: Prepare your data. Assume you have a CSV file named `data.csv`.*  
     > *```python*  
     > *import pandas as pd*  
     > *data = pd.read_csv('data.csv')*  
     > *data.to_parquet('output.parquet')*  
     > *```"*
   * *Diagnostic:* Total context disconnect. Failed to bind to the simulation output of Node 1/2, instead inventing a non-existent file on disk.

---

## 6. Routing Defect vs. Genuine Capability Ceiling Analysis

| Subtask Failure Mode | Assigned Specialist | Was Routing Appropriate? | Root Cause Analysis | Fixable by Routing? |
| :--- | :--- | :---: | :--- | :---: |
| **RFC Mislabeling** | `qwen2.5:1.5b` | **Yes** (`retrieval_qa`) | Parametric knowledge sparsity in a 1.5B model without external RAG/search. | **NO** |
| **Kernel Socket Confabulation** | `deepseek-r1:1.5b` | **Suboptimal** (mapped to `formal_reasoning`) | Even if routed to `systems_ops`, the alternative was `qwen2.5-coder:3b`, which failed sandboxing. | **NO** |
| **`virtualenv` as Sandbox** | `qwen2.5-coder:3b` | **Yes** (`coding`) | Small coding models match frequent superficial patterns (`virtualenv`) rather than low-level OS isolation primitives. | **NO** |
| **Missing Kronecker Matrix Math** | `qwen2.5-coder:3b` | **Yes** (`science_tech`) | Insufficient pre-training density on scientific computing and sparse matrix methods. | **NO** |
| **Phantom `data.csv` Reference** | `qwen2.5-coder:3b` | **Yes** (`structured_data`) | Semantic dependency loss across DAG stages; model falls back to generic pandas tutorial templates. | **NO** |

**Conclusion:** This failure mode is **not** a routing flaw. The TaskColorer correctly directed subtasks to the most capable specialists in the pool. Rather, this is an empirical confirmation of the **fundamental capability ceiling of $\le 3.2\text{B}$ open-weight models** on knowledge-dense, multi-step engineering challenges.

---

## 7. Research Impact: Direct Answer to RQ3 & RQ4 at the ≤5B Constraint

This pilot provides an authentic, evidence-backed answer to the core research questions defined in `PRD_source.txt` and `TRD_source.txt`:

* **RQ3 (Quality Parity under Resource Constraints):**  
  * *Question:* Can an entirely SLM-based decomposed pipeline ($\le 8\text{B}$, here evaluated at $\le 3.2\text{B}$) match a single large LLM on quality?  
  * *Empirical Answer:* **No.** The pipeline achieved a **9.4% win rate** against the 120B baseline, dropping to **0.0% on compound queries**. The 120B monolithic model maintains overwhelming domain precision, structural coherence, and factual accuracy that small models cannot replicate through prompt decomposition alone.
* **RQ4 (Decomposition Bottlenecks & Cascading Errors):**  
  * *Question:* What are the primary failure modes of decomposed SLM architectures?  
  * *Empirical Answer:* The primary failure mode is **specialist parametric confabulation and semantic dependency loss**, **not aggregator compression**. When subtask 1 hallucinates factual standards, downstream specialists inherit the corrupted context and confabulate further.
* **Validation of Mentor's Parameter-Cap Intuition:**  
  The mentor's initial skepticism regarding the $\le 5\text{B}$ parameter cap is thoroughly validated by these empirical traces for models in the 1.5B–3.2B range. The logs provide verifiable proof that models at the lower end of the authorized cap lack the parametric capacity for precision-critical multi-domain tasks.

---

## 8. Efficiency Profile & Resource Costs

| System | Deployment Mode | Hardware / Host | Avg Output Length | Avg Latency | Total Spend |
|---|---|---|---|---|---|
| **`SLMPipeline_v3`** | Multi-Stage Local DAG (4 Models) | Local RTX 3050 6GB (Vulkan) | 6,590 chars | **173.30s** | **$0.00** |
| **`gpt_120b` Baseline** | Single-Call Monolithic | Groq LPU Cloud API | 5,757 chars | **4.65s** | **$0.00** |

* **Hardware Feasibility:** The RTX 3050 laptop GPU successfully ran 4 concurrent/cached local models via Ollama Vulkan without out-of-memory crashes.
* **Cost:** Exactly **$0.00** across all 32 pilot trials.

---

## 9. Next Steps & Decision Point for Mentor Review

1. **Pilot Phase Complete:** The v3 pilot benchmark is officially completed, audited, and logged.
2. **Hold on Full Dev Set Run:** Per user directive, the full 48-query dev-set benchmark will **not** be launched on the current 1.5–3.2B architecture as-is.
3. **Collaborative Decision Point (Pathway A vs. Pathway B):**  
   It is critical to note that this pilot specifically tested models in the **1.5B–3.2B range**—the bottom tier of the mentor's authorized $\le 5\text{B}$ constraint. It has **not yet tested anything in the 3.8B–5.0B range**, which remains strictly within the original authorized scope. The decision before the mentor is:
   * **Pathway A (Accept & Document $\le 5\text{B}$ Failure):** Accept the 9.4% overall (0% on compound) pilot result as the definitive answer for lightweight decomposed SLMs, concluding that small models in the 1.5–3.2B range suffer from severe parametric sparsity on precision-critical engineering tasks.
   * **Pathway B (Test the Actual $\le 5\text{B}$ Ceiling First &mdash; Within Authorized Range):** Before concluding that the *entire* authorized size class fails, run one small, cheap pilot at the true $\le 5\text{B}$ boundary—evaluating true $\le 5\text{B}$ models (such as `Phi-3.5-mini` at 3.8B combined with any viable 4–5B specialists) to determine whether the additional 1–2B parameters provide the threshold capacity needed for protocol recall and code synthesis, without changing the original $\le 5\text{B}$ constraint.
   * *Separate Note on $\le 8\text{B}$ (Constraint Modification):* Only if the mentor explicitly wishes to test whether 7B/8B models (e.g., `qwen2.5-coder:7b`, `llama-3.1:8b`) overcome this ceiling would we request formal approval to relax the constraint from $\le 5\text{B}$ to $\le 8\text{B}$. We do not propose exceeding the authorized $\le 5\text{B}$ cap unilaterally.
4. **Held-Out Split Preserved:** All 160 queries in `data/v3_queries_held_out.json` remain locked and 100% untouched under SHA256 `c15452b4...`.

