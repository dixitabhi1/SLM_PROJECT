# Pathway B Pre-Flight Experiment Plan: 7B–8B SLM Scaling Investigation

**Status:** DRAFT / PENDING MENTOR AUTHORIZATION (Hard Stop)  
**Author:** AI Search Framework Research Agent  
**Date:** September 21, 2026  
**Repository Constraints:** Extends candidate pool from $\le 5\text{B}$ ceiling to $\le 8\text{B}$ ceiling (TRD §3 alignment) upon formal authorization.

---

## 1. Executive Motivation & Core Hypothesis

### 1.1 Empirical Context from v3 through Step B
Across Steps 1–6, Step A, and Step B, small language models constrained to $\le 5\text{B}$ parameters (`phi3.5:3.8b`, `llama3.2:3b`, `qwen2.5:3b`) demonstrated an empirical **0.0% win rate** against functioning $\ge 32\text{B}$ frontier baselines on compound multi-domain engineering tasks.
- External deterministic tools (retrieval grounding, AST parsing, sandboxed execution retry loops) improved syntactic hygiene and yielded an audited **+0.166 pt (+3.3%)** gain on clean coding execution (`V3_CD_01`).
- However, 100% of executable Python scripts in Step 6 and Step B required substitution from reference templates because the 3.82B coding specialist failed execution de novo on 4/4 attempts.
- Furthermore, the surrounding mathematical formulations and cross-domain synthesis contained fatal conceptual confabulations that external tools could not remediate.

### 1.2 Core Hypothesis of Pathway B
**Hypothesis H_B:** Increasing the specialist and aggregator parametric capacity from $\sim 3.8\text{B}$ to the authorized $\le 8\text{B}$ ceiling (`Qwen2.5-Coder-7B`, `Qwen2.5-Math-7B`, `Llama-3.1-8B-Instruct`) will provide sufficient non-linear reasoning depth to:
1. Generate syntactically valid and mathematically correct coupled engineering code *without* triggering template fallback ($>50\%$ autonomous pass rate on mechanical verification).
2. Bridge the quality gap against $\sim 32\text{B}$ monolithic baselines, achieving a non-zero audited win rate ($>0.0\%$) on compound multi-domain tasks under symmetrical double-blind evaluation.

### 1.3 Mandatory Inheritance of Session Hardening Fixes
Pathway B explicitly inherits and retains every mechanical, harness, and algorithmic hardening fix established and audited during previous phases:
- **Judge Harness Trim-Truncation Fix (`pairwise_harness.py`):** Expanded `max_chars = 11000` to prevent severing code blocks and ensure total judge payload remains safely under Groq's 7,000 ITPM ceiling. Middle-truncation is replaced with tail truncation at natural section boundaries, and all trimming events log active warnings.
- **Parse-vs-Execute Mechanical Verification Gate (`MechanicalCodeVerifier`):** Cleanly decouples static AST parsing (`ast.parse`) from sandboxed runtime execution (`subprocess.run`), preventing uncaught runtime exceptions from crashing the pipeline runner and enforcing an 8.0s timeout ceiling.
- **Multi-Turn Execution Feedback Loop (`GroundedVerifiedCodingModelRunner`):** Implements an iterative retry loop (up to 4 attempts) where actual compiler tracebacks and runtime error exceptions are formatted and fed directly back into subsequent prompt turns for self-repair.
- **Edge-Case & Degradation Handling:**
  - *Empty Response Handling:* Traps empty model completions or null text immediately, substituting diagnostic fallbacks instead of propagating empty strings into the verification gate.
  - *Commentary-Only Handling:* Traps responses that produce conversational prose without an executable ` ```python ... ``` ` code block, rejecting commentary and prompting for formatted code.
  - *Repetition-Loop Detection & Temperature Nudge:* Tracks repeated error traces across consecutive turns, applying a bounded temperature nudge ($0.0 \to 0.4$) on duplicate errors to break sampling stagnation, while terminating runaway generation loops early.

---

## 2. Candidate Model Roster & Hardware Feasibility

### 2.1 Pinned Model Architecture ($\le 8\text{B}$, Zero LLMs)

| Pipeline Role | Pinned Model & Checkpoint | Parameter Count | Primary Host Target | Fallback Host Target |
| :--- | :--- | :---: | :--- | :--- |
| **Decomposition** | `meta-llama/Llama-3.2-3B-Instruct` | 3.21B | Local Ollama (`llama3.2:3b`) | Fixed $\le 3\text{B}$ per TRD §2 |
| **Coding Specialist** | `Qwen/Qwen2.5-Coder-7B-Instruct` | 7.61B | Local Ollama (`qwen2.5-coder:7b`) | OpenRouter API / vLLM |
| **Math Specialist** | `Qwen/Qwen2.5-Math-7B-Instruct` | 7.61B | Local Ollama (`qwen2.5-math:7b`) | OpenRouter API / vLLM |
| **Reasoning Specialist** | `Qwen/Qwen2.5-7B-Instruct` | 7.61B | Local Ollama (`qwen2.5:7b`) | OpenRouter API / vLLM |
| **Retrieval QA Specialist**| `Qwen/Qwen2.5-7B-Instruct` + RFC Tool | 7.61B | Local Ollama (`qwen2.5:7b`) | OpenRouter API / vLLM |
| **Aggregator** | `meta-llama/Llama-3.1-8B-Instruct` | 8.03B | Local Ollama (`llama3.1:8b`) | Groq LPU (`llama-3.1-8b-instant`) |

*Note on Hard Rule 13 Compliance:* All pipeline models are $\le 8.03\text{B}$. The comparative baselines remain strictly $\ge 20\text{B}$ (`openai/gpt-oss-20b`, `gemini-2.5-flash`, `openai/gpt-oss-120b`). No candidate pipeline model overlaps with any comparative baseline endpoint.

### 2.2 Local Host Feasibility (Host: 16 GB RAM + RTX 3050 6GB VRAM)
- A 7B–8B model quantized at Q4_K_M requires $\sim 4.7\text{ GB}$ to $5.2\text{ GB}$ of memory.
- With sequential specialist execution (orchestrator invokes one specialist at a time and unloads previous weights), a 7B model fits entirely within 16 GB system RAM with partial GPU layer offloading (12–16 layers in 4 GB VRAM, remaining in system RAM).
- Local execution is 100% feasible on current host hardware with zero API expense.

---

## 3. Compute Budget & Cost Estimate

### Scenario 1: Pure Local Execution (Recommended Default)
- Specialist Inference: Local Ollama on host CPU/GPU (`qwen2.5-coder:7b`, `llama3.1:8b`).
- Pairwise Judge: `qwen/qwen3.8-27b` on Groq Cloud free tier (paged with 6,000-token payload guardrails).
- **Total Financial Cost: $0.00**.

### Scenario 2: Cloud-Accelerated Inference (If Faster Latency Required)
- OpenRouter API pricing for 7B/8B models:
  - `qwen/qwen-2.5-coder-7b-instruct`: \$0.06 / 1M prompt tokens, \$0.06 / 1M completion tokens.
  - `meta-llama/llama-3.1-8b-instruct`: \$0.055 / 1M prompt tokens, \$0.055 / 1M completion tokens.
- Pilot Scope (Target Pair `V3_CD_01` + `V3_CD_41`):
  - Total tokens generated across specialists and aggregator: $\sim 60,000\text{ tokens}$.
  - Cloud inference cost: **$0.01**.
- Extended Scope (8 Compound & Two-Domain Queries):
  - Total tokens: $\sim 350,000\text{ tokens}$.
  - Cloud inference cost: **$0.04 – $0.06**.

---

## 4. Staged Isolated Test Design

To prevent compounding architectural confounds, Pathway B will proceed in four strictly isolated stages under the **Autonomous Audit Loop Protocol**:

### Stage 1: Coding Specialist Mechanical Autonomy Gate (Isolation Test)
- **Target Subtasks:** `V3_CD_01_NODE3` (Augmented Lagrangian MDO solver) and `V3_CD_41_NODE2` (2D Laplacian Kronecker stencil).
- **Input:** Up to 4 execution attempts with subprocess error traceback injection.
- **Success Criterion:** `Qwen2.5-Coder-7B-Instruct` must achieve AST validity and successful subprocess execution *without* triggering `FALLBACK_TO_TEMPLATE`.
- **Pass Threshold:** $\ge 1 / 2$ tasks pass autonomously (breaking the 100% fallback dependence observed in 3.8B models).

### Stage 2: Specialist Theoretical Rigor Isolation (Math & Systems)
- **Target Tasks:** `V3_CD_01_NODE1` (primal-dual Augmented Lagrangian derivation) and `V3_CD_41_NODE1` (continuous heat equation PDE formulation).
- **Evaluation:** Direct inspection of generated formulations against reference standards (AIAA MDO, Crank-Nicolson stability criteria).
- **Success Criterion:** Elimination of dimensional mismatches, scalar-vector confusion, and toy 1D ODE substitutions.

### Stage 3: Target Pair Pairwise Benchmark (`V3_CD_01` + `V3_CD_41`)
- **Systems Compared:** `SLMPipeline_v5_8B` vs 3-Tier Baseline Ladder (~20B, ~32B, 120B).
- **Protocol:** 12 verified trials evaluated symmetrically (forward and swapped) via `qwen/qwen3.8-27b` on Groq LPU.
- **Autonomous Audit Loop Execution:**
  1. Raw-file concordance check on every trial.
  2. General Rule truncation diagnostic (any win against a truncated baseline is voided by default).
  3. Strict no-blending enforcement (separate reporting per query).
  4. Temperature confound check (assert 100% at temperature=0.0).
  5. Positional swap agreement logging.
  6. Transparent reconciliation against stated rules.

### Stage 4: Scaling Boundary Analysis
- If Stage 3 demonstrates clean, un-truncated wins against ~32B baselines, expand benchmark to all 8 development compound and two-domain queries (`results/v5_8b_extended/`).
- If Stage 3 results in 0.0% win rate against un-truncated 32B baselines, formally establish that the parametric capacity ceiling extends through 8B parameters.

---

## 5. Hard Stops & Governance Checklist

- [ ] **Hard Stop 1:** Explicit mentor authorization to relax the $\le 5\text{B}$ constraint and test 7B–8B models.
- [ ] **Hard Stop 2:** Verification that candidate models are pinned and distinct under Hard Rule 13.
- [ ] **Hard Stop 3:** Confirmation that held-out split (`data/v3_queries_held_out.json`, SHA256 `c15452b4...`) remains unread and locked.

