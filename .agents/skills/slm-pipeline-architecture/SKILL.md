---
name: slm-pipeline-architecture
description: Use when building, modifying, or explaining the proposed all-SLM pipeline — the decomposition SLM, capability router, SLM pool, orchestrator sub-agent, or aggregator sub-agent. Also use when the user asks to add a component, change routing logic, or wire dependency-aware dispatch. Encodes the exact component contracts from the TRD so the agent doesn't drift toward a generic RAG/agent architecture.
---

# SLM Pipeline Architecture

Source of truth: `.agents/knowledge/TRD_source.txt` §1–3. Re-read it
before non-trivial changes — this summary is a quick-reference, not a
replacement.

## Fixed pipeline (every component ≤8B, zero LLMs)

```
User Query
  -> Decomposition SLM (<=3B, prompted)         # query -> task graph DAG
  -> Capability Router (rule-based/embedding)   # NOT a generative call
  -> SLM Pool (3-5 models, <=8B each, parallel)  # coding / math / reasoning / retrieval-aug / general
  -> Orchestrator Sub-Agent                     # dependency-aware dispatch, confidence-based replication
  -> Aggregator Sub-Agent (<=8B)                # fusion, explicit contradiction resolution
  -> Final Response
```

The LLM baseline (single large model, direct answer, no decomposition)
is a **separate, parallel path** run on the same queries for comparison
only — it is never a fallback or component inside this pipeline.

## Component contracts (do not relax these without flagging it)

- **Decomposition SLM**: ≤3B params. Must emit schema-valid JSON task
  graph (subtask text, capability tag, dependency edges) for ≥90% of
  eval queries. Deterministic given a fixed seed (reproducibility
  requirement — don't leave temperature/sampling non-deterministic in
  eval runs).
- **Capability Router**: maps each subtask to one or more pool SLMs
  with a confidence score. Must be rule-based or embedding-similarity —
  **not itself a generative model call**. If you're tempted to route
  with an LLM prompt "just for this," that violates the near-zero
  routing-overhead requirement in TRD §2.
- **SLM Pool**: 3–5 models, covering coding/math/reasoning/general at
  minimum, all ≤8B. Each model's version/checkpoint must be pinned and
  logged per experiment (see `experiment-instrumentation` skill).
  Candidates by capability are listed in TRD §3 — treat that table as a
  starting point for your own per-capability eval, not as the final
  answer (TRD explicitly says final selection should come from your own
  eval, not published benchmarks alone).
- **Orchestrator**: executes independent subtasks in parallel, respects
  DAG dependencies, replicates subtasks that fall below a confidence
  threshold. Must produce a full trace of dispatch order and
  replication decisions — this trace is a required log field, not
  optional debug output.
- **Aggregator**: ≤8B. Fuses subtask outputs into one answer and must
  **explicitly resolve contradictions rather than concatenate**. Log
  both aggregation input and output for post-hoc quality review.

## Parallelism honesty

TRD §6 requires stating clearly whether "parallel" execution is real
concurrent GPU execution or simulated/sequential-with-logged-timestamps
(if hardware can't host 3–5 SLMs concurrently). Never let code silently
run sequentially while comments/logs claim parallel execution — label
simulated parallel time as an estimate wherever it appears (Implementation
Plan, Key Risks).

## When implementing

1. Check which phase you're in (Implementation Plan §1) — don't build
   the full orchestrator before the eval dataset and gold DAGs exist to
   test decomposition against (Phase 3 precedes Phase 6).
2. Every stage boundary needs a timestamp hook for the instrumentation
   layer (TRD §1, "Instrumentation layer taps every stage") — wire this
   in from the start rather than retrofitting it.
3. If asked to add a feature not in the TRD's component table (TRD §2),
   flag that it's outside the current spec before implementing it.
