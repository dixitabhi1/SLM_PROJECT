---
name: anti-hallucination-guardrails
description: Use whenever the agent is about to state a number (latency, cost, quality score, CI, p-value, accuracy %), name a model/checkpoint, describe the system architecture, or summarize a research question/metric/deliverable for the AI Search Framework project. Also use before writing any part of the report/write-up. Ensures every claim traces to either a logged experiment run or the PRD/TRD/Implementation Plan source docs, never to memory or a plausible guess.
---

# Anti-Hallucination Guardrails

This project is a research artifact meant to survive an adversarial
technical read (see PRD §6). The single biggest risk to that is an
agent stating something that *sounds* right but isn't traceable to a
real source. This skill is a checklist to run before any output that
contains a claim, number, or spec detail.

## Before stating any number

Ask: **where did this number come from?**

- If it's a measured result (latency, cost, quality score, decomposition
  accuracy, CI width, p-value) → it must come from a file under
  `results/` or `logs/` that was produced by an actual run in this repo.
  Cite the file path and run ID inline, e.g. "3.4x latency reduction on
  the 2-domain bucket (`results/run_2026-08-20_slm.json`, n=62)."
- If no such file exists yet → write `TBD — pending Phase <N> run`.
  Never fill the gap with a number that "seems about right," and never
  quote the PRD's ~3–6x latency / ~5–10x cost figures as if they were
  measured — those are explicitly *planning targets*, not results (PRD
  §5, "Expectation-setting").
- Never average, round, or cherry-pick a subset of buckets to make a
  headline number look better than the per-bucket data supports. TRD
  §8 and the Implementation Plan's Key Risks both require per-bucket
  reporting, especially where the LLM baseline wins.

## Before describing the architecture

- Re-check `.agents/knowledge/TRD_source.txt` §1–3 rather than
  reconstructing the pipeline from memory of "how RAG/agent pipelines
  usually work." The component set is fixed: Decomposition SLM (≤3B) →
  Capability Router (rule-based/embedding, non-generative) → SLM Pool
  (3–5 models, coding/math/reasoning/general, all ≤8B) → Orchestrator
  → Aggregator (≤8B) → Final Response. The LLM baseline is a *separate*
  path, not a component of the pipeline.
- If asked to "improve" the pipeline by adding a bigger model or an
  LLM call anywhere inside it, flag that this would violate the TRD's
  zero-LLM constraint (TRD §0) rather than silently doing it.

## Before naming a model

- Every model reference needs an exact identifier + checkpoint/revision,
  not a family name. Check `.agents/knowledge/TRD_source.txt` §3–4 for
  the candidate pools; if a specific model isn't listed there, say so
  and ask, don't substitute a similar-sounding one silently.
- The baseline model, once chosen, is fixed for the entire experiment
  (TRD §4, Implementation Plan Phase 5). If code or config would change
  it mid-project, stop and flag this as invalidating the paired
  comparison rather than proceeding.

## Before summarizing methodology, metrics, or deliverables

- Quote/paraphrase from `.agents/knowledge/PRD_source.txt` or
  `TRD_source.txt` directly rather than from general knowledge of how
  "eval frameworks usually work." This project has specific choices
  (within-subject paired design, blind rubric scoring, non-inferiority
  as the primary quality claim, graph-edit-distance for decomposition
  accuracy) that a generic answer would miss or contradict.

## When uncertain

State the uncertainty explicitly in the output ("I don't see a decision
recorded on X — checked TRD §4 and PRD §5, it's not specified") rather
than picking a reasonable-sounding default and presenting it as settled.
