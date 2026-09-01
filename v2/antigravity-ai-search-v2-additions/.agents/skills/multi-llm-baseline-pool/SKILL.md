---
name: multi-llm-baseline-pool
description: Use when setting up, configuring, or running the v2 baseline comparison — 4-5 LLMs of varying parameter sizes, replacing v1's single fixed 70B baseline. Use before running any baseline call, and before writing any per-baseline-model results table.
---

# Multi-LLM Baseline Pool (v2)

Extends `baseline-model-runner` (v1 skill, still valid for the pinning
discipline it describes) from one fixed baseline to a fixed ROSTER of
4-5 baselines spanning a parameter range.

## What changes from v1

v1 held ONE model fixed (Llama-3.1-70B-Instruct) for the whole study.
v2 holds a ROSTER of 4-5 models fixed for the whole study — same
discipline, more models. Every rule from v1's baseline-model-runner
still applies (pin exact checkpoint/revision, no swapping mid-study,
run on the same queries as the SLM pipeline) — applied per-model in
the roster now, not once.

## Choosing the roster

Pick 4-5 models spanning distinct parameter tiers so the results can
show a genuine scale trend, not five similar-sized models. A reasonable
spread (confirm actual choices with the user, don't assume this exact
list):
- small (~7-8B) — same class as the SLM pool components, useful to see
  if "large but not decomposed" alone helps
- mid (~13-34B)
- large (~70B) — reuse v1's exact baseline here for continuity with v1
  results if possible (same model = a v1-to-v2 comparison point)
- frontier-class (proprietary API, e.g. current GPT/Claude/Gemini-class)
- optionally a second frontier or a very large open-weight model
  (405B-class) if budget allows

**Every model in the roster needs the same pin discipline as v1**:
exact name + checkpoint/revision (or API model version string) +
hosting method, recorded in run config. If a task would swap a roster
member after any real run has been logged against it, stop and flag —
this breaks the paired comparison the same way it would for a
single-baseline study.

## What this changes downstream

- **Instrumentation**: every run log needs a `baseline_model_id` field
  identifying which of the 4-5 baselines produced that particular
  baseline response — this is a NEW required field vs. v1's schema
  (v1 didn't need it, there was only one baseline).
- **Statistics**: RQ1/RQ2/RQ3 results tables become per-baseline-model
  (5 rows) x per-complexity-tier, not one row per tier like v1's Table
  4. See `statistical-analysis` skill for the updated reporting shape.
- **Judge**: the LLM-judge (see `llm-judge-blind-eval`) now picks
  from up to 6 candidate responses per query (1 SLM pipeline + up to 5
  baselines), not a 2-way comparison — the judge skill's shuffling and
  logging logic depends on knowing the roster size.

## Running discipline

- Run all 4-5 baselines on the exact same query set and exact same
  stratification as the SLM pipeline (within-subject paired design,
  unchanged from v1).
- If any roster member is API-based, run multiple trials and report
  variance per TRD §4/§8 — don't let one flaky API call become "the"
  latency number for that model.
- Cost table (`config/pricing_table.json`) needs an entry per roster
  model — if a model's real-world pricing isn't public (e.g. some
  self-hosted large models), use compute-cost estimation and label it
  as such rather than leaving it blank or guessing a number that looks
  like real pricing.
