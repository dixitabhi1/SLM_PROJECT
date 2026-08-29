---
name: baseline-model-runner
description: Use when setting up, configuring, or running the LLM baseline (the single large model that answers queries directly with no decomposition, used as the comparison point for the whole study). Also use if asked to change, swap, or add a second baseline model mid-project.
---

# LLM Baseline Runner

Source of truth: `.agents/knowledge/TRD_source.txt` §4 and §6, and
`.agents/knowledge/Implementation_Plan_source.txt` Phase 5.

## What the baseline is (and isn't)

A single large model (frontier-class API model, e.g. current GPT/
Claude/Gemini-class, OR a large open-weight model such as
Llama-3.1-70B/405B or Qwen2.5-72B) answering the full query directly,
with no decomposition. It is **not part of the proposed architecture**
— it exists purely to measure what the all-SLM system gives up or
gains relative to it.

## Two options, pick one, hold it fixed

| Option | Trade-off (TRD §4) |
|---|---|
| Frontier-class API model | Highest quality bar, but API latency is confounded by network/provider load — control by running many trials and reporting variance, or prefer the self-hosted option |
| Large self-hosted open-weight model (Llama-3.1-70B/405B, Qwen2.5-72B) | Preferred for latency measurement — same hardware class as the SLM pool's hosting, removes the network confound, reproducible end-to-end |

**Once chosen, the model/checkpoint is fixed for the entire experiment.**
Switching baseline models mid-study invalidates the paired comparison
(TRD §4). If a task would change the baseline model or checkpoint after
Phase 5 has produced any logged runs, stop and flag this explicitly
instead of just doing it — this is a project-breaking change, not a
routine config edit.

## Non-functional requirement

Not part of the proposed system; fixed model/checkpoint for every run
in the experiment (TRD §2 component table). Record the exact model
name + checkpoint/revision + hosting method (API vs. self-hosted, and
if API, provider + endpoint) in the run config every single run, not
just once at setup.

## Implementation notes

- Same instrumentation requirements apply as the pipeline: wall-clock
  latency, token counts, cost from the fixed pricing/compute-cost
  table, logged per run (see `experiment-instrumentation` skill).
- Run on the *same queries, same stratification* as the pipeline —
  this is a within-subject paired design (TRD §7), so baseline and
  pipeline runs must share query IDs to be paired correctly in
  analysis.
- If self-hosting: same hardware class as the SLM pool where possible,
  to keep the latency comparison honest (TRD §6).
- If using an API: run many trials per query and report variance
  explicitly — don't report a single-trial latency number as if it
  were representative (TRD §8 threats-to-validity table).
