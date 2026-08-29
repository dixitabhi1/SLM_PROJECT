---
name: experiment-instrumentation
description: Use whenever writing or modifying code that runs a pipeline stage, a baseline call, or anything that produces a latency/cost/quality number. Ensures every run is logged with enough structure (config, seed, timestamps per stage, token counts, model checkpoints) to be independently reproduced and to satisfy the anti-hallucination rule that every reported number traces to a real logged run.
---

# Experiment Instrumentation

Source of truth: `.agents/knowledge/TRD_source.txt` §1, §2 ("Instrumentation"
row), and §6.

## Non-negotiable requirement

> Every experimental run reproducible from logged config + seed.
> (TRD §2, Instrumentation non-functional requirement)

No code that produces a number used anywhere downstream (a chat
message, a report table, a plot) is complete until it also writes a
structured log satisfying this.

## What every run log must contain

Per TRD §1 ("Instrumentation layer taps every stage of both paths") and
§2:

- **Run identity**: run ID, timestamp, git commit hash of the code that
  produced it, fixed random seed used.
- **Per-stage timestamps**: wall-clock start/end for every pipeline
  stage (decomposition, routing, each pool call, orchestration,
  aggregation) *and* for the baseline call — both paths, same schema.
- **Token counts**: per call, per stage.
- **Cost**: computed from a fixed pricing/compute-cost table — the
  table itself must be a versioned file, not inline magic numbers
  scattered through code, so cost figures don't silently drift if
  pricing assumptions change mid-project.
- **Confidence scores**: router confidence per subtask, orchestrator
  replication decisions.
- **Model pins**: exact model name + checkpoint/revision for every
  component in that run (decomposer, each pool model used, aggregator,
  baseline) — pull this from config, don't hardcode.
- **Dispatch trace**: orchestrator's full dispatch order and replication
  decisions (TRD §2, Orchestrator row) — required, not optional debug
  output.
- **Aggregation I/O**: aggregator input and output logged for post-hoc
  quality review (TRD §2, Aggregator row).

## Format

TRD §6 recommends structured run logs (JSON/CSV) plus a lightweight
tracking tool. Suggested layout:

```
logs/
  runs/
    run_<id>.json        # one file per run: config + seed + all stage timestamps/tokens
  pricing_table.json      # versioned, fixed cost-per-token table (never inline)
results/
  aggregated_<phase>.csv  # derived from logs/runs/*.json only — never hand-edited
```

Anything in `results/` must be regenerable by re-running the aggregation
script over `logs/runs/`. If a number appears in `results/` that
doesn't trace back to a file in `logs/runs/`, that's a bug to fix
before using the number anywhere.

## Parallel vs. simulated timing

If the SLM pool cannot be hosted with true concurrency on available
hardware (TRD §6, Compute row), log real per-call timestamps under
sequential execution and clearly tag the run config with
`"parallelism": "simulated"` vs `"real"`. Any report or chart built from
these logs must carry that label forward — never let a simulated-serial
run get summarized as if it measured real parallel latency.
