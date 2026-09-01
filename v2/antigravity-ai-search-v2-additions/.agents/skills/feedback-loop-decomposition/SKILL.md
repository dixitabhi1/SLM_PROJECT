---
name: feedback-loop-decomposition
description: Use when implementing or modifying the Decomposer SLM, SLM-2 task analyser, SLM-3 task colorer, Matching SLM, or Scheduling SLM from the v2 proposed architecture — specifically the re-decomposition feedback loop that sends a task back to the Decomposer when it spans multiple colors and the depth limit hasn't been reached. Use before writing any code that decides whether a task loops back or proceeds to scheduling.
---

# Feedback Loop Decomposition (v2)

Source of truth: `.agents/knowledge/Proposed_Architecture_v2_source.txt`.
This is a literal transcription of the diagram — re-read it before
implementing, this skill is a checklist, not a replacement.

## The loop condition, exactly

At the Matching SLM stage, for each task, evaluate two boolean checks:

```
spans_multiple_colors = (task's required skills span > 1 color)
depth_limit_reached    = (current re-decomposition depth >= max_depth)

if spans_multiple_colors AND NOT depth_limit_reached:
    -> loop back to Decomposer SLM (re-decompose this task further)
else:
    -> proceed to Scheduling SLM
    (if still spans_multiple_colors here, it's because depth_limit_reached —
     Scheduling SLM must assign MULTI-AGENT collaboration for it,
     not force a single-agent match)
```

Do not implement this as "retry until confident" or any open-ended
loop — `max_depth` must be an explicit configured integer, logged in
the run config alongside model pins. An unbounded loop is a
implementation bug regardless of how it's phrased in a request.

## Before implementing, resolve the open questions — don't guess

`Proposed_Architecture_v2_source.txt` lists open questions the diagram
doesn't answer (color taxonomy, depth limit value, whether loop-back
reuses the task ID or spawns a child task, where the v1 Aggregator fits
after Scheduling). If any of these come up in the middle of
implementation and aren't yet decided, stop and ask rather than picking
a default silently — these are structural decisions that change the
run-log schema and downstream stats, not stylistic ones.

## Logging requirements (extends experiment-instrumentation)

Every re-decomposition event needs its own log entry:
- task ID (and whether it's the same ID or a new child ID — per the
  resolved open question above)
- recursion depth at time of loop-back
- which color(s) the task spanned that triggered the loop
- the task's skill vector (from SLM-2) at each iteration, so you can
  see whether re-decomposition is actually converging toward
  single-color subtasks or oscillating
- final resolution: single-color match, OR multi-agent collaboration
  at depth limit — log which one, for every task, no exceptions

This log is what lets you later answer "how often does the loop fire,
and does it help" — a question your v1 report's RQ5 (GED / decomposition
accuracy) directly anticipates needing (GED contributed ~35% of quality
delta in v1; the feedback loop is the direct fix being tested for that).

## Testing the loop before full integration

Before wiring this into the full pipeline, unit-test the loop condition
in isolation against synthetic tasks: one clearly single-color, one
clearly multi-color with room to recurse, one multi-color already at
depth limit. Confirm all three routes (direct match, loop-then-match,
loop-then-multi-agent-at-limit) produce the expected path before
running it on real eval queries — this is cheap to verify and expensive
to debug after it's tangled into real runs.
