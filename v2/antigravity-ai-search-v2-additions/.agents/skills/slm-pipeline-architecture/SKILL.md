---
name: slm-pipeline-architecture
description: Use when building, modifying, or explaining the proposed all-SLM pipeline — Decomposer SLM, SLM-2 task analyser, SLM-3 task colorer, SLM agent pool, Agent analyser/colorer SLM, Matching SLM, Scheduling SLM, and the Aggregator. Also use when the user asks to add a component or change matching/scheduling logic. This is the v2 architecture; v1's simpler decomposer -> router -> pool -> orchestrator -> aggregator is documented for history in Research_Report_v1_source.txt.
---

# SLM Pipeline Architecture (v2)

Source of truth: `.agents/knowledge/Proposed_Architecture_v2_source.txt`
(the current target) and `.agents/knowledge/Research_Report_v1_source.txt`
§1 (v1's shipped architecture, for reference/comparison only — not the
build target anymore for the decomposition/routing stages).

## v2 pipeline (current build target)

```
                    Decomposer SLM
                          |
                          v
                 SLM-2: task analyser
                          |
                          v
                 SLM-3: task colorer  ---------\
                                                 v
SLM agent pool -> Agent analyser SLM      Matching SLM
                     |                    (within-color match,
                     v                     parallel/series mark,
              Agent colorer SLM  -------->  multi-color check)
                                                 |
                            multi-color AND      | depth limit
                            depth limit NOT      | reached
                            reached: loop back    v
                            to Decomposer SLM  Scheduling SLM
                                    ^          (execution graph,
                                    |           parallel/series by
                                    +-----------  cost/availability,
                                                  multi-agent for
                                                  multi-color-at-limit)
                                                 |
                                                 v
                                    (Aggregator — placement TBD,
                                     see open question 5 in
                                     Proposed_Architecture_v2_source.txt)
```

For the exact loop condition, use the `feedback-loop-decomposition`
skill — don't re-derive it here, it's easy to paraphrase the "multi-
color and depth limit not reached" condition slightly wrong.

## What's reused from v1 vs. new in v2

**Reused, unchanged**: the SLM agent pool itself (coding/math/logic/
retrieval/general specialists from v1's pinning table), the concept of
an Aggregator doing synthesis + contradiction resolution, the ≤8B /
≤3B parameter caps.

**New in v2**: SLM-2 task analyser, SLM-3 task colorer, Agent analyser
SLM, Agent colorer SLM, Matching SLM, Scheduling SLM, and the
feedback loop. v1's single non-generative "Capability Router" is
superseded by this multi-stage analyse/color/match/schedule sequence —
don't keep v1's router as a separate component unless the user
confirms they want both.

## Before implementing any v2 component

1. Check `Proposed_Architecture_v2_source.txt`'s open-questions section
   — color taxonomy, depth limit value, task-ID versioning on loop-back,
   cost model in Scheduling SLM, and Aggregator placement are all
   unresolved by the diagram alone. Don't implement around an assumed
   answer to any of these; ask.
2. Decide (with the user) whether SLM-2/SLM-3/Matching/Scheduling are
   actual model calls or rule-based/embedding components like v1's
   router — the diagram's "SLM" naming suggests model calls, but this
   has real latency/cost implications worth confirming explicitly
   given v1's RQ1/RQ2 focus on overhead.
3. Wire instrumentation hooks (per `experiment-instrumentation` and
   `feedback-loop-decomposition`'s logging section) at every new stage
   boundary from the start, same discipline as v1.
