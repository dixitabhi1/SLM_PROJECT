
---

# v2 ADDENDUM (append this section to the end of your existing AGENTS.md)

## v2 scope (builds on v1, does not replace it)

v1 is complete and its report is locked as historical ground truth in
`.agents/knowledge/Research_Report_v1_source.txt`. v2 adds, per the
user's identified gaps:

1. A feedback loop in decomposition (`feedback-loop-decomposition` skill,
   architecture in `Proposed_Architecture_v2_source.txt`)
2. A multi-LLM baseline roster of 4-5 models (`multi-llm-baseline-pool`
   skill) replacing v1's single 70B baseline
3. An anonymous LLM-judge selecting the best response from the full
   candidate pool (`llm-judge-blind-eval` skill)
4. A larger, more structured dataset with per-query JSON records
   (updated `eval-dataset-builder` skill)
5. Two report outputs — detailed and condensed (`report-generation` skill)

## v2-specific hard rules (in addition to the original 8)

9. **v1 results are immutable.** Never edit, "correct," or silently
   improve v1's numbers in `Research_Report_v1_source.txt`. If v2
   methodology changes make a direct comparison to a v1 number
   misleading, say so in the report rather than adjusting the v1 number.
10. **The v2 architecture diagram is the literal spec, not a starting
    point to improve on.** Implement the loop condition exactly as
    described in `Proposed_Architecture_v2_source.txt` (multi-color AND
    depth-limit-not-reached). Where the diagram is genuinely silent
    (color taxonomy, depth limit value, task-ID versioning, cost model,
    Aggregator placement), ask — don't fill the gap with a "reasonable"
    default and move on.
11. **Judge independence is non-negotiable.** The LLM-judge model must
    never be one of the models in the pool it's judging (SLM pool
    components or any of the 4-5 baselines). If this constraint can't
    be met, it must be flagged in the report as a limitation, not
    silently violated.
12. **v1 and v2 held-out sets are separate and both locked.** Don't reuse
    v1's held-out queries as v2's held-out set (they were exposed during
    v1 prompt development). Generate and record a new hash for v2's set.

## Updated skills table (v2)

| Skill | Loads when |
|---|---|
| `anti-hallucination-guardrails` | Unchanged — any time stating a metric, spec, or model detail |
| `slm-pipeline-architecture` | Now v2: Decomposer, SLM-2/3, Matching, Scheduling, Aggregator |
| `feedback-loop-decomposition` | NEW — implementing the loop-back condition |
| `baseline-model-runner` | v1 pinning discipline, still applies per-model |
| `multi-llm-baseline-pool` | NEW — the 4-5 model baseline roster |
| `llm-judge-blind-eval` | NEW — anonymous judge selection |
| `eval-dataset-builder` | Updated for v2: larger N, JSON schema, new held-out lock |
| `experiment-instrumentation` | Unchanged core requirements, extended fields per new skills |
| `statistical-analysis` | Updated for v2: per-baseline breakdown, judge stats, loop effectiveness |
| `report-generation` | NEW — detailed + condensed report generation |
