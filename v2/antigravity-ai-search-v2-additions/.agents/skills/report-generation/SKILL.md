---
name: report-generation
description: Use when drafting the v2 research report — both the full detailed report and the shorter presentation version. Use only after v2 statistical analysis is complete; do not use to draft a report around results that don't exist yet.
---

# Report Generation (v2 — detailed + condensed)

Source of truth for style/structure: `.agents/knowledge/Research_Report_v1_source.txt`
(the v1 report is the template — same section structure, same
insistence on measured numbers with CI and run-log traceability).

## Two outputs, one source of numbers

Generate both from the SAME results files (`results/v2/aggregated_results.json`,
`results/v2/ablations/`, judge win-rate outputs) — never let the
detailed and condensed versions be drafted independently, or numbers
will drift between them. Build the condensed version by summarizing
the detailed version's tables, not by recomputing anything.

## Detailed report — structure (mirrors v1)

1. Executive Summary — key findings up front, same style as v1's
2. System Architecture & Model Pinning — v2 diagram + full pinning
   table (decomposer, SLM-2/3, agent analyser/colorer, matching,
   scheduling, aggregator, all 4-5 baselines, judge model)
3. Literature Grounding & Scoping — reuse/extend v1's table if nothing
   new needs adding
4. Hypotheses & Evaluation Protocol — RQ1-RQ5 as in v1, PLUS new RQs
   for judge win-rate and feedback-loop effectiveness if the user wants
   them formalized as RQ6/RQ7
5. Empirical Results — full per-baseline x per-tier x
   loop-triggered-or-not tables (see `statistical-analysis`)
6. Ablation Studies — v1's ablations plus any new v2 ablations (e.g.
   depth-limit sensitivity, judge-model choice sensitivity)
7. Threats to Validity & Discussion — carry v1's three points forward,
   add judge bias risks from `llm-judge-blind-eval` and the open
   architectural questions from `Proposed_Architecture_v2_source.txt`
   that got resolved during implementation (document HOW they were
   resolved, since they weren't specified in the original diagram)
8. Conclusion & Research Significance
9. v1-to-v2 comparison section — explicit, since v1 is a real
   completed study, not superseded history to omit

## Condensed / presentation version — structure

Aim for something that fits 8-12 slides worth of content:
1. One-line problem statement + one-line headline result
2. Architecture diagram (v2, simplified — the feedback loop is the one
   new thing worth visually emphasizing vs. v1)
3. One results table: aggregate per-baseline win/loss on
   latency/cost/quality (collapse complexity tiers here — this is the
   one place aggregation is fine, since the detailed report carries the
   full breakdown)
4. Judge win-rate chart (per system)
5. Feedback-loop effectiveness — one before/after GED comparison
6. Conclusion — same core claim as the detailed report, shorter

## Non-negotiable, both versions

- Every number needs the same traceability as everywhere else in this
  project — a results-file/run-id pointer, even if it's a footnote in
  the condensed version rather than inline.
- Don't let the condensed version's brevity turn into overclaiming —
  "SLM pipeline wins on cost, loses on deep synthesis" stays true and
  stated plainly in both versions, not softened into an unqualified win
  for the short version.
- If v2 results are incomplete when asked to draft either report, say
  so and draft only the sections backed by real data — don't fill
  missing sections with placeholder numbers to make the draft look
  complete.
