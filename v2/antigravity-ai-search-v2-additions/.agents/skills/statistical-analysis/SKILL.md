---
name: statistical-analysis
description: Use when computing latency/cost ratios, quality non-inferiority, crossover plots, decomposition accuracy, judge win-rates, or feedback-loop effectiveness for the v2 multi-baseline study. Use before writing any results section of the v2 report.
---

# Statistical Analysis (v2 — extends v1's single-baseline analysis)

Source of truth: `.agents/knowledge/TRD_source.txt` §7-8,
`.agents/knowledge/Research_Report_v1_source.txt` §4 (v1's actual
results table, for continuity/comparison), and the
`multi-llm-baseline-pool` / `llm-judge-blind-eval` skills.

## What's structurally new vs. v1

v1's Table 4 was: complexity tier (4 rows) x {latency ratio, cost
ratio, quality delta, GED}, one baseline. v2 needs:

1. **Per-baseline-model breakdown**: every metric from v1's table now
   repeated per roster model (4-5x more rows) — SLM pipeline vs. each
   baseline individually, not collapsed into one "vs. LLM" number.
2. **A scale trend view**: since the v2 roster spans parameter sizes,
   add a plot/table of metric-vs-baseline-parameter-count — this is
   the actual point of using varying sizes (does the SLM pipeline's
   advantage shrink monotonically as baseline size grows, or is there
   a specific tier where it crosses over?).
3. **Judge win-rate stats**: per-system win-rate from
   `llm-judge-blind-eval`, plus agreement rate between judge picks and
   rubric scores.
4. **Feedback-loop effectiveness**: compare GED and quality delta
   between queries that triggered the loop vs. those that didn't
   (requires the `triggers_feedback_loop` stratification from
   `eval-dataset-builder`) — this is the direct test of whether the v2
   architecture actually fixes the ~35% GED-attributed quality loss v1
   identified.

## Design remains within-subject and paired

Still a paired design — same queries answered by SLM pipeline and
every baseline. With 5 baselines this is a paired comparison replicated
5 times per query, not 5 independent studies — use a method that
accounts for the shared query set (e.g. repeated-measures or a paired
test per baseline pair with appropriate multiple-comparison correction
across the 5 baseline comparisons, rather than treating each baseline
comparison's p-value as if it were the only test run).

## v1-to-v2 comparability

Where a v2 baseline roster member is the SAME model as v1's baseline
(if you reuse Llama-3.1-70B-Instruct), that specific pairwise comparison
can be reported alongside v1's original numbers as a direct
"replication check" — report both v1's and v2's numbers for that pair
side by side, and flag if they diverge meaningfully (different N,
different held-out set, or a real architecture change from the
feedback loop could all cause divergence — say which you think it is,
don't just report the discrepancy unexplained).

## Reporting discipline (unchanged principle, more rows)

- Primary table: per-baseline-model x per-complexity-tier x
  feedback-loop-triggered-or-not. This is a large table — the
  short-report version (see `report-generation`) needs an aggregated
  view, but the detailed report needs the full breakdown, same
  principle as v1's "don't smooth into an aggregate."
- Every number still needs a `run_id`/log-file pointer, same as v1 and
  per `experiment-instrumentation`.
- If non-inferiority is rejected for some baselines but not others
  (plausible once smaller baselines are in the roster), report this
  per-baseline — a mixed result across the roster is itself a finding,
  not a case to average away.
