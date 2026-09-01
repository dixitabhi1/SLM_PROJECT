---
name: statistical-analysis
description: Use when computing latency ratios, cost ratios, quality non-inferiority tests, the crossover-point plot, decomposition accuracy (graph edit distance vs. gold DAGs), sample-size/power calculations, or confidence intervals for the AI Search Framework study. Use before writing any results section of the report.
---

# Statistical Analysis

Source of truth: `.agents/knowledge/TRD_source.txt` §7–8 and
`.agents/knowledge/PRD_source.txt` §5.

## Design

**Within-subject, paired comparison** — every query is answered by
*both* the LLM baseline and the all-SLM pipeline, so differences are
paired, not between independent samples. All statistical tests must use
a paired method (e.g. paired t-test/Wilcoxon signed-rank for latency
and cost, paired non-inferiority test for quality) — an unpaired test
here would misrepresent the design and understate power.

## The five required analyses (map directly to RQ1–RQ5)

1. **Latency ratio (RQ1)**: wall-clock time per query, both systems,
   same hardware/time window where possible. Report mean, median, and
   95% CI **per complexity bucket** (single/2-domain/3+-domain) — not
   just an aggregate.
2. **Cost ratio (RQ2)**: token counts × the fixed pricing/compute-cost
   table (from `experiment-instrumentation`), same queries, same
   stratification as latency.
3. **Quality (RQ3)**: blind rubric scoring, outputs shuffled/unlabeled
   before scoring (see `eval-dataset-builder`). **Non-inferiority is
   the primary claim** — "the SLM pipeline doesn't meaningfully lose to
   the LLM," not "beats it." Pre-register the non-inferiority margin
   before running the test, don't pick it post-hoc to make the result
   look favorable.
4. **Crossover point (RQ4)**: plot latency/cost ratio against
   complexity bucket; identify where the SLM pipeline's advantage
   shrinks or disappears. This should be a real plot from logged data,
   not an illustrative sketch.
5. **Decomposition accuracy (RQ5)**: compare produced DAGs against the
   ≥50-query gold-labeled subset via structural similarity (e.g. graph
   edit distance); correlate with quality gaps vs. the LLM baseline.

## Sample size

Pre-compute the required sample size for the target CI width **before**
finalizing the 150–250 query dataset size (TRD §8, first threat-to-
validity row) — this is an input to `eval-dataset-builder`, not
something to back-justify after the dataset already exists.

## Reporting discipline

- Report **per-complexity-bucket** results as the primary table; only
  show an aggregate as a secondary summary, since TRD §8 and the
  Implementation Plan's Key Risks both anticipate the LLM baseline
  winning decisively on hard subtasks and require that this not be
  "smoothed into an aggregate."
- If more than one quality judge is used, report inter-rater agreement
  (TRD §8).
- Every number in a results table needs a pointer back to the log
  file(s)/run IDs it was computed from (AGENTS.md rule 2/8) — treat
  this the same way you'd treat a citation in a paper, except the
  citation target is a file in this repo, not an external source.
- **Reconciliation Rule (MANDATORY)**: Any derived statistic (pairwise matrices,
  Bradley-Terry ratings, Elo scores, or multi-criteria aggregations) must
  include an explicit reconciliation check against the primary win-count
  table (`results/v2_aggregated_results.json`) before being included in any
  report. The full derivation code and per-query trace must be verifiable
  from logged run records on request. Synthetic or heuristically assigned
  scoring matrices are strictly prohibited.
- A result where latency/cost win clearly but quality trails on hard
  subtasks is, per the PRD, "a realistic and still-valid outcome" —
  write it up as such rather than reframing it as an unqualified win.
