---
name: llm-judge-blind-eval
description: Use when implementing, running, or analyzing the LLM-as-judge step — an anonymous, randomized selection of the best response from the pool of SLM-pipeline and multi-LLM-baseline responses. Use before writing judge prompts, before any code that shows the judge candidate responses, and before reporting judge win-rates.
---

# LLM-as-Judge Blind Evaluation (v2)

New in v2 — v1 used a fixed rubric scored by human/single-process
blind scoring (see `eval-dataset-builder`'s rubric section). v2 adds an
LLM-judge that picks the single best response out of the full candidate
pool (SLM pipeline + all 4-5 LLM baselines) per query. The two methods
are complementary, not replacements for each other — keep both,
report both, don't let the judge's picks silently become the only
quality signal.

## Why "anonymous" matters here, specifically

The stated intent is a *practical* signal of response quality — which
only holds if the judge cannot infer which system produced which
response from anything other than the response content itself. Two
distinct bias risks to guard against:

1. **Position bias**: judges (human and LLM) are known to favor
   whichever response appears first/last in a list. Randomize candidate
   order per query, and ideally re-run each judgment with the order
   reversed/reshuffled as a check.
2. **Self-preference bias**: an LLM judge tends to favor outputs
   written in a style similar to its own, and especially favors outputs
   it recognizes as its own. **The judge model must not be one of the
   models in the candidate pool** (not the SLM pool, not any of the 4-5
   baselines) — pick a separate model for judging, and record that
   choice explicitly. If budget forces reusing a pool/baseline model as
   judge, flag this as a known limitation in the report rather than
   letting it pass silently.

## Mechanics

1. **Strip all identifying signal** before the judge sees anything:
   no model names, no system labels, no formatting fingerprints
   introduced by your own pipeline code (e.g. don't let the SLM
   pipeline's aggregator add a signature phrase that survives into the
   judge prompt).
2. **Shuffle candidate order** per query, independently each time —
   don't reuse the same shuffle across all queries.
3. **Log the true identity mapping separately**, in a file the judge
   process never reads, so you can un-blind after judging completes.
4. **Require the judge to output structured reasoning**, not just a
   winner: which candidate it picked, and which criteria drove the
   pick (correctness / completeness / coherence — same three axes as
   v1's rubric, so judge picks and rubric scores are comparable). This
   is the "on what parameter it's picking the response" requirement —
   it must be a required structured field, not inferred after the fact
   from the judge's free-text explanation.
5. **Multiple queries per judge run, one decision per query** — don't
   batch multiple queries' candidates into one judge call; keep
   judgments independent to avoid cross-query anchoring.

## Required JSON output schema per judged query

```json
{
  "query_id": "string",
  "judge_model_id": "string (pinned, not in candidate pool)",
  "candidate_order_shuffle_seed": "int",
  "candidates_shown": ["anon_id_1", "anon_id_2", "..."],
  "selected_anon_id": "string",
  "criteria_scores": {
    "correctness": "...",
    "completeness": "...",
    "coherence": "..."
  },
  "reasoning": "string (judge's stated rationale)",
  "true_identity_mapping": {
    "anon_id_1": "slm_pipeline | baseline_model_id",
    "...": "..."
  }
}
```

`true_identity_mapping` lives in a separate log file from the rest of
this record in practice (per point 3 above) — shown together here only
to specify the full field set.

## Analysis

- Report **judge win-rate per system** (SLM pipeline vs. each of the
  4-5 baselines), not just "the judge preferred X% of the time"
  aggregated — this parallels the per-baseline-model breakdown in
  `multi-llm-baseline-pool`.
- Cross-check judge picks against the v1-style rubric scores for
  agreement — report agreement rate, don't just report whichever
  method gives a more favorable result.
- If the judge shows measurable position or self-preference bias in
  your position-swapped trials, report that as a limitation, not
  something to quietly correct for and hide.
