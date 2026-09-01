---
name: eval-dataset-builder
description: Use when building, stratifying, labeling, or splitting the evaluation query dataset — including gold task graphs (DAGs), held-out splits, the blind quality rubric, or the v2 structured per-query JSON records (SLM-pipeline response, multi-LLM responses, comparison, judge output). Use before any decomposer prompt-iteration work to confirm the held-out split has already been separated.
---

# Eval Dataset Builder (v2 — supersedes v1 dataset sizing)

Source of truth: `.agents/knowledge/TRD_source.txt` §5/§7,
`.agents/knowledge/Research_Report_v1_source.txt` (v1 used N=180, 65
gold DAGs, held-out set locked under SHA-256), and
`.agents/knowledge/Proposed_Architecture_v2_source.txt`.

## v1 vs v2 sizing

v1 shipped N=180 stratified queries, 65 gold DAGs, held-out set locked
by hash. The user wants v2 "more robust and larger" — this means grow
N and gold-DAG count from v1's actual numbers, not from the smaller
150-250 range the original TRD suggested before v1 ran. Confirm the
new target N with the user rather than picking an arbitrary larger
number — but whatever is chosen, it must exceed v1's N=180 and 65 gold
DAGs to count as "larger," and the sample-size justification (per
`statistical-analysis`) needs to be redone for the new target CI width,
not copy-pasted from v1's justification.

## v1 stratification still applies, plus one new axis

Keep v1's complexity stratification (single-domain / 2-domain /
3+-domain). Consider whether the v2 architecture's "color" concept
(from `Proposed_Architecture_v2_source.txt`) needs its own
stratification axis — e.g. queries that are guaranteed single-color
(loop never fires) vs. queries designed to trigger the feedback loop at
least once. This second axis is what actually lets you evaluate whether
the feedback loop helps, not just whether it exists — without it, you
can't isolate loop-triggering queries in the RQ5-style GED analysis.

## Held-out discipline (unchanged, still enforced by AGENTS.md rule 5)

Lock the v2 held-out split BEFORE any v2 decomposer/task-analyser
prompt iteration, exactly as v1 did (v1's lock: SHA-256 hash
`c13e3c1eb4bcd7889a439cea3a64102234b1325d012ca1bfdc8a23fce030890a`).
Generate and record a new hash for the v2 held-out set — do not reuse
or silently extend v1's locked set, since v1's set was already exposed
during v1 prompt development and can't be treated as clean for v2.

## v2 required JSON record schema (new)

Per the user's requirement — dataset must contain, per query, JSON
holding: SLM-pipeline response, each LLM-baseline response, the
comparison, and the judge's response + judged parameter. One file per
query (or one JSONL file with one record per line — confirm which with
the user before generating thousands of individual files):

```json
{
  "query_id": "string",
  "query_text": "string",
  "complexity_tier": "single-domain | 2-domain | 3+-domain",
  "triggers_feedback_loop": true,
  "gold_dag": { "...": "... (only for the gold-DAG subset)" },

  "slm_pipeline_response": {
    "final_response": "string",
    "run_id": "string (points to logs/runs/<id>.json)",
    "decomposition_depth_used": "int",
    "feedback_loop_fired": true
  },

  "baseline_responses": {
    "<baseline_model_id_1>": { "response": "string", "run_id": "string" },
    "<baseline_model_id_2>": { "response": "string", "run_id": "string" },
    "...": "... one entry per roster model, see multi-llm-baseline-pool"
  },

  "comparison": {
    "rubric_scores": {
      "slm_pipeline": { "correctness": "...", "completeness": "...", "coherence": "..." },
      "<baseline_model_id>": { "...": "..." }
    }
  },

  "judge_result": {
    "judge_model_id": "string",
    "selected_system": "slm_pipeline | <baseline_model_id>",
    "criteria_scores": { "correctness": "...", "completeness": "...", "coherence": "..." },
    "reasoning": "string",
    "run_id": "string (points to judge log, see llm-judge-blind-eval)"
  }
}
```

Every `run_id` field must point to a real file under `logs/runs/` — a
record with a fabricated or missing `run_id` is not a valid dataset
entry and shouldn't be written until the real run exists.

## Blind rubric scoring (unchanged from v1, still required alongside the judge)

Keep v1's blind rubric scoring process (shuffle, strip labels,
un-blind only after scoring) as an independent quality signal alongside
the new LLM-judge — the two are meant to be cross-checked against each
other (see `llm-judge-blind-eval` analysis section), not one replacing
the other.

## Deliverable checklist (v2)

- [ ] New, larger stratified dataset (exceeds v1's N=180 / 65 gold DAGs)
- [ ] New stratification axis for feedback-loop-triggering vs. not
- [ ] New held-out split, separately hash-locked from v1's
- [ ] Per-query JSON/JSONL records matching the schema above, with
      every `run_id` traceable to a real log file
- [ ] Rubric spec (can reuse v1's, versioned) kept independent of judge
      prompt/schema
