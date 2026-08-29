---
name: eval-dataset-builder
description: Use when building, stratifying, labeling, or splitting the evaluation query dataset — including gold task graphs (DAGs), held-out splits, or the blind quality rubric. Use before any decomposer prompt-iteration work to confirm the held-out split has already been separated.
---

# Eval Dataset Builder

Source of truth: `.agents/knowledge/TRD_source.txt` §5 and §7, and
`.agents/knowledge/Implementation_Plan_source.txt` Phase 2–3.

## Dataset requirements

- **150–250 queries**, stratified by complexity:
  - single-domain (control)
  - 2-domain compound
  - 3+-domain compound
- **Gold task graphs**: hand-labeled DAGs for **at least 50 queries**,
  used to measure decomposition accuracy directly (RQ5).
- **Held-out split**: must be separated **before any decomposer prompt
  iteration begins**, to avoid overfitting the prompt to the eval set.
  If you are about to iterate on decomposer prompts and no held-out
  split has been created yet and locked, stop and create/lock it first.
- **Quality rubric**: pre-registered scoring criteria — correctness,
  completeness, coherence — applied **blind** to which system produced
  the answer. Blinding matters more here than in an SLM-vs-SLM
  comparison, because output style differences between SLMs and a large
  LLM could otherwise bias a human rater (TRD §5).

## Sample size

TRD §8 requires pre-computing the required sample size for the target
CI width **before finalizing dataset size** — don't treat 150–250 as
just picked from the range; run/request the power calculation and
record which end of the range (and why) it lands on. This connects to
the `statistical-analysis` skill.

## Held-out discipline (enforced by AGENTS.md rule 5)

Once locked, no code may read, print, filter on, or otherwise use the
held-out split's queries or answers until Phase 7/8 (pipeline runs on
eval set / statistical analysis). This includes:
- printing held-out examples for "sanity checking" a prompt
- computing any metric against held-out queries during Phase 6 dev
- accidentally including held-out query IDs in a prompt few-shot set

If a task risks any of the above, flag it and propose using a small
separate dev/validation slice instead.

## Blind scoring mechanics

When implementing the rubric scoring step:
- Shuffle and strip system labels before presenting outputs to a rater
  (human or LLM-judge) — TRD §7.
- Log which system produced which output *separately*, in a mapping
  file the scorer doesn't see, so results can be un-blinded only after
  scoring is complete.
- If more than one judge is used, log per-judge scores separately to
  support inter-rater agreement reporting (TRD §8 threats table).

## Deliverable checklist (Implementation Plan §2)

- [ ] Stratified eval dataset with gold task graphs, saved as a
      reusable artifact (not regenerated ad hoc per run)
- [ ] Held-out split file, separated and locked with a timestamp/commit
      marking when it was frozen
- [ ] Rubric spec (pre-registered, versioned) separate from any scoring
      code, so the rubric itself is auditable
