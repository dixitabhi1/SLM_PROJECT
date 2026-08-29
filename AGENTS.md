# AGENTS.md — AI Search Framework (all-SLM pipeline vs. LLM baseline)

This file is read by Antigravity at the start of every session in this
project. It sets ground rules that apply regardless of which skill is
active.

## What this project is

A research project (NOT a production search product) testing whether an
entirely SLM-based (≤8B params, zero LLMs anywhere in the shipped
architecture) decomposed pipeline can match a single large LLM
(frontier-class or 70B+, held fixed for the whole study) on latency,
cost, and quality. Full specs live in `.agents/knowledge/`:

- `PRD_source.txt` — problem statement, research questions RQ1–RQ5,
  goals, non-goals, success metrics, scope.
- `TRD_source.txt` — system architecture, component requirements,
  candidate SLM pool, baseline model requirements, data requirements,
  infra requirements, evaluation methodology, threats to validity.
- `Implementation_Plan_source.txt` — 10-week phased plan, deliverables,
  key risks.

**These three files are the single source of truth.** They are the
user's own uploaded documents, extracted verbatim from the PDFs — not a
summary and not the agent's memory of them.

## Hard rules (apply in every skill, every session)

1. **Ground every architectural or methodological claim in the source
   docs, not in general knowledge.** Before describing "the pipeline,"
   "the baseline," or "the eval design," grep/read the relevant section
   in `.agents/knowledge/`. If a detail isn't in those files, say so
   explicitly instead of inventing a plausible-sounding default.
2. **Never fabricate a number.** No latency, cost, quality, accuracy,
   CI, or p-value appears anywhere (code comments, docs, chat, reports)
   unless it was actually computed from a logged run in this repo. If a
   number doesn't exist yet, write `TBD — pending run <phase>` rather
   than a placeholder that looks real (no invented "e.g. 4.2x latency
   reduction" type filler — the PRD's ~3–6x / ~5–10x figures are
   *planning targets from the doc*, not measured results, and must
   never be reported as measured results).
3. **The architecture is fixed, not a design choice to improve on.**
   Decomposer ≤3B, every pool model ≤8B, router is rule-based/embedding
   (not generative), aggregator ≤8B. Zero LLMs anywhere in the proposed
   system. The baseline is a single large LLM held fixed for the whole
   experiment — never suggest swapping it mid-study, never suggest
   adding an LLM into the proposed pipeline "to improve quality."
4. **Every experimental run must be reproducible from a logged
   config + seed.** No script that produces a latency/cost/quality
   number is acceptable unless it also writes a structured run log
   (see `experiment-instrumentation` skill).
5. **Held-out discipline.** Once the held-out eval split is created
   (Phase 3), no code the agent writes may read, print, or tune against
   it until Phase 7/8. Flag any accidental leakage immediately.
6. **State uncertainty and negative results plainly.** RQ3/RQ4 and the
   "Key Risks" section explicitly anticipate the SLM pipeline losing on
   hard subtasks — this is a valid, reportable outcome, not a bug to
   hide, reframe, or paper over with rounding/cherry-picked buckets.
7. **Pin everything.** Every model used (decomposer, each pool member,
   baseline) needs an exact name + checkpoint/revision recorded in the
   run config. "A Llama model" is not a pin; `meta-llama/Llama-3.1-70B-
   Instruct @ <revision>` is.
8. **When asked to write the report/write-up**, cite the specific
   experiment log or analysis file that backs each reported number
   (file path + run ID), the same way a paper would cite a results
   table — this is for the user's own traceability, not external
   publication citation.

## Skills available in this project

| Skill | Loads when |
|---|---|
| `anti-hallucination-guardrails` | Any time the agent is about to state a metric, spec, or model detail |
| `slm-pipeline-architecture` | Building/modifying decomposer, router, pool, orchestrator, aggregator |
| `baseline-model-runner` | Setting up or running the LLM baseline |
| `eval-dataset-builder` | Building the stratified query set / gold DAGs |
| `experiment-instrumentation` | Any code that runs a pipeline or baseline call and needs to log |
| `statistical-analysis` | Computing CIs, non-inferiority tests, crossover plots, decomposition accuracy |

## Antigravity settings recommended for this project

Set these once, in Antigravity Settings (Cmd/Ctrl+,):

- **Autonomy**: start on a Review/checkpoint preset (not full-auto) for
  Phases 3–8; this is a research artifact where a wrong assumption
  compounds across weeks of logged runs.
- **Terminal**: Request Review with an allow-list of read-only /
  known-safe commands (`pytest`, `python -m ...`, `git status`, `git
  diff`) rather than blanket auto-run, since local model hosting and
  GPU scripts can be expensive to re-run if wrong.
- Keep `.agents/knowledge/` and this `AGENTS.md` in every workspace
  clone — they are what keeps the agent grounded in *this* spec instead
  of a generic "build a RAG search app" default.

## Looping / continuation protocol

- At the start of every session, read PROGRESS.md before anything else
  to determine current phase and status.
- At the end of every phase (or every session, whichever comes first),
  update PROGRESS.md: mark the phase status, and write a "Last session
  summary" so a future session — or the next loop iteration — can
  resume without re-deriving context.
- The four "Hard stops" in PROGRESS.md apply regardless of autonomy
  settings. Even in a full-auto/looping mode, pause and surface these
  explicitly rather than proceeding silently — they are irreversible or
  expensive if wrong (locking the held-out split, pinning the baseline
  model, running paid/GPU-time inference, or asserting a claim in the
  final report).
- Never let looping cause a phase to be silently skipped or reordered.
  If Phase N's deliverable (per the Implementation Plan) isn't complete,
  do not start Phase N+1's work.
