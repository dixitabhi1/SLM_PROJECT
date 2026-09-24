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
9. **Zero synthetic score imputation.** Never fill missing evaluation
   trials with imputed, averaged, or mirrored scores.
10. **Symmetric judge evaluation.** Every pairwise comparison must
    evaluate both forward and swapped positions to detect positional bias.
11. **Cryptographic separation of evaluation keys.** Candidate identity
    unblinding keys must be kept in separate directories from public judge logs.
12. **Model host and catalog verification.** Never assume a model exists
    on an inference endpoint without verifying its published catalog status.
13. **Distinct-model pre-flight verification requirement.** Before any
    generation or evaluation run begins, the runner MUST execute an
    automated pre-flight assertion verifying that every system in the
    roster (all pool specialists, the aggregator, and every comparative
    baseline) maps to a genuinely distinct `api_model_name`/endpoint, and
    that no pool/pipeline component matches any baseline's model. The runner
    must fail loudly and abort immediately if this check fails. Single-model
    proxies or shared model fallbacks are strictly prohibited.
14. **Editor tab and background write lock discipline.** Do not keep
    `PROGRESS.md`, result JSONLs, or any auto-updated run files open in
    an active editor tab while a background generation or judge task is
    actively running. This prevents IDE in-memory buffer desync, auto-save
    file collisions, and `.git/index` write contention. Git commands
    must never be executed concurrently while background file-writing
    tasks are active.
15. **Compound-query decomposition non-collapse assertion.** The pipeline
    runner MUST execute an automated pre-flight assertion verifying that
    any query tagged as compound (`three_plus_domain`, `compound_dag`, or
    `compound`) produces >= 2 distinct subtask nodes. If the decomposer
    collapses a compound query to a single node, the runner must fail loudly
    and refuse to proceed to generation. Single-node collapse on compound
    queries is strictly prohibited.
16. **Mentor Experiment Protocol (E1–E4) Operational Governance.**
    When executing experiments under the Mentor Experiment Protocol (`mentor_experiment_protocol_source.txt`), the following rules strictly apply:
    (a) **SLM Pool Sizing:** The SLM pool component constraint is 5–8B parameters per component (supersedes the earlier ≤5B rule for this new protocol only; both constraints coexist as separate tracked lineages, and the validity of prior ≤5B results is preserved).
    (b) **Fairness Constraint Pre-Flight Check:** The baseline LLM parameter count must exceed the SLM pool's *combined* parameter sum participating in the comparison (sum(P_SLM) < P_Baseline). This must be asserted and verified before execution against all comparative baseline tiers.
    (c) **Dual-Framework Independent Judging:** Both the established 1–5 criteria-based framework (Correctness, Completeness, Coherence) and the new 1–10 holistic framework must be evaluated and logged independently for every judge trial. Never convert or map scores from one framework into the other.
    (d) **First-Class Draw Accounting:** Draws (QS = QL) are a distinct, first-class outcome category and must be logged and reported separately. Never fold draws into win or loss counts.

## Skills available in this project

| Skill | Loads when |
|---|---|
| `anti-hallucination-guardrails` | Any time the agent is about to state a metric, spec, or model detail |
| `slm-pipeline-architecture` | Building/modifying decomposer, router, pool, orchestrator, aggregator |
| `baseline-model-runner` | Setting up or running the LLM baseline |
| `multi-llm-baseline-pool` | Setting up multi-tier baseline ladders and pre-flight fairness assertions |
| `llm-judge-blind-eval` | Symmetrical double-blind judging under 1-5 and 1-10 protocols with draw accounting |
| `eval-dataset-builder` | Building the stratified query set / gold DAGs / multi-domain benchmark sets |
| `experiment-instrumentation` | Any code that runs a pipeline or baseline call and needs to log |
| `statistical-analysis` | Computing CIs, dual Quality Proximity metrics, non-inferiority tests |

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

## Autonomous Audit Loop Protocol (post-Step B)

From this point forward, operate in a self-auditing loop: after any generation, judging, or scoring step, automatically apply the following checks before reporting or advancing — do not wait to be asked.

- **Raw-file score/label concordance check** on every judge trial (`selected_candidate` matches the higher score sum; flag and correct any mismatch).
- **Baseline truncation diagnostic** — any SLM win where the comparator's judge reasoning mentions "cut off," "truncated," or similar is void by default; state this explicitly, never average it in.
- **No-blending rule** — never report a combined/pooled score across queries with different truncation/validity status. Report each query's audited number separately.
- **Temperature/sampling confound check** — if any non-zero temperature fired during a run, confirm the result isn't a sampling artifact before crediting it as a real gain.
- **Positional swap consistency** — report this rate every time a judging pass runs, not only when it happens to be high.
- **Reconciliation rule** — any derived or aggregate statistic must trace to a stated inclusion/exclusion rule, applied consistently across all trials, never decided case-by-case after seeing the result.
- **Three-Dimensional Evaluation Paradigm** — in addition to binary win-rate, every comparative evaluation against cross-tier baselines MUST report continuous **Mean Quality Proximity ($P_{\text{mean}} = \frac{1}{N}\sum (1 - \frac{|Q_S - Q_L|}{4}) \times 100\%$)** and **Mean Signed Quality Delta ($\overline{\Delta Q} = \frac{1}{N}\sum (Q_S - Q_L)$)** with 95% Confidence Intervals alongside cost/compute ratios. Win rate alone must never be presented as the sole quality metric against frontier baselines.

Continue looping through subsequent steps automatically once each step's audit passes clean. Hard stops (pause and wait for my explicit confirmation) remain limited to: locking a new held-out dataset, pinning a new baseline/judge model, spending beyond an already-approved compute/API budget, any request to relax an existing constraint (e.g. ≤5B → 8B), and anything going into a mentor-facing report. Everything else — generation, judging, the seven checks above, and reporting the outcome — proceeds without waiting for a prompt at each step.

If a self-audit check ever fails (label mismatch, truncation confound, blending violation, etc.), log it in PROGRESS.md as an incident the same way the mid-session hand-edit was logged earlier in this project, fix it, and continue the loop — never silently drop or bury the finding.

## Phase F: Accuracy-Focused Specialist Fine-Tuning Initiative

This initiative is clearly distinct from the closed v1–Step B study (which evaluated frozen baseline models under prompting and external tools). Phase F explores parameter adaptation: whether targeted LoRA/QLoRA domain fine-tuning of individual $\le 5\text{B}$ specialists on verified ground-truth corpora closes the quality gap toward the 65–70% target against frontier baselines.

### Phase F Hard Stops (Pause and wait for explicit confirmation)
1. **F-HS 1 (Compute Feasibility & Budget Confirmation):** Confirm real compute feasibility, VRAM headroom, and dollar cost before spending anything or launching training runs.
2. **F-HS 2 (Individual Specialist Fine-Tuning Gate):** Obtain explicit confirmation before fine-tuning each individual specialist model.
3. **F-HS 3 (Report Accuracy Claim Gate):** Symmetrical double-blind evaluation and Autonomous Audit Loop pass before any accuracy claim enters a mentor report.

