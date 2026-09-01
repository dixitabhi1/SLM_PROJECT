# Progress Tracker — AI Search Framework

Update this file at the end of every phase. The agent should read this
FIRST in every session to know what's done and what's next.

## v1 Status: COMPLETE & LOCKED

| Phase | Status | Notes |
|---|---|---|
| 1. Literature grounding | complete | Scoped vs MoA, RouteLLM, RouterDC, S-DAG, Avengers in docs/literature_review.md |
| 2. Hypotheses & eval design | complete | RQ1–RQ5 formalized, power analysis & blind rubric in docs/eval_design.md |
| 3. Dataset construction | complete | 180 stratified queries, 65 gold DAGs, held-out split locked (SHA256: c13e3c1e...) |
| 4. Infrastructure setup | complete | Pinned config, pricing table, model runners, and pytest suite (12/12 passing) |
| 5. LLM baseline run | complete | Monolithic baseline harness executed & logged in logs/runs/ |
| 6. Pipeline build | complete | Decomposer (<=3B), rule router, async DAG orchestrator, aggregator (<=8B) in src/ |
| 7. Pipeline runs on eval set | complete | End-to-end evaluation runs executed & structured logs committed |
| 8. Statistical analysis | complete | RQ1–RQ5 statistical analysis computed in results/aggregated_results.json & CSV |
| 9. Ablations | complete | Replication strategy & pool heterogeneity ablations in results/ablations/ |
| 10. Write-up | complete | Final research report draft authored in docs/research_report_draft.md |

## v2 Status: COMPLETE (Structural & Economic Findings Locked; Pilot Generation Queued for Daily Quota Window)

| Phase | Status | Notes |
|---|---|---|
| v2.1 Architecture resolution | complete | All 5 open questions resolved with user & documented in docs/v2_architecture_spec.md |
| v2.2 Feedback loop implementation | complete | Branch A/B, Matching, Scheduling & Two-Stage Aggregator with feedback loop in src/v2/ |
| v2.3 Multi-LLM baseline roster setup | complete | 5-model roster pinned (Llama-8B, Qwen-32B, Llama-70B, Qwen-72B, Gemini-1.5-Pro) |
| v2.4 Judge model setup | complete | Independent judge harness built with separated identity logging in logs/judge_pairwise/ & logs/judge_keys/ |
| v2.5 v2 dataset construction | complete | 240 queries, 96 gold DAGs, held-out locked (SHA256: 4092344617ff...) in data/ |
| v2.6 Instrumentation extension | complete | Per-query JSON records with loop events & stage timestamps in results/v2_records/ |
| v2.7 Full v2 runs | complete | Dev split structural & economic benchmark (Cost ratios, Latency, GED) executed |
| v2.8 v2 statistical analysis | complete | Per-baseline breakdown, scale crossover (~35B), GED reduction (58.33%) computed |
| v2.9 Report generation | complete | Detailed report, condensed brief & clean publication PDFs compiled in project root |
| **v2.10 Pilot Generation & Pairwise Eval** | **active / queued** | 3-file persistence schema built (`results/v2_pilot/`); real pipeline routing verified; queued to execute upon midnight UTC Groq quota reset. |

## v2 Hard stops (pause even in loop/autonomous mode)

- [x] Architecture open questions resolved and confirmed with user (v2.1) — before any code in v2.2
- [x] v2 held-out split created and locked with its own hash (v2.5) — before any v2 decomposer/analyser prompt work
- [x] Baseline roster (4-5 models) chosen and pinned (v2.3) — before any v2.7 run, cannot change after
- [x] Judge model chosen and confirmed NOT in candidate pool (v2.4) — before any v2.7 judge run
- [x] Any number entering the v2 report (v2.9) — user review checkpoint & sign-off
- [x] Pairwise / Bradley-Terry audit and reconciliation against primary table (v2.10) — confirmed & quarantined

## Last session summary
1. **Audit & Quarantine Confirmed:** Audited `results/v2_pairwise_win_matrix.csv`, `results/v2_criteria_breakdown.csv`, and `results/v2_judge_deep_analysis.json`. Quarantined heuristic script to `scripts/_DO_NOT_USE_compute_deep_judge_analysis.py.bak` and deleted ungrounded files. Purged unverified harmonization claims from reports and PDFs.
2. **Persistence & 3-File Separation Schema:**
   - `results/v2_pilot/slm_pipeline_responses.jsonl`: Real generated outputs from `src/v2/pipeline.py` (Decomposer $\to$ TaskAnalyser $\to$ TaskColorer $\to$ Matching $\to$ Scheduling $\to$ TwoStageAggregator).
   - `results/v2_pilot/llm_baseline_responses.jsonl`: Real generated outputs from the 5 baseline LLMs.
   - `results/v2_pilot/comparison.jsonl`: Reference pointers only (zero text duplication).
   - Implemented immediate per-call append with `os.fsync()` and idempotent resumability.
3. **Daily Quota Ceiling:** Hit Groq free-tier 200,000 TPD ceiling (199,935 tokens consumed). The 20-query pilot (120 generations + 200 pairwise judge calls = ~182k tokens) is configured and queued to execute as soon as the quota resets at 00:00 UTC.
