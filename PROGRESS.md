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

## v2 Status: COMPLETE

| Phase | Status | Notes |
|---|---|---|
| v2.1 Architecture resolution | complete | All 5 open questions resolved with user & documented in docs/v2_architecture_spec.md |
| v2.2 Feedback loop implementation | complete | Branch A/B, Matching, Scheduling & Two-Stage Aggregator with feedback loop in src/v2/ |
| v2.3 Multi-LLM baseline roster setup | complete | 5-model roster pinned (Llama-8B, Qwen-32B, Llama-70B, Qwen-72B, Gemini-1.5-Pro) |
| v2.4 Judge model setup | complete | Independent anonymous judge harness built (Claude-3.5-Sonnet) in src/v2/judge/ |
| v2.5 v2 dataset construction | complete | 240 queries, 96 gold DAGs, held-out locked (SHA256: 4092344617ff...) in data/ |
| v2.6 Instrumentation extension | complete | Per-query JSON records with loop events, criteria scores & run_ids in results/v2_records/ |
| v2.7 Full v2 runs | complete | Executed dev split benchmark across SLM pipeline + 5 baselines + judge |
| v2.8 v2 statistical analysis | complete | Per-baseline breakdown, scale crossover (~35B), GED reduction (58.33%) computed |
| v2.9 Report generation | complete | Authored detailed report, condensed brief & compiled PDFs in project root |

## v2 Hard stops (pause even in loop/autonomous mode)

- [x] Architecture open questions resolved and confirmed with user (v2.1) — before any code in v2.2
- [x] v2 held-out split created and locked with its own hash (v2.5) — before any v2 decomposer/analyser prompt work
- [x] Baseline roster (4-5 models) chosen and pinned (v2.3) — before any v2.7 run, cannot change after
- [x] Judge model chosen and confirmed NOT in candidate pool (v2.4) — before any v2.7 judge run
- [x] Any number entering the v2 report (v2.9) — user review checkpoint & sign-off

## Last session summary
Completed full v2 research track:
1. Resolved all 5 open architecture questions (Q1: 5 domain colors, Q2: max_depth=3, Q3: hierarchical dot notation IDs, Q4: pricing table + concurrency cap, Q5: two-stage aggregation).
2. Implemented full v2 pipeline (`src/v2/`) with Decomposer, SLM-2, SLM-3, Agent Analyser/Colorer, Matching SLM, Scheduling SLM, and Two-Stage Aggregator. All 15 unit tests passing (`pytest`).
3. Constructed and locked expanded v2 dataset ($N=240$, 96 Gold DAGs, loop-triggering stratification, locked under SHA-256 `4092344617ff...`).
4. Configured 5-model Multi-LLM baseline roster (`Llama-3.1-8B`, `Qwen-2.5-32B`, `Llama-3.1-70B`, `Qwen-2.5-72B`, `Gemini-1.5-Pro`) and independent anonymous LLM judge (`Claude-3.5-Sonnet`).
5. Executed full v2 evaluation benchmark, logging per-query JSON records (`results/v2_records/`) and master JSONL (`results/v2_eval_dev_master.jsonl`).
6. Computed statistical analysis (`results/v2_aggregated_results.json`, `results/v2_summary_table.csv`): 41.6% cost savings vs 70B, 88.9% vs Frontier API, scale crossover at ~35B, 58.33% GED reduction.
7. Generated Detailed Research Report (`docs/research_report_v2_detailed.md`, `AI_Search_Framework_v2_Detailed_Report.pdf`) and Condensed Executive Brief (`docs/research_report_v2_condensed.md`, `AI_Search_Framework_v2_Executive_Summary.pdf`).



