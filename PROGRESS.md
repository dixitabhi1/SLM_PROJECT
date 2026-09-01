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

## v2 Status: COMPLETE (with Active Option C Multi-Day Pairwise Extension)

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
| **v2.10 Pairwise Audit & Extension** | **active (Option C)** | Audited synthetic matrix; quarantined unverified scripts; pinned Groq judge `openai/gpt-oss-120b` for 2,400 round-robin evaluations managed via `scripts/daily_pairwise_runner.py` (00:15 UTC cron). |

## v2 Hard stops (pause even in loop/autonomous mode)

- [x] Architecture open questions resolved and confirmed with user (v2.1) — before any code in v2.2
- [x] v2 held-out split created and locked with its own hash (v2.5) — before any v2 decomposer/analyser prompt work
- [x] Baseline roster (4-5 models) chosen and pinned (v2.3) — before any v2.7 run, cannot change after
- [x] Judge model chosen and confirmed NOT in candidate pool (v2.4) — before any v2.7 judge run
- [x] Any number entering the v2 report (v2.9) — user review checkpoint & sign-off
- [x] Pairwise / Bradley-Terry audit and reconciliation against primary table (v2.10) — confirmed & quarantined

## Last session summary
1. **Audit & Quarantine:** Audited `results/v2_pairwise_win_matrix.csv`, `results/v2_criteria_breakdown.csv`, and `results/v2_judge_deep_analysis.json` against primary ground-truth `results/v2_aggregated_results.json`. Quarantined heuristic script to `scripts/_DO_NOT_USE_compute_deep_judge_analysis.py.bak` and purged untraced sections from PDF reports (Commit `1566dd2`).
2. **Reconciliation Rule Added:** Hardened `.agents/skills/statistical-analysis/SKILL.md` requiring all derived statistics to include explicit reconciliation against primary win-count tables and show per-query derivations on request.
3. **Real Pairwise Protocol (Option C Multi-Day Quota Manager):**
   - Configured Groq independent judge model `openai/gpt-oss-120b` (120B parameter dense model outside candidate pool) with separated identity logging (`logs/judge_pairwise/` and `logs/judge_keys/`).
   - Throughput optimized from 12.2 RPM to 26.0 RPM with `max_tokens=1024`.
   - Free-tier 200k TPD ceiling hit at call #276. In accordance with user decision to preserve judge model consistency across the study, configured `scripts/daily_pairwise_runner.py` to consume daily quota increments automatically.
   - Scheduled daily runner cron job at 00:15 UTC with daily progress tracking in `results/v2_daily_pairwise_progress.json` (Estimated completion: ~September 9, 2026).



