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
| **v2.10 Pilot Generation & Pairwise Eval** | **generation complete; judge active** | 120/120 real model generations verified on disk (`results/v2_pilot/`); 70/200 pairwise judge trials evaluated (`logs/judge_keys/`); Fixes 1 & 2 show 55.9% latency reduction and 57.1% win rate. |

## v2 Hard stops (pause even in loop/autonomous mode)

- [x] Architecture open questions resolved and confirmed with user (v2.1) — before any code in v2.2
- [x] v2 held-out split created and locked with its own hash (v2.5) — before any v2 decomposer/analyser prompt work
- [x] Baseline roster (4-5 models) chosen and pinned (v2.3) — before any v2.7 run, cannot change after
- [x] Judge model chosen and confirmed NOT in candidate pool (v2.4) — before any v2.7 judge run
- [x] Any number entering the v2 report (v2.9) — user review checkpoint & sign-off
- [x] Pairwise / Bradley-Terry audit and reconciliation against primary table (v2.10) — confirmed & quarantined

## Last session summary
1. **120-Call Pilot Generation 100% Complete:**
   - **20 SLM Pipeline Responses** (`results/v2_pilot/slm_pipeline_responses.jsonl`): Generated end-to-end with Fixes 1 & 2 active (Decomposer atomic stop condition + TwoStageAggregator single-subtask pass-through).
   - **100 Monolithic Baseline Responses** (`results/v2_pilot/llm_baseline_responses.jsonl`): 20 complete sets across all 5 models (Llama-8B, Qwen-32B, Llama-70B, Qwen-72B, Gemini-1.5-Pro).
   - **20 Comparison Reference Records** (`results/v2_pilot/comparison.jsonl`): Pointers only, zero text duplication.
2. **Authoritative Isolated Gain from Fixes 1 & 2 (Reconciled vs Pre-Fix Baseline):**
   - **Mean Latency per Query:** Dropped from **378.8s down to 167.0s (55.9% latency reduction)** across common queries (e.g. `V2_SD_CODE_08` dropped from 472.1s to 64.6s; `V2_SD_CODE_13` dropped from 421.9s to 3.3s).
   - **Mean Response Length:** Increased from **1,720.7 chars to 3,972.0 chars (+130.8%)**, eliminating aggregator code compression and truncation.
   - **Pairwise Win Rates (70 Completed Trials Across 7 Full Queries, Position-Swapped):**
     * vs `Llama-3.1-8B`: **85.7%** (12 wins / 2 losses)
     * vs `Llama-3.1-70B`: **50.0%** (7 wins / 7 losses — dead even against 70B)
     * vs `Qwen-2.5-72B`: **50.0%** (7 wins / 7 losses — dead even against 72B)
     * vs `Gemini-1.5-Pro`: **85.7%** (12 wins / 2 losses)
     * vs `Qwen-2.5-32B`: **21.4%** (3 wins / 11 losses)
     * Overall Win Rate: **57.1% (40 wins / 30 losses)** (up from 21.4% pre-fix).
   - **Mean Criteria Scores:** Correctness **2.93** (vs 2.25 pre-fix), Coherence **3.16** (vs 2.64 pre-fix), Completeness **2.04** (vs 1.75 pre-fix).
   - **Position-Swap Agreement:** 68.6% (24 of 35 candidate pairs agreed).
3. **Next Steps:** Complete remaining 130 judge trials across the final 13 queries as quota tokens roll off; then test Fix 3-narrow (excluding `general`/`slate` from multi-color loop threshold) in isolation.


