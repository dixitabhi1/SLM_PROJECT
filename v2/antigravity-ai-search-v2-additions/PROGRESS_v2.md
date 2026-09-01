# Progress Tracker — v2 (append after v1 section in your existing PROGRESS.md)

## v1 status: COMPLETE, LOCKED

v1 (single 70B baseline, no feedback loop, rubric-only quality scoring,
N=180) is finished. Results in `.agents/knowledge/Research_Report_v1_source.txt`.
Do not reopen v1 phases.

## v2 status

| Phase | Status | Notes |
|---|---|---|
| v2.1 Architecture resolution | not started | Resolve open questions in Proposed_Architecture_v2_source.txt (color taxonomy, depth limit, task-ID versioning, cost model, Aggregator placement) BEFORE building |
| v2.2 Feedback loop implementation | not started | Decomposer/SLM-2/SLM-3/Matching/Scheduling + loop-back condition |
| v2.3 Multi-LLM baseline roster setup | not started | Choose + pin 4-5 baseline models spanning param sizes |
| v2.4 Judge model setup | not started | Choose judge model NOT in candidate pool; build blind/shuffle harness |
| v2.5 v2 dataset construction | not started | Larger N, new stratification axis (loop-triggering), new held-out lock, JSON schema |
| v2.6 Instrumentation extension | not started | New log fields: baseline_model_id, loop events, judge records |
| v2.7 Full v2 runs | not started | SLM pipeline + all baselines + judge, on v2 eval set |
| v2.8 v2 statistical analysis | not started | Per-baseline breakdown, judge win-rates, loop effectiveness, v1-v2 comparability |
| v2.9 Report generation | not started | Detailed + condensed, from same results files |

## v2 hard stops (pause even in loop/autonomous mode)

- [ ] Architecture open questions resolved and confirmed with user (v2.1) — before any code in v2.2
- [ ] v2 held-out split created and locked with its own hash (v2.5) — before any v2 decomposer/analyser prompt work
- [ ] Baseline roster (4-5 models) chosen and pinned (v2.3) — before any v2.7 run, cannot change after
- [ ] Judge model chosen and confirmed NOT in candidate pool (v2.4) — before any v2.7 judge run
- [ ] Any number entering the v2 report (v2.9) — needs user sign-off, same as v1's Phase 10 rule

## Last v2 session summary
(agent fills this in at the end of each session)
