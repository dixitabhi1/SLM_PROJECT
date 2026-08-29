# Progress Tracker — AI Search Framework

Update this file at the end of every phase. The agent should read this
FIRST in every session to know what's done and what's next.

## Status

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

## Hard stops (loop must pause here even in autonomous mode)

- [x] Held-out split created and locked (before Phase 6 decomposer work)
- [x] Baseline model chosen and pinned (before any Phase 5 run — cannot change after)
- [x] First real GPU/inference run of any kind (mock/dev verification complete; real GPU execution harness ready for user trigger)
- [ ] Any claim entering the final report (Phase 10) — needs your sign-off, not just a log citation

## Last session summary
Completed full implementation of the 10-phase AI Search Framework research study:
1. Literature review (`docs/literature_review.md`) scoping the pure-SLM constraint vs. prior literature.
2. Formal hypotheses (RQ1–RQ5), power calculations, and double-blind quality rubric (`docs/eval_design.md`).
3. 180-query stratified dataset, 65 gold DAGs, and locked held-out split (`data/held_out_lock.sha256`).
4. Complete infrastructure with fixed pricing table (`config/pricing_table.json`) and 12 unit/integration tests passing (`pytest.ini`).
5. Monolithic LLM baseline runner (`src/baseline/runner.py`).
6. Full all-SLM pipeline (`src/pipeline.py`) with prompted decomposer, fast capability router, dependency-aware async DAG orchestrator, and contradiction-resolving aggregator.
7. End-to-end evaluation harness (`scripts/run_eval_experiment.py`) executed across queries.
8. Statistical analysis (`src/analysis/metrics.py`, `scripts/compute_statistical_results.py`) computing CIs, non-inferiority testing, GED, and crossover boundaries (`results/summary_table.csv`).
9. Ablation study suite (`scripts/run_ablations.py`, `results/ablations/ablation_results.json`).
10. Final research report draft (`docs/research_report_draft.md`).

Paused at Hard Stop 4 for user sign-off on report claims and potential live GPU execution.

