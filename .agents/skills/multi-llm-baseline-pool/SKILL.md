---
name: multi-llm-baseline-pool
description: Setting up multi-tier baseline ladders (20B, 32B, 72B, 120B), pre-flight fairness constraint verification (baseline params > combined pool params), and live endpoint health checking.
---

# Multi-LLM Baseline Pool & Fairness Governance

Source of truth: `.agents/knowledge/mentor_experiment_protocol_source.txt` §2, §8 and `TRD_source.txt` §4.

## Four-Tier Baseline Ladder

For the Mentor Experiment Protocol (E1–E4), comparisons evaluate across a calibrated four-tier baseline hierarchy:

| Tier ID | Nominal Size | Model Checkpoint / Endpoint | Provider / Host | Parameter Count |
|---|---|---|---|---|
| Tier 1 | ~20B | `openai/gpt-oss-20b` | Groq Cloud LPU API | ~20B |
| Tier 2 | ~32B | `gemini-2.5-flash` | Google Gemini API | ~32B equivalent |
| Tier 3 | ~72B | `Qwen/Qwen2.5-72B-Instruct` | Hugging Face Router API | 72.7B |
| Tier 4 | ~120B | `openai/gpt-oss-120b` | Groq Cloud LPU API | ~120B |

## Mandatory Pre-Flight Fairness Constraint Check

Before initiating any evaluation run under the Mentor Experiment Protocol, the runner MUST execute an automated pre-flight assertion verifying the fairness constraint:

```python
assert baseline_params > combined_pool_params, (
    f"Fairness violation: Baseline ({baseline_params}B) must strictly exceed "
    f"the combined parameter count of participating SLMs ({combined_pool_params}B)!"
)
```

- For every query execution, calculate the sum of parameters across all active SLM components participating in that query.
- The comparative baseline model's parameter count must strictly exceed this sum across every tier, particularly the smallest tier (~20B).

## Live Endpoint Availability Verification

Never assume a prior session's catalog snapshot or endpoint health remains valid. Before launching generation or judging passes:
1. Dispatch an automated ping / probe query to all four baseline endpoints (`openai/gpt-oss-20b`, `gemini-2.5-flash`, `Qwen/Qwen2.5-72B-Instruct`, and `openai/gpt-oss-120b`).
2. Verify HTTP 200 responses, non-empty text generation, and absence of rate-limit / token exhaustion errors.
3. Fail loudly and abort pre-flight if any tier is degraded or offline.

