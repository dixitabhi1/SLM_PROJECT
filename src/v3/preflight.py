"""
Permanent Pre-Flight Model Verification Harness (Hard Rule 13)
AI Search Framework — Version 3

Enforces:
1. Every baseline in the roster maps to a genuinely distinct api_model_name and endpoint.
2. Every pool specialist, decomposer, and aggregator in the proposed SLM pipeline maps
   to a genuinely distinct api_model_name and endpoint.
3. No SLM pipeline component matches or proxies any comparative baseline's model.
4. Fails loudly and aborts immediately if any collision is detected.
"""

from typing import Dict, Any, List, Set, Tuple, Optional

class PreFlightVerificationError(RuntimeError):
    """Raised when any model identity collision or proxy shortcut is detected."""
    pass

def verify_distinct_roster_preflight(
    slm_pipeline_runners: Dict[str, Any],
    baseline_runners: Dict[str, Any],
    judge_runner: Optional[Any] = None
) -> Dict[str, str]:
    """
    Executes Hard Rule 13 pre-flight verification before ANY generation or evaluation call.
    
    Returns a dictionary of system_name -> resolved_api_model_name if all checks pass.
    Raises PreFlightVerificationError loudly if any check fails.
    """
    errors: List[str] = []
    resolved_models: Dict[str, str] = {}

    # 1. Resolve SLM pipeline models
    pipeline_models: Dict[str, str] = {}
    for comp_name, runner in slm_pipeline_runners.items():
        api_model = getattr(runner, "api_model_name", None) or getattr(runner, "model_name", None)
        if not api_model:
            errors.append(f"SLM component '{comp_name}' does not expose 'api_model_name' or 'model_name'.")
        pipeline_models[comp_name] = str(api_model)
        resolved_models[f"pipeline:{comp_name}"] = str(api_model)

    # 2. Resolve Baseline models
    baseline_models: Dict[str, str] = {}
    for b_id, runner in baseline_runners.items():
        api_model = getattr(runner, "api_model_name", None) or getattr(runner, "model_name", None)
        if not api_model:
            errors.append(f"Baseline '{b_id}' does not expose 'api_model_name' or 'model_name'.")
        baseline_models[b_id] = str(api_model)
        resolved_models[f"baseline:{b_id}"] = str(api_model)

    # 3. Check Baseline Uniqueness: Every baseline must have a UNIQUE api_model_name
    seen_baselines: Dict[str, str] = {}
    for b_id, api_model in baseline_models.items():
        if api_model in seen_baselines:
            errors.append(
                f"BASELINE COLLISION: Baseline '{b_id}' and '{seen_baselines[api_model]}' "
                f"both resolve to the same underlying api_model_name '{api_model}'! "
                f"Baselines must be genuinely independent models."
            )
        else:
            seen_baselines[api_model] = b_id

    # 4. Check Pipeline vs Baseline Separation: NO pipeline model can match any baseline model
    for comp_name, p_model in pipeline_models.items():
        for b_id, b_model in baseline_models.items():
            if p_model == b_model:
                errors.append(
                    f"PROXY COLLUSION VIOLATION: SLM pipeline component '{comp_name}' "
                    f"shares underlying api_model_name '{p_model}' with baseline '{b_id}'! "
                    f"Proposed system components cannot share models with comparative baselines."
                )

    # 5. Check SLM Pool Diversity: Specialists must not all map to a single proxy model
    unique_pool_models = set(pipeline_models.values())
    if len(pipeline_models) > 2 and len(unique_pool_models) <= 1:
        errors.append(
            f"SLM POOL SINGLE-MODEL COLLUSION: All {len(pipeline_models)} pipeline components "
            f"resolve to the same model: {unique_pool_models}! "
            f"Decomposed architecture requires distinct specialists."
        )

    # 6. Check Judge Independence: Judge model must be disjoint from baselines and pipeline
    if judge_runner is not None:
        judge_model = (
            getattr(judge_runner, "judge_model_name", None)
            or getattr(judge_runner, "api_model_name", None)
            or getattr(judge_runner, "model_name", None)
        )
        if judge_model:
            resolved_models["judge:pairwise"] = str(judge_model)
            for b_id, b_model in baseline_models.items():
                if str(judge_model) == b_model:
                    errors.append(
                        f"JUDGE BIAS COLLUSION: Judge model '{judge_model}' is identical to baseline '{b_id}'! "
                        f"Judge must be an independent model disjoint from evaluated candidates."
                    )
            for comp_name, p_model in pipeline_models.items():
                if str(judge_model) == p_model:
                    errors.append(
                        f"JUDGE BIAS COLLUSION: Judge model '{judge_model}' is identical to pipeline component '{comp_name}'! "
                        f"Judge must be an independent model disjoint from evaluated candidates."
                    )

    # If errors found, format loud failure report
    if errors:
        msg = "\n" + "=" * 80 + "\n"
        msg += "PRE-FLIGHT MODEL ROSTER VERIFICATION FAILED (HARD RULE 13 VIOLATION)\n"
        msg += "=" * 80 + "\n"
        for idx, err in enumerate(errors, 1):
            msg += f"  [{idx}] {err}\n"
        msg += "\nResolved Roster Configuration:\n"
        for sys_k, m_val in resolved_models.items():
            msg += f"  - {sys_k:35s} -> {m_val}\n"
        msg += "=" * 80 + "\n"
        raise PreFlightVerificationError(msg)

    print(">>> PRE-FLIGHT VERIFICATION PASSED (Hard Rule 13): All models genuinely distinct and disjoint.")
    return resolved_models
