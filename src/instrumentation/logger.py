"""
Experiment Instrumentation Logger
AI Search Framework
Records structured logs for every pipeline stage and baseline run.
"""

import json
import os
import time
import subprocess
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional

class ExperimentLogger:
    def __init__(self, pricing_table_path: str = "config/pricing_table.json", log_dir: str = "logs/runs"):
        self.pricing_table_path = pricing_table_path
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.pricing_table = self._load_pricing_table()

    def _load_pricing_table(self) -> Dict[str, Any]:
        if os.path.exists(self.pricing_table_path):
            with open(self.pricing_table_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"models": {}}

    def _get_git_commit(self) -> str:
        try:
            res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
            return res.stdout.strip()
        except Exception:
            return "uncommitted_or_no_git"

    def compute_cost(self, model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
        models = self.pricing_table.get("models", {})
        pricing = models.get(model_name)
        if not pricing:
            # Fallback based on name heuristic
            if "70B" in model_name or "72B" in model_name:
                pricing = models.get("meta-llama/Llama-3.1-70B-Instruct", {"input_cost_per_1m_tokens": 0.80, "output_cost_per_1m_tokens": 0.90})
            elif "3B" in model_name or "3.2B" in model_name:
                pricing = models.get("default_slm_3b", {"input_cost_per_1m_tokens": 0.05, "output_cost_per_1m_tokens": 0.08})
            else:
                pricing = models.get("default_slm_8b", {"input_cost_per_1m_tokens": 0.15, "output_cost_per_1m_tokens": 0.20})
        
        inp_rate = pricing.get("input_cost_per_1m_tokens", 0.10) / 1_000_000.0
        out_rate = pricing.get("output_cost_per_1m_tokens", 0.15) / 1_000_000.0
        return (prompt_tokens * inp_rate) + (completion_tokens * out_rate)

    def create_run_record(
        self,
        run_id: str,
        system_type: str, # "slm_pipeline" or "llm_baseline"
        query_id: str,
        query_text: str,
        complexity_tier: str,
        seed: int,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "run_id": run_id,
            "system_type": system_type,
            "query_id": query_id,
            "query_text": query_text,
            "complexity_tier": complexity_tier,
            "seed": seed,
            "git_commit": self._get_git_commit(),
            "timestamp_start_utc": datetime.now(timezone.utc).isoformat(),
            "timestamp_end_utc": None,
            "total_wall_clock_latency_ms": 0.0,
            "simulated_parallel_latency_ms": 0.0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
            "stages": {},
            "dispatch_trace": [],
            "final_response": "",
            "config_snapshot": config
        }

    def record_stage(
        self,
        record: Dict[str, Any],
        stage_name: str,
        model_name: str,
        model_revision: str,
        start_time_s: float,
        end_time_s: float,
        prompt_tokens: int,
        completion_tokens: int,
        input_data: Any,
        output_data: Any,
        confidence_score: Optional[float] = None,
        extra_metadata: Optional[Dict[str, Any]] = None
    ):
        latency_ms = (end_time_s - start_time_s) * 1000.0
        cost_usd = self.compute_cost(model_name, prompt_tokens, completion_tokens)
        
        stage_entry = {
            "stage_name": stage_name,
            "model_name": model_name,
            "model_revision": model_revision,
            "start_time_rel_s": start_time_s,
            "end_time_rel_s": end_time_s,
            "latency_ms": latency_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost_usd": cost_usd,
            "confidence_score": confidence_score,
            "input": input_data,
            "output": output_data,
            "extra_metadata": extra_metadata or {}
        }
        
        if stage_name not in record["stages"]:
            record["stages"][stage_name] = []
        record["stages"][stage_name].append(stage_entry)

        # Update running totals
        record["total_prompt_tokens"] += prompt_tokens
        record["total_completion_tokens"] += completion_tokens
        record["total_tokens"] += (prompt_tokens + completion_tokens)
        record["total_cost_usd"] += cost_usd

    def finalize_run(self, record: Dict[str, Any], start_total_s: float, end_total_s: float, final_response: str) -> str:
        record["timestamp_end_utc"] = datetime.now(timezone.utc).isoformat()
        record["total_wall_clock_latency_ms"] = (end_total_s - start_total_s) * 1000.0
        record["final_response"] = final_response
        
        # Save to disk
        filename = f"run_{record['system_type']}_{record['query_id']}_{int(time.time()*1000)}.json"
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        
        return filepath

