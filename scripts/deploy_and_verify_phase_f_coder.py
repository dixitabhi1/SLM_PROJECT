"""
Phase F-B: Deployment, Mechanical Verification, and Symmetrical Pairwise Evaluation
for Fine-Tuned Coding Specialist (phi3.5-ft-coding:latest).

Workflow:
1. Converts trained LoRA adapter to GGUF via scratch/llama_cpp/convert_lora_to_gguf.py.
2. Registers phi3.5-ft-coding in Ollama with CPU execution parameters.
3. Tests both base (phi3.5:cpu) and fine-tuned (phi3.5-ft-coding:latest) models on
   held-out coding queries (data/phase_f/coding_qa_eval_held_out.json).
4. Runs deterministic AST syntax validation and sandboxed execution via MechanicalCodeVerifier.
5. Executes symmetrical double-blind pairwise judging via qwen/qwen3.8-27b on Groq LPU.
6. Enforces Autonomous Audit Protocol (concordance, truncation diagnostic, swap consistency).
"""

import os
import sys
import json
import time
import asyncio
import subprocess
import glob
from typing import Dict, Any, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.ollama_runner import OllamaModelRunner
from src.v4.tools.code_verifier import MechanicalCodeVerifier
from src.v2.judge.pairwise_harness import PairwiseLLMJudgeHarness

ADAPTER_SRC = "models/phi35_coding_adapter"
GGUF_OUT = "models/phi3.5_ft_coding_adapter.gguf"
MODELFILE_PATH = "models/Modelfile.phi35_ft_coding"
EVAL_DATASET_PATH = "data/phase_f/coding_qa_eval_held_out.json"
RESULTS_DIR = "results/phase_f"
JUDGE_LOG_DIR = "logs/phase_f_judge_pairwise"
KEY_LOG_DIR = "logs/phase_f_judge_keys"

def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(JUDGE_LOG_DIR, exist_ok=True)
    os.makedirs(KEY_LOG_DIR, exist_ok=True)

def convert_adapter_to_gguf():
    print("=" * 80)
    print("[1/5] CONVERTING LORA SAFETENSORS TO GGUF FORMAT")
    print("=" * 80)
    
    if os.path.exists(GGUF_OUT):
        size_mb = os.path.getsize(GGUF_OUT) / (1024 * 1024)
        print(f"GGUF adapter already exists: {GGUF_OUT} ({size_mb:.1f} MB). Skipping conversion.")
        return

    script_path = "scratch/llama_cpp/convert_lora_to_gguf.py"
    assert os.path.exists(script_path), f"Converter missing: {script_path}"
    assert os.path.exists(ADAPTER_SRC), f"Adapter directory missing: {ADAPTER_SRC}"

    cmd = [
        sys.executable,
        script_path,
        "--base-model-id", "microsoft/Phi-3.5-mini-instruct",
        "--outfile", GGUF_OUT,
        ADAPTER_SRC
    ]
    print(f"Running conversion: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"STDERR: {res.stderr}")
        raise RuntimeError(f"GGUF conversion failed with code {res.returncode}")
    
    assert os.path.exists(GGUF_OUT), "GGUF output file was not created!"
    size_mb = os.path.getsize(GGUF_OUT) / (1024 * 1024)
    print(f"Conversion successful! GGUF adapter: {GGUF_OUT} ({size_mb:.1f} MB)")

def register_ollama_model():
    print("\n" + "=" * 80)
    print("[2/5] REGISTERING phi3.5-ft-coding IN OLLAMA")
    print("=" * 80)

    modelfile_content = f"""FROM phi3.5:cpu
ADAPTER ./{os.path.basename(GGUF_OUT)}
PARAMETER num_gpu 0
PARAMETER temperature 0.0
PARAMETER num_predict 1024
PARAMETER stop "<|end|>"
PARAMETER stop "<|user|>"
PARAMETER stop "<|system|>"
TEMPLATE \"\"\"{{{{ if .System }}}}<|system|>
{{{{ .System }}}}<|end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|user|>
{{{{ .Prompt }}}}<|end|>
{{{{ end }}}}<|assistant|>
{{{{ .Response }}}}<|end|>\"\"\"
"""
    with open(MODELFILE_PATH, "w", encoding="utf-8") as f:
        f.write(modelfile_content)
    print(f"Created Modelfile at {MODELFILE_PATH}")

    # Run ollama create
    ollama_exe = r"C:\Users\ACER\AppData\Local\Programs\Ollama\ollama.exe"
    cmd = [ollama_exe, "create", "phi3.5-ft-coding:latest", "-f", MODELFILE_PATH]
    print(f"Registering model: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"STDERR: {res.stderr}")
        raise RuntimeError(f"Ollama registration failed with code {res.returncode}")
    print("Ollama model 'phi3.5-ft-coding:latest' registered successfully!")

async def evaluate_models_mechanically(num_eval_samples: int = 10) -> List[Dict[str, Any]]:
    print("\n" + "=" * 80)
    print(f"[3/5] RUNNING MECHANICAL CODE VERIFICATION ({num_eval_samples} HELD-OUT SAMPLES)")
    print("=" * 80)

    with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
        held_out = json.load(f)

    samples = held_out[:num_eval_samples]
    verifier = MechanicalCodeVerifier(execution_timeout_sec=8.0)

    base_runner = OllamaModelRunner(
        logical_model_name="phi3.5-base",
        api_model_name="phi3.5:cpu",
        max_tokens=1024,
        timeout_sec=300.0
    )

    ft_runner = OllamaModelRunner(
        logical_model_name="phi3.5-ft-coding",
        api_model_name="phi3.5-ft-coding:latest",
        max_tokens=1024,
        timeout_sec=300.0
    )

    results = []

    for idx, item in enumerate(samples, 1):
        qid = item["id"]
        prompt = item["prompt"]
        category = item["category"]
        print(f"\n[{idx}/{len(samples)}] Testing Query: {qid} ({category})")
        print(f"  Prompt: {prompt[:80]}...")

        # 1. Base Model Generation
        t0 = time.perf_counter()
        resp_base = await base_runner.generate(prompt)
        base_lat = time.perf_counter() - t0
        base_text = resp_base.text

        # Base Verification
        base_blocks = verifier.extract_python_blocks(base_text)
        base_code = base_blocks[0] if base_blocks else base_text
        base_pass, base_err, base_out = verifier.verify_execution(base_code)

        print(f"  [Base Model] Latency: {base_lat:.1f}s | Exec Pass: {base_pass} | Err: {base_err[:60] if base_err else 'None'}")

        # 2. Fine-Tuned Model Generation
        t1 = time.perf_counter()
        resp_ft = await ft_runner.generate(prompt)
        ft_lat = time.perf_counter() - t1
        ft_text = resp_ft.text

        # FT Verification
        ft_blocks = verifier.extract_python_blocks(ft_text)
        ft_code = ft_blocks[0] if ft_blocks else ft_text
        ft_pass, ft_err, ft_out = verifier.verify_execution(ft_code)

        print(f"  [FT Model]   Latency: {ft_lat:.1f}s | Exec Pass: {ft_pass} | Err: {ft_err[:60] if ft_err else 'None'}")

        results.append({
            "query_id": qid,
            "category": category,
            "prompt": prompt,
            "base_response": base_text,
            "base_code": base_code,
            "base_latency_sec": base_lat,
            "base_exec_passed": base_pass,
            "base_error": base_err,
            "base_stdout": base_out,
            "ft_response": ft_text,
            "ft_code": ft_code,
            "ft_latency_sec": ft_lat,
            "ft_exec_passed": ft_pass,
            "ft_error": ft_err,
            "ft_stdout": ft_out
        })

    # Save mechanical verification log
    mech_log_path = os.path.join(RESULTS_DIR, "coding_ft_mechanical_verification.json")
    with open(mech_log_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    base_pass_rate = sum(1 for r in results if r["base_exec_passed"]) / len(results) * 100.0
    ft_pass_rate = sum(1 for r in results if r["ft_exec_passed"]) / len(results) * 100.0
    avg_base_lat = sum(r["base_latency_sec"] for r in results) / len(results)
    avg_ft_lat = sum(r["ft_latency_sec"] for r in results) / len(results)

    print("\n" + "-" * 60)
    print("MECHANICAL EXECUTION VERIFICATION SUMMARY:")
    print(f"  Base Model Pass Rate:  {base_pass_rate:.1f}% | Avg Latency: {avg_base_lat:.1f}s")
    print(f"  FT Model Pass Rate:    {ft_pass_rate:.1f}% | Avg Latency: {avg_ft_lat:.1f}s")
    print("-" * 60)

    return results

async def run_pairwise_judge_eval(mechanical_results: List[Dict[str, Any]]):
    print("\n" + "=" * 80)
    print("[4/5] RUNNING SYMMETRICAL DOUBLE-BLIND PAIRWISE EVALUATION (GROQ LPU)")
    print("=" * 80)

    judge_harness = PairwiseLLMJudgeHarness(judge_model_name="qwen/qwen3.8-27b")
    trials = []

    for idx, r in enumerate(mechanical_results, 1):
        qid = r["query_id"]
        prompt = r["prompt"]
        ft_text = r["ft_response"]
        base_text = r["base_response"]

        print(f"\n[{idx}/{len(mechanical_results)}] Pairwise judging for {qid}...")

        # Forward presentation: A = FT, B = Base
        print("  -> Forward presentation (A = FT, B = Base)...")
        res_f = await asyncio.to_thread(
            judge_harness.evaluate_pair,
            query_id=qid,
            query_text=prompt,
            system_a_id="phi3.5_coder_ft",
            text_a=ft_text,
            system_b_id="phi3.5_coder_base",
            text_b=base_text,
            order_tag="forward",
            judge_log_dir=JUDGE_LOG_DIR,
            key_log_dir=KEY_LOG_DIR
        )
        print(f"     Winner: {res_f['unblinded_winner']}")
        await asyncio.sleep(2.0)

        # Swapped presentation: A = Base, B = FT
        print("  -> Swapped presentation (A = Base, B = FT)...")
        res_s = await asyncio.to_thread(
            judge_harness.evaluate_pair,
            query_id=qid,
            query_text=prompt,
            system_a_id="phi3.5_coder_ft",
            text_a=ft_text,
            system_b_id="phi3.5_coder_base",
            text_b=base_text,
            order_tag="swapped",
            judge_log_dir=JUDGE_LOG_DIR,
            key_log_dir=KEY_LOG_DIR
        )
        print(f"     Winner: {res_s['unblinded_winner']}")
        await asyncio.sleep(2.0)

        trials.append((qid, res_f, res_s))

    # Autonomous Audit Analysis
    print("\n" + "=" * 80)
    print("[5/5] AUTONOMOUS AUDIT LOOP ANALYSIS")
    print("=" * 80)

    audited_trials = []
    swap_agreements = 0

    for qid, fwd, swp in trials:
        for order_name, trial in [("forward", fwd), ("swapped", swp)]:
            pub_log = trial["public_log"]
            with open(pub_log, "r", encoding="utf-8") as f:
                j = json.load(f)

            scores = j["criteria_scores"]
            sel = j["selected_candidate"]
            sc_a = scores["Candidate A"]["correctness"] + scores["Candidate A"]["completeness"] + scores["Candidate A"]["coherence"]
            sc_b = scores["Candidate B"]["correctness"] + scores["Candidate B"]["completeness"] + scores["Candidate B"]["coherence"]
            expected_sel = "Candidate A" if sc_a > sc_b else ("Candidate B" if sc_b > sc_a else "Tie")
            concordance = (sel == expected_sel)

            exp = j.get("reasoning", "")
            trunc = any(w in exp.lower() for w in ["truncat", "cut off", "abrupt"])

            audited_trials.append({
                "query_id": qid,
                "order": order_name,
                "winner": trial["unblinded_winner"],
                "concordance": concordance,
                "truncation_flagged": trunc,
                "public_log": pub_log,
                "key_log": trial["key_log"]
            })

        if fwd["unblinded_winner"] == swp["unblinded_winner"]:
            swap_agreements += 1

    total_trial_count = len(audited_trials)
    concordance_rate = sum(1 for t in audited_trials if t["concordance"]) / total_trial_count
    swap_consistency = swap_agreements / len(trials)
    truncation_free = sum(1 for t in audited_trials if not t["truncation_flagged"]) / total_trial_count

    ft_wins = sum(1 for t in audited_trials if t["winner"] == "phi3.5_coder_ft")
    base_wins = sum(1 for t in audited_trials if t["winner"] == "phi3.5_coder_base")
    ties = sum(1 for t in audited_trials if t["winner"] == "Tie")

    summary = {
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "total_queries": len(trials),
        "total_trials": total_trial_count,
        "concordance_rate_pct": round(concordance_rate * 100.0, 1),
        "swap_consistency_pct": round(swap_consistency * 100.0, 1),
        "truncation_free_pct": round(truncation_free * 100.0, 1),
        "ft_wins": ft_wins,
        "base_wins": base_wins,
        "ties": ties,
        "ft_win_rate_pct": round(ft_wins / total_trial_count * 100.0, 1),
        "trials": audited_trials
    }

    summary_path = os.path.join(RESULTS_DIR, "coding_ft_audit_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Audit Summary saved to {summary_path}")
    print(f"  Concordance: {summary['concordance_rate_pct']}%")
    print(f"  Swap Consistency: {summary['swap_consistency_pct']}%")
    print(f"  Truncation Free: {summary['truncation_free_pct']}%")
    print(f"  FT Wins: {ft_wins}/{total_trial_count} ({summary['ft_win_rate_pct']}%)")
    print(f"  Base Wins: {base_wins}/{total_trial_count}")
    print(f"  Ties: {ties}/{total_trial_count}")

async def main():
    ensure_dirs()
    convert_adapter_to_gguf()
    register_ollama_model()
    mechanical_results = await evaluate_models_mechanically(num_eval_samples=10)
    await run_pairwise_judge_eval(mechanical_results)

if __name__ == "__main__":
    asyncio.run(main())

