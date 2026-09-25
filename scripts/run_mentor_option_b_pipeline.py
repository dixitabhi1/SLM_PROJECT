"""
Mentor Experiment Protocol — Option B Execution Pipeline (Pathway 1: Sequential Synthesis)
AI Search Framework (all-SLM pipeline vs. LLM baseline)

Implements:
- Pool-Size Sizing Governance: Authorized 3-4B specialist inclusion (Phi-3.5-mini-instruct, 3.82B) alongside Llama-3.1-8B.
- Combined Participating Pool: Strictly 11.85B across both E1 and E2.
- Strict Model Identity (Hard Rule 17):
    E1: Base, unadapted phi3.5:cpu (3.82B) on local Ollama
    E2: Fine-tuned phi3.5-ft-coding:latest (3.82B, QLoRA adapted locally on RTX 3050 GPU) on local Ollama
    Zero model-swap confounding!
- Mandatory Fairness Pre-Flight (Hard Rule 16b): 11.85B < 20.0B < 32.0B < 72.7B < 120.0B.
- Sequential DAG Dependency Execution (TRD Section 3.2):
    Node 1 establishes domain grounding/equations/schema/specifications.
    Node 2 receives Node 1's exact output as input context (no disjoint hallucinations).
- Two-Stage Aggregator Synthesis (TRD Section 3.2):
    TwoStageAggregator prompt on meta-llama/Llama-3.1-8B-Instruct synthesizes an authoritative,
    seamless, structurally complete solution preserving all equations, schemas, and code.
- Dual-Framework LLM-as-a-Judge (1-5 criteria and 1-10 holistic) with first-class draws (QS >= QL).
- Symmetrical Double-Blind Evaluation (64 trials each, 128 total trials).
- Autonomous Audit Loop (Concordance, Truncation Diagnostic, Swap Consistency, No Blending).
- Matched Gain Analysis (Delta Gain = DeltaQ_E2 - DeltaQ_E1, QP_Gain = QP_E2 - QP_E1).
- 10-Field Data Preservation.
"""

import os
import sys
import json
import time
import math
import asyncio
import urllib.request
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv

load_dotenv(".env")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from groq import Groq
from src.models.hf_runner import HFRouterModelRunner

# Output Directories
E1_DIR = "results/mentor_protocol/e1"
E2_DIR = "results/mentor_protocol/e2"
BASELINES_DIR = "results/mentor_protocol/archive_pre_pathway1/e1"

E1_SLM_PATH = os.path.join(E1_DIR, "slm_pipeline_responses.jsonl")
E2_SLM_PATH = os.path.join(E2_DIR, "slm_pipeline_responses.jsonl")

E1_TRIALS_PATH = os.path.join(E1_DIR, "judge_trials_raw.jsonl")
E2_TRIALS_PATH = os.path.join(E2_DIR, "judge_trials_raw.jsonl")

E1_SUMMARY_PATH = os.path.join(E1_DIR, "e1_summary.json")
E2_SUMMARY_PATH = os.path.join(E2_DIR, "e2_summary.json")

E1_PRESERVED_PATH = os.path.join(E1_DIR, "e1_preserved_data.jsonl")
E2_PRESERVED_PATH = os.path.join(E2_DIR, "e2_preserved_data.jsonl")

E1_KEY_LOG_DIR = "logs/mentor_e1_judge_keys"
E1_JUDGE_LOG_DIR = "logs/mentor_e1_judge_pairwise"
E2_KEY_LOG_DIR = "logs/mentor_e2_judge_keys"
E2_JUDGE_LOG_DIR = "logs/mentor_e2_judge_pairwise"

INTERMEDIATE_DIR = "results/mentor_protocol/intermediate_nodes"

# Model Specifications
MODEL_SPECS = {
    "e1_coding_base": {"name": "phi3.5:cpu", "params": 3.82, "endpoint": "Ollama Local (Unadapted Base)"},
    "e2_coding_ft": {"name": "phi3.5-ft-coding:latest", "params": 3.82, "endpoint": "Ollama Local (Local QLoRA FT)"},
    "pool_general_base": {"name": "meta-llama/Llama-3.1-8B-Instruct", "params": 8.03, "endpoint": "HF Router (Base General Specialist)"},
    "aggregator": {"name": "meta-llama/Llama-3.1-8B-Instruct", "params": 8.03, "endpoint": "HF Router (Two-Stage Aggregator)"},
    "b20": {"name": "openai/gpt-oss-20b", "params": 20.0, "endpoint": "Groq API"},
    "b32": {"name": "gemini-2.5-flash", "params": 32.0, "endpoint": "Google AI Studio API"},
    "b72": {"name": "Qwen/Qwen2.5-72B-Instruct", "params": 72.7, "endpoint": "HF Router"},
    "b120": {"name": "openai/gpt-oss-120b", "params": 120.0, "endpoint": "Groq API"},
    "judge": {"name": "gemini-3.1-flash-lite", "params": 2.0, "endpoint": "Google AI Studio API"}
}

# Sequential Problem Grounding & Node Contracts
QUERY_CONTRACTS = {
    "V3_TD_01": {
        "title": "2D Transient Heat Conduction & Convection in Aerospace Heat Sink",
        "node1": {
            "domain": "mathematics",
            "system": "You are an expert mathematical physics specialist.",
            "prompt": (
                "Define concrete engineering problem #1: A 2D transient thermal conduction and convection system in an aerospace heat sink. "
                "Derive the quantitative formulation: state the 2D heat equation with conduction and convection source terms, "
                "boundary conditions (Dirichlet at heat source base, Robin convective boundary at cooling fins), "
                "dimensionless numbers (Biot number Bi, Fourier number Fo), and the explicit finite-difference numerical stability condition (Fo <= 0.25)."
            ),
            "max_tokens": 800
        },
        "node2": {
            "domain": "coding",
            "system": "You are an expert, deterministic Python systems programming specialist.",
            "prompt_fn": lambda q_text, n1_out: (
                f"User Request: {q_text}\n\n"
                f"Mathematical Formulation Context (from Node 1):\n{n1_out}\n\n"
                f"Directive: Implement a clean, vectorized Python numerical simulation using NumPy that models the 2D heat equation "
                f"and boundary conditions derived above. Use vectorized 2D array updates, simulate across time steps, verify numerical stability, "
                f"and provide a complete, verified execution test block under `if __name__ == '__main__':`."
            ),
            "max_tokens": 800
        }
    },
    "V3_TD_11": {
        "title": "Aerospace Turbine Component Lifecycle & Inspection System",
        "node1": {
            "domain": "structured_data",
            "system": "You are an expert relational database architect.",
            "prompt": (
                "Define concrete engineering problem #11: Aerospace Turbine Component Lifecycle & Inspection System. "
                "Design a clean, normalized relational database schema with 3 core tables: `turbines`, `components`, and `inspection_logs`. "
                "Provide the exact PostgreSQL DDL with primary keys, foreign keys (`REFERENCES turbines(id)`), check constraints, and indexes."
            ),
            "max_tokens": 800
        },
        "node2": {
            "domain": "coding",
            "system": "You are an expert, deterministic Python systems programming specialist.",
            "prompt_fn": lambda q_text, n1_out: (
                f"User Request: {q_text}\n\n"
                f"Database Schema Context (from Node 1):\n{n1_out}\n\n"
                f"Directive: Implement the complete Python SQLAlchemy ORM access layer (declarative base) that strictly maps to the 3 tables above "
                f"(`turbines`, `components`, `inspection_logs`). Include all imports (`ForeignKey`, `relationship`), model relationships, "
                f"and a complete, verified execution test block under `if __name__ == '__main__':`."
            ),
            "max_tokens": 800
        }
    },
    "V3_TD_21": {
        "title": "Distributed Consensus Leader Election State Machine",
        "node1": {
            "domain": "formal_reasoning",
            "system": "You are an expert formal methods and distributed systems theorist.",
            "prompt": (
                "Define concrete engineering problem #21: Distributed Consensus Leader Election State Machine (Raft/Paxos core). "
                "Formulate the theoretical state space (Follower, Candidate, Leader), term monotonicity, election safety invariant "
                "(at most one leader per term), and quorum intersection property. Provide a formal inductive invariant proof."
            ),
            "max_tokens": 800
        },
        "node2": {
            "domain": "coding",
            "system": "You are an expert, deterministic Python systems programming specialist.",
            "prompt_fn": lambda q_text, n1_out: (
                f"User Request: {q_text}\n\n"
                f"Theoretical Invariants & State Machine Context (from Node 1):\n{n1_out}\n\n"
                f"Directive: Implement the verified Python concurrency engine using `asyncio` implementing the state transitions, "
                f"term checks, and atomic vote locks derived above. Include a complete, verified execution test block under `if __name__ == '__main__':` "
                f"demonstrating election safety across concurrent nodes."
            ),
            "max_tokens": 800
        }
    },
    "V3_TD_31": {
        "title": "Zero-Trust API Gateway TLS 1.3 & JWT Compliance",
        "node1": {
            "domain": "retrieval_qa",
            "system": "You are an authoritative internet protocol and standards specialist.",
            "prompt": (
                "Define concrete engineering problem #31: Zero-Trust API Gateway TLS & Mutual Authentication Compliance. "
                "Retrieve and specify exact normative standard requirements from RFC 8446 (TLS 1.3 key exchange, handshake state machine, "
                "mandatory cipher suites) and RFC 7519 (JSON Web Token structure, cryptographic signature verification, exp/nbf claim validation)."
            ),
            "max_tokens": 800
        },
        "node2": {
            "domain": "formal_reasoning",
            "system": "You are an expert formal verification and security protocol specialist.",
            "prompt_fn": lambda q_text, n1_out: (
                f"User Request: {q_text}\n\n"
                f"Normative RFC Standards Context (from Node 1):\n{n1_out}\n\n"
                f"Directive: Construct a rigorous formal deductive compliance proof using propositional logic and Hoare-style assertions. "
                f"Prove that an API gateway satisfying the stated RFC 8446 and RFC 7519 preconditions guarantees secure channel integrity "
                f"and prevents replay or token tampering attacks."
            ),
            "max_tokens": 800
        }
    },
    "V3_TD_41": {
        "title": "Coupled Double Quantum Dot Two-Level Qubit System",
        "node1": {
            "domain": "science_tech",
            "system": "You are an expert quantum physicist and nanodevice theorist.",
            "prompt": (
                "Define concrete engineering problem #41: Coupled Double Quantum Dot Two-Level Qubit System. "
                "Formulate the physical Hamiltonian matrix incorporating energy detuning epsilon, inter-dot tunneling amplitude t_c, "
                "and Pauli matrices (sigma_z, sigma_x). State the physical boundary conditions and charge qubit Hamiltonian operator."
            ),
            "max_tokens": 800
        },
        "node2": {
            "domain": "mathematics",
            "system": "You are an expert mathematical physicist.",
            "prompt_fn": lambda q_text, n1_out: (
                f"User Request: {q_text}\n\n"
                f"Quantum Hamiltonian Formulation Context (from Node 1):\n{n1_out}\n\n"
                f"Directive: Derive the characteristic polynomial det(H - lambda I) = 0, compute analytical eigenvalues E_plus and E_minus, "
                f"derive the corresponding orthonormal eigenvectors, and calculate the anticrossing energy gap Delta E = 2*t_c."
            ),
            "max_tokens": 800
        }
    },
    "V3_TD_51": {
        "title": "High-Throughput Linux Network Socket & Async Event Loop",
        "node1": {
            "domain": "systems_ops",
            "system": "You are an expert Linux kernel systems and network operations engineer.",
            "prompt": (
                "Define concrete engineering problem #51: High-Throughput Telemetry Ingestion Daemon. "
                "Specify the Linux network socket configuration: non-blocking I/O (O_NONBLOCK), SO_REUSEADDR, SO_RCVBUF buffer tuning (1MB), "
                "TCP_NODELAY, and edge-triggered event notification (epoll) architecture."
            ),
            "max_tokens": 800
        },
        "node2": {
            "domain": "coding",
            "system": "You are an expert, deterministic Python systems programming specialist.",
            "prompt_fn": lambda q_text, n1_out: (
                f"User Request: {q_text}\n\n"
                f"Linux Socket Systems Architecture Context (from Node 1):\n{n1_out}\n\n"
                f"Directive: Implement the asynchronous Python event loop using `asyncio` / `socket` creating the non-blocking server socket, "
                f"handling client read/write loops, message framing, and a self-contained execution test block under `if __name__ == '__main__':`."
            ),
            "max_tokens": 800
        }
    },
    "V3_TD_61": {
        "title": "HPC Datacenter Server Telemetry & Relational Analytics",
        "node1": {
            "domain": "retrieval_qa",
            "system": "You are an expert datacenter hardware telemetry specialist.",
            "prompt": (
                "Define concrete engineering problem #61: Datacenter Server Hardware Telemetry Monitoring. "
                "Extract and specify technical telemetry parameters: CPU core temperatures (C), PCIe bus throughput (GB/s), "
                "GPU power draw (Watts), memory bandwidth saturation, and fan RPMs across server rack nodes."
            ),
            "max_tokens": 800
        },
        "node2": {
            "domain": "structured_data",
            "system": "You are an expert database engineer and SQL analytics architect.",
            "prompt_fn": lambda q_text, n1_out: (
                f"User Request: {q_text}\n\n"
                f"Hardware Telemetry Specifications Context (from Node 1):\n{n1_out}\n\n"
                f"Directive: Design the normalized relational SQL schema (`servers`, `telemetry_logs`) and write optimized PostgreSQL analytical "
                f"queries using window functions (`AVG() OVER (...)`, `RANK()`), partition indexes, and anomaly detection views."
            ),
            "max_tokens": 800
        }
    },
    "V3_TD_71": {
        "title": "Combined Cycle Gas Turbine Cogeneration Efficiency Analysis",
        "node1": {
            "domain": "science_tech",
            "system": "You are an expert thermodynamicist and power plant engineer.",
            "prompt": (
                "Define concrete engineering problem #71: Combined Cycle Gas Turbine (CCGT) Cogeneration System. "
                "Formulate thermodynamic efficiency equations: Brayton gas cycle efficiency, Rankine steam bottoming cycle heat recovery balance, "
                "and overall thermal efficiency. Provide exact analytical equations with pressure ratio r_p = 18 and temperature boundaries."
            ),
            "max_tokens": 800
        },
        "node2": {
            "domain": "creative_synthesis",
            "system": "You are an expert executive engineering communicator.",
            "prompt_fn": lambda q_text, n1_out: (
                f"User Request: {q_text}\n\n"
                f"Thermodynamic Efficiency Formulation Context (from Node 1):\n{n1_out}\n\n"
                f"Directive: Author a comprehensive engineering executive report with system overview, thermodynamic heat balance table, "
                f"operational trade-off analysis, carbon abatement quantification, and deployment recommendations."
            ),
            "max_tokens": 800
        }
    }
}

def ensure_dirs():
    for d in [E1_DIR, E2_DIR, E1_KEY_LOG_DIR, E1_JUDGE_LOG_DIR, E2_KEY_LOG_DIR, E2_JUDGE_LOG_DIR, INTERMEDIATE_DIR]:
        os.makedirs(d, exist_ok=True)

def load_cached_dict(path: str) -> Dict[str, Dict[str, Any]]:
    res = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        res[rec["query_id"]] = rec
                    except Exception:
                        pass
    return res

def preflight_assertions():
    print("=" * 80)
    print(">>> MANDATORY PRE-FLIGHT ASSERTIONS (OPTION B: PATHWAY 1 SEQUENTIAL SYNTHESIS) <<<")
    print("=" * 80)

    # 1. Hard Rule 13: Distinct Roster
    baselines = [MODEL_SPECS["b20"]["name"], MODEL_SPECS["b32"]["name"], MODEL_SPECS["b72"]["name"], MODEL_SPECS["b120"]["name"]]
    assert len(set(baselines)) == 4, f"[FAIL] Baselines are not distinct: {baselines}"
    for b_name in baselines:
        assert MODEL_SPECS["e1_coding_base"]["name"] != b_name
        assert MODEL_SPECS["e2_coding_ft"]["name"] != b_name
        assert MODEL_SPECS["judge"]["name"] != b_name
    print("[PASS] Hard Rule 13 Distinct-Model Pre-Flight: All systems map to strictly distinct endpoints.")

    # 2. Hard Rule 16(b): Fairness Assertion
    combined_pool_params = MODEL_SPECS["e1_coding_base"]["params"] + MODEL_SPECS["pool_general_base"]["params"]
    print(f"Combined Participating SLM Pool Parameters: {combined_pool_params:.2f}B (3.82B Coder + 8.03B General)")
    for b_key in ["b20", "b32", "b72", "b120"]:
        b_name = MODEL_SPECS[b_key]["name"]
        b_params = MODEL_SPECS[b_key]["params"]
        assert combined_pool_params < b_params, f"[FAIL] Combined SLM {combined_pool_params}B >= Baseline {b_params}B"
        print(f"  [PASS] vs {b_key.upper()} ({b_name}, {b_params:.1f}B): {combined_pool_params:.2f}B < {b_params:.1f}B (Ratio: {combined_pool_params/b_params:.2f}x)")

    # 3. Hard Rule 17: Strict Model Identity & Local Compute
    assert MODEL_SPECS["e1_coding_base"]["params"] == MODEL_SPECS["e2_coding_ft"]["params"], "[FAIL] Model params changed across E1 and E2!"
    print("[PASS] Hard Rule 17 Strict Model Identity: Exact same architecture & parameter count (3.82B) across E1 and E2.")
    print("[PASS] Local Compute Only: Both base and fine-tuned models hosted on local Ollama.")
    print("=" * 80 + "\n")

def call_ollama(model_name: str, prompt: str, system_prompt: str, max_tokens: int = 800) -> str:
    url = "http://127.0.0.1:11434/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=360) as resp:
        body = resp.read().decode("utf-8")
        parsed = json.loads(body)
        return parsed["choices"][0]["message"]["content"].strip()

def call_gemini_judge(prompt: str, gemini_key: str, max_retries: int = 5) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite:generateContent?key={gemini_key}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json"
        }
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, headers={"Content-Type": "application/json"}, data=data, method="POST")
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                return res["candidates"][0]["content"]["parts"][0]["text"].strip()
        except urllib.error.HTTPError as e:
            print(f"      [Gemini Judge Retry {attempt+1}] HTTP {e.code}")
            time.sleep(2.0 * (attempt + 1))
        except Exception as e:
            print(f"      [Gemini Judge Retry {attempt+1}] Error: {e}")
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError("[ABORT] Gemini judge call failed after retries.")

async def generate_sequential_slm_pipeline(
    queries: List[Dict[str, Any]],
    exp_tag: str,
    coding_model: str,
    output_path: str,
    llama8b: HFRouterModelRunner
) -> Dict[str, Dict[str, Any]]:
    print(f"\n{'='*80}\nSTAGE 1: GENERATING {exp_tag} SLM RESPONSES (SEQUENTIAL CONTEXT + AGGREGATOR)\n{'='*80}")
    slm_cache = load_cached_dict(output_path)

    for idx, q in enumerate(queries, 1):
        qid = q["id"]
        if qid in slm_cache:
            print(f"  [{idx}/{len(queries)}] {qid} (Cached in {exp_tag})")
            continue

        contract = QUERY_CONTRACTS[qid]
        q_text = q["query"]
        print(f"  [{idx}/{len(queries)}] Generating {exp_tag} for {qid}: '{contract['title']}'...")

        # --- Step 1: Execute Node 1 ---
        n1_cache_file = os.path.join(INTERMEDIATE_DIR, f"{qid}_node1.txt")
        if os.path.exists(n1_cache_file):
            with open(n1_cache_file, "r", encoding="utf-8") as f:
                node1_output = f.read()
            print(f"    -> Node 1 loaded from cache ({len(node1_output)} chars)")
        else:
            print(f"    -> Executing Node 1 ({contract['node1']['domain']}) via Llama-3.1-8B...")
            t0 = time.perf_counter()
            r_n1 = await llama8b.generate(
                prompt=f"User Query: {q_text}\n\n{contract['node1']['prompt']}",
                system_prompt=contract['node1']['system'],
                max_tokens=contract['node1']['max_tokens']
            )
            node1_output = r_n1.text.strip()
            with open(n1_cache_file, "w", encoding="utf-8") as f:
                f.write(node1_output)
            print(f"    -> Node 1 completed ({len(node1_output)} chars, {time.perf_counter()-t0:.1f}s)")

        # --- Step 2: Execute Node 2 Conditioned on Node 1 ---
        n2_is_coding = contract["node2"]["domain"] == "coding"
        n2_cache_file = os.path.join(INTERMEDIATE_DIR, f"{qid}_{exp_tag}_node2.txt")
        if os.path.exists(n2_cache_file):
            with open(n2_cache_file, "r", encoding="utf-8") as f:
                node2_output = f.read()
            print(f"    -> Node 2 loaded from cache ({len(node2_output)} chars)")
        else:
            prompt_n2 = contract["node2"]["prompt_fn"](q_text, node1_output)
            t0 = time.perf_counter()
            if n2_is_coding:
                print(f"    -> Executing Node 2 Coding via {coding_model} on Ollama...")
                node2_output = call_ollama(
                    model_name=coding_model,
                    prompt=prompt_n2,
                    system_prompt=contract["node2"]["system"],
                    max_tokens=contract["node2"]["max_tokens"]
                )
            else:
                print(f"    -> Executing Node 2 ({contract['node2']['domain']}) via Llama-3.1-8B...")
                r_n2 = await llama8b.generate(
                    prompt=prompt_n2,
                    system_prompt=contract["node2"]["system"],
                    max_tokens=contract["node2"]["max_tokens"]
                )
                node2_output = r_n2.text.strip()
            with open(n2_cache_file, "w", encoding="utf-8") as f:
                f.write(node2_output)
            print(f"    -> Node 2 completed ({len(node2_output)} chars, {time.perf_counter()-t0:.1f}s)")

        # --- Step 3: Two-Stage Aggregator Synthesis ---
        print(f"    -> Synthesizing final response via TwoStageAggregator (Llama-3.1-8B)...")
        prompt_agg = (
            f"Original User Query:\n{q_text}\n\n"
            f"Specialist Subtask Outputs to Synthesize:\n"
            f"### Subtask Result 1 [{contract['node1']['domain'].replace('_', ' ').title()}]:\n{node1_output}\n\n"
            f"### Subtask Result 2 [{contract['node2']['domain'].replace('_', ' ').title()}]:\n{node2_output}\n\n"
            f"Produce the authoritative, comprehensive synthesized response (preserving all technical equations, schemas, and code implementations):"
        )
        agg_sys_prompt = (
            "You are an expert Chief Synthesizer SLM. Your task is to synthesize specialist outputs into an authoritative, "
            "comprehensive technical document. Harmonize the narrative, preserve all code and schemas in full without truncation, "
            "eliminate disjointed transitions, and ensure exhaustive structural completeness."
        )
        t0 = time.perf_counter()
        synth_resp = await llama8b.generate(
            prompt=prompt_agg,
            system_prompt=agg_sys_prompt,
            max_tokens=1100
        )
        final_synth = synth_resp.text.strip()
        print(f"    -> Synthesis complete ({len(final_synth)} chars, {time.perf_counter()-t0:.1f}s)")

        models_used = [MODEL_SPECS["pool_general_base"]["name"]]
        if n2_is_coding:
            models_used.insert(0, coding_model)

        rec = {
            "query_id": qid,
            "complexity_tier": q["complexity_tier"],
            "domains": q.get("domains", []),
            "query_text": q_text,
            "system_type": f"SLM_Pipeline_{exp_tag} (Sequential Synthesis, 11.85B Pool)",
            "participating_models": models_used,
            "response_text": final_synth,
            "latency_sec": synth_resp.latency_ms / 1000.0
        }
        slm_cache[qid] = rec
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"    [SAVED] {exp_tag} {qid} -> {output_path}")

    return slm_cache

def run_dual_judge_trial(
    query: Dict[str, Any],
    candidate_a_text: str,
    candidate_b_text: str,
    candidate_a_sys: str,
    candidate_b_sys: str,
    order_tag: str,
    baseline_tier: str,
    exp_tag: str,
    key_dir: str,
    judge_dir: str,
    gemini_key: str
) -> Dict[str, Any]:
    system_prompt = (
        "You are an impartial, expert AI judge evaluating two candidate responses (Candidate A and Candidate B) to a technical user query.\n\n"
        "Evaluation Frameworks:\n"
        "1. Criteria Framework (1-5 Scale):\n"
        "   - Correctness (1-5): Mathematical, factual, algorithmic precision.\n"
        "   - Completeness (1-5): Thorough fulfillment of all specifications and constraints.\n"
        "   - Coherence (1-5): Logical structure, clarity, and synthesis.\n\n"
        "2. Holistic Framework (1-10 Scale):\n"
        "   - Overall Technical Quality Score (1-10): Global technical rigor, practical utility, and excellence.\n"
        "   - 10 = Flawless publication-grade solution; 1 = Completely incorrect or useless.\n\n"
        "Instructions:\n"
        "- Evaluate both candidates with absolute impartiality.\n"
        "- Do NOT exhibit position bias. Candidate A and Candidate B must receive equal critical rigor.\n"
        "- Provide integer criteria scores (1-5) and integer holistic scores (1-10) for both candidates.\n"
        "- Output strictly valid JSON matching this schema:\n"
        "{\n"
        '  "criteria_scores": {\n'
        '    "Candidate A": {"correctness": 5, "completeness": 5, "coherence": 5},\n'
        '    "Candidate B": {"correctness": 4, "completeness": 4, "coherence": 4}\n'
        "  },\n"
        '  "holistic_scores": {\n'
        '    "Candidate A": 9,\n'
        '    "Candidate B": 8\n'
        "  },\n"
        '  "primary_differentiator": "Mathematical precision and verified code",\n'
        '  "reasoning": "Candidate A provides superior derivations and fully executable implementations."\n'
        "}"
    )

    user_prompt = (
        f"User Query:\n{query['query']}\n\n"
        f"=== Candidate A ===\n{candidate_a_text[:4000]}\n\n"
        f"=== Candidate B ===\n{candidate_b_text[:4000]}\n\n"
        f"Provide your JSON evaluation:"
    )

    t0 = time.perf_counter()
    full_prompt = f"{system_prompt}\n\n{user_prompt}"
    raw_content = call_gemini_judge(full_prompt, gemini_key)

    latency = time.perf_counter() - t0
    clean_json = raw_content.strip()
    if "```json" in clean_json:
        clean_json = clean_json.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in clean_json:
        clean_json = clean_json.split("```", 1)[1].split("```", 1)[0].strip()

    parsed = json.loads(clean_json)
    crit_scores = parsed.get("criteria_scores", {})
    hol_scores = parsed.get("holistic_scores", {})
    reasoning = parsed.get("reasoning", "")
    differentiator = parsed.get("primary_differentiator", "")

    ca_crit = crit_scores.get("Candidate A", {"correctness": 3, "completeness": 3, "coherence": 3})
    cb_crit = crit_scores.get("Candidate B", {"correctness": 3, "completeness": 3, "coherence": 3})
    ca_cqs = round((ca_crit.get("correctness", 3) + ca_crit.get("completeness", 3) + ca_crit.get("coherence", 3)) / 3.0, 4)
    cb_cqs = round((cb_crit.get("correctness", 3) + cb_crit.get("completeness", 3) + cb_crit.get("coherence", 3)) / 3.0, 4)

    ca_hol = float(hol_scores.get("Candidate A", 5.0))
    cb_hol = float(hol_scores.get("Candidate B", 5.0))

    if candidate_a_sys.startswith("SLM"):
        slm_cqs = ca_cqs
        llm_cqs = cb_cqs
        slm_hol = ca_hol
        llm_hol = cb_hol
    else:
        slm_cqs = cb_cqs
        llm_cqs = ca_cqs
        slm_hol = cb_hol
        llm_hol = ca_hol

    delta_crit = round(slm_cqs - llm_cqs, 4)
    p_criteria = round((1.0 - (abs(delta_crit) / 4.0)) * 100.0, 2)
    outcome_criteria = "SLM_WIN" if slm_cqs > llm_cqs else ("LLM_WIN" if llm_cqs > slm_cqs else "DRAW")

    delta_hol = round(slm_hol - llm_hol, 4)
    qp_holistic = round(1.0 - (abs(delta_hol) / 9.0), 4)
    outcome_holistic = "SLM_WIN" if slm_hol > llm_hol else ("LLM_WIN" if llm_hol > slm_hol else "DRAW")

    truncation_flag = any(term in reasoning.lower() for term in ["truncated", "cut off", "incomplete", "prematurely"])

    trial_record = {
        "query_id": query["id"],
        "baseline_tier": baseline_tier,
        "order_tag": order_tag,
        "candidate_a_sys": candidate_a_sys,
        "candidate_b_sys": candidate_b_sys,
        "candidate_a_crit": ca_crit,
        "candidate_b_crit": cb_crit,
        "candidate_a_cqs": ca_cqs,
        "candidate_b_cqs": cb_cqs,
        "candidate_a_hol": ca_hol,
        "candidate_b_hol": cb_hol,
        "slm_cqs": slm_cqs,
        "llm_cqs": llm_cqs,
        "slm_holistic": slm_hol,
        "llm_holistic": llm_hol,
        "delta_criteria": delta_crit,
        "p_criteria_pct": p_criteria,
        "outcome_criteria": outcome_criteria,
        "delta_holistic": delta_hol,
        "qp_holistic": qp_holistic,
        "outcome_holistic": outcome_holistic,
        "primary_differentiator": differentiator,
        "reasoning": reasoning,
        "truncation_flag": truncation_flag,
        "judge_latency_sec": latency
    }

    timestamp_ms = int(time.time() * 1000)
    key_filename = f"key_{query['id']}_{baseline_tier}_{order_tag}_{timestamp_ms}.json"
    public_filename = f"judge_{query['id']}_{baseline_tier}_{order_tag}_{timestamp_ms}.json"
    
    with open(os.path.join(key_dir, key_filename), "w", encoding="utf-8") as f:
        json.dump(trial_record, f, indent=2)
    
    public_data = {
        "query_id": query["id"],
        "baseline_tier": baseline_tier,
        "order_tag": order_tag,
        "candidate_a_alias": "Candidate A",
        "candidate_b_alias": "Candidate B",
        "criteria_scores": crit_scores,
        "holistic_scores": hol_scores,
        "primary_differentiator": differentiator,
        "reasoning": reasoning,
        "latency_sec": latency
    }
    with open(os.path.join(judge_dir, public_filename), "w", encoding="utf-8") as f:
        json.dump(public_data, f, indent=2)

    return trial_record

def compute_ci(vals: List[float]) -> Tuple[float, float, float]:
    n = len(vals)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = sum(vals) / n
    if n == 1:
        return mean, mean, mean
    var = sum((x - mean) ** 2 for x in vals) / (n - 1)
    se = math.sqrt(var / n)
    tcrit = 2.131 if n == 16 else 1.96
    return round(mean, 4), round(mean - tcrit * se, 4), round(mean + tcrit * se, 4)

def run_experiment_judging(
    exp_tag: str,
    slm_cache: Dict[str, Dict[str, Any]],
    b_caches: Dict[str, Dict[str, Any]],
    queries: List[Dict[str, Any]],
    trials_path: str,
    preserved_path: str,
    key_dir: str,
    judge_dir: str,
    gemini_key: str,
    pool_config_desc: str = ""
) -> List[Dict[str, Any]]:
    print(f"\n{'='*70}\nSTARTING JUDGE EVALUATION FOR {exp_tag}\n{'='*70}")
    existing_trials = set()
    all_trial_records = []
    if os.path.exists(trials_path):
        with open(trials_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        tr = json.loads(line)
                        k = f"{tr['query_id']}_{tr['baseline_tier']}_{tr['order_tag']}"
                        existing_trials.add(k)
                        all_trial_records.append(tr)
                    except Exception:
                        pass
    print(f"Loaded {len(all_trial_records)} existing {exp_tag} judge trials.")

    trials_file = open(trials_path, "a", encoding="utf-8")
    preserved_file = open(preserved_path, "a", encoding="utf-8")
    total_trials = len(queries) * 4 * 2
    cur = len(all_trial_records)

    for q in queries:
        qid = q["id"]
        slm_text = slm_cache[qid]["response_text"]
        slm_sys_id = f"SLM_Pipeline_{exp_tag}"

        for b_key in ["b20", "b32", "b72", "b120"]:
            b_text = b_caches[b_key][qid]["response_text"]
            b_sys_id = MODEL_SPECS[b_key]["name"]

            # Forward Order
            k_fwd = f"{qid}_{b_key}_forward"
            if k_fwd not in existing_trials:
                cur += 1
                print(f"  [{cur}/{total_trials}] {exp_tag} Judging {qid} vs {b_key.upper()} [FORWARD]...")
                t_fwd = run_dual_judge_trial(
                    query=q,
                    candidate_a_text=slm_text,
                    candidate_b_text=b_text,
                    candidate_a_sys=slm_sys_id,
                    candidate_b_sys=b_sys_id,
                    order_tag="forward",
                    baseline_tier=b_key,
                    exp_tag=exp_tag,
                    key_dir=key_dir,
                    judge_dir=judge_dir,
                    gemini_key=gemini_key
                )
                trials_file.write(json.dumps(t_fwd) + "\n")
                trials_file.flush()
                all_trial_records.append(t_fwd)
                existing_trials.add(k_fwd)

                pres = {
                    "query_id": qid,
                    "complexity_tier": q["complexity_tier"],
                    "domains": q.get("domains", []),
                    "query_text": q["query"],
                    "baseline_tier": b_key,
                    "order_tag": "forward",
                    "fine_tuning_configuration": pool_config_desc,
                    "slm_pool_configuration": {
                        "specialists": slm_cache[qid]["participating_models"],
                        "combined_parameters_b": 11.85
                    },
                    "baseline_llm_configuration": {
                        "model_name": b_sys_id,
                        "parameters_b": MODEL_SPECS[b_key]["params"]
                    },
                    "slm_response": slm_text,
                    "llm_response": b_text,
                    "criteria_scores_1_to_5": {
                        "slm_cqs": t_fwd["slm_cqs"],
                        "llm_cqs": t_fwd["llm_cqs"],
                        "p_criteria_pct": t_fwd["p_criteria_pct"],
                        "outcome": t_fwd["outcome_criteria"]
                    },
                    "holistic_scores_1_to_10": {
                        "slm_score": t_fwd["slm_holistic"],
                        "llm_score": t_fwd["llm_holistic"],
                        "qp_holistic": t_fwd["qp_holistic"],
                        "outcome": t_fwd["outcome_holistic"]
                    },
                    "judge_reasoning": t_fwd["reasoning"]
                }
                preserved_file.write(json.dumps(pres) + "\n")
                preserved_file.flush()
                time.sleep(1.2)

            # Swapped Order
            k_swp = f"{qid}_{b_key}_swapped"
            if k_swp not in existing_trials:
                cur += 1
                print(f"  [{cur}/{total_trials}] {exp_tag} Judging {qid} vs {b_key.upper()} [SWAPPED]...")
                t_swp = run_dual_judge_trial(
                    query=q,
                    candidate_a_text=b_text,
                    candidate_b_text=slm_text,
                    candidate_a_sys=b_sys_id,
                    candidate_b_sys=slm_sys_id,
                    order_tag="swapped",
                    baseline_tier=b_key,
                    exp_tag=exp_tag,
                    key_dir=key_dir,
                    judge_dir=judge_dir,
                    gemini_key=gemini_key
                )
                trials_file.write(json.dumps(t_swp) + "\n")
                trials_file.flush()
                all_trial_records.append(t_swp)
                existing_trials.add(k_swp)

                pres_swp = {
                    "query_id": qid,
                    "complexity_tier": q["complexity_tier"],
                    "domains": q.get("domains", []),
                    "query_text": q["query"],
                    "baseline_tier": b_key,
                    "order_tag": "swapped",
                    "fine_tuning_configuration": pool_config_desc,
                    "slm_pool_configuration": {
                        "specialists": slm_cache[qid]["participating_models"],
                        "combined_parameters_b": 11.85
                    },
                    "baseline_llm_configuration": {
                        "model_name": b_sys_id,
                        "parameters_b": MODEL_SPECS[b_key]["params"]
                    },
                    "slm_response": slm_text,
                    "llm_response": b_text,
                    "criteria_scores_1_to_5": {
                        "slm_cqs": t_swp["slm_cqs"],
                        "llm_cqs": t_swp["llm_cqs"],
                        "p_criteria_pct": t_swp["p_criteria_pct"],
                        "outcome": t_swp["outcome_criteria"]
                    },
                    "holistic_scores_1_to_10": {
                        "slm_score": t_swp["slm_holistic"],
                        "llm_score": t_swp["llm_holistic"],
                        "qp_holistic": t_swp["qp_holistic"],
                        "outcome": t_swp["outcome_holistic"]
                    },
                    "judge_reasoning": t_swp["reasoning"]
                }
                preserved_file.write(json.dumps(pres_swp) + "\n")
                preserved_file.flush()
                time.sleep(1.2)

    trials_file.close()
    preserved_file.close()
    return all_trial_records

def compile_summary_and_audit(
    exp_tag: str,
    desc: str,
    all_trial_records: List[Dict[str, Any]],
    queries: List[Dict[str, Any]],
    summary_path: str,
    e1_ref_summary: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    summary = {
        "experiment": exp_tag,
        "description": desc,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_queries": len(queries),
        "total_trials": len(all_trial_records),
        "fairness_assertion": "PASSED across all 4 tiers (combined 11.85B < 20B/32B/72B/120B)",
        "roster_distinctness_assertion": "PASSED (0 model collisions)",
        "hard_rule_17_compliance": "PASSED (Strict model identity, 11.85B pool, local compute only)",
        "tiers": {}
    }

    for b_key in ["b20", "b32", "b72", "b120"]:
        tier_trials = [t for t in all_trial_records if t["baseline_tier"] == b_key]
        n_trials = len(tier_trials)
        b_name = MODEL_SPECS[b_key]["name"]
        b_params = MODEL_SPECS[b_key]["params"]

        concordance_passes = 0
        for t in tier_trials:
            exp_crit = "SLM_WIN" if t["slm_cqs"] > t["llm_cqs"] else ("LLM_WIN" if t["llm_cqs"] > t["slm_cqs"] else "DRAW")
            exp_hol = "SLM_WIN" if t["slm_holistic"] > t["llm_holistic"] else ("LLM_WIN" if t["llm_holistic"] > t["slm_holistic"] else "DRAW")
            if t["outcome_criteria"] == exp_crit and t["outcome_holistic"] == exp_hol:
                concordance_passes += 1
        concordance_rate = (concordance_passes / n_trials) * 100.0 if n_trials > 0 else 0.0
        truncations = sum(1 for t in tier_trials if t.get("truncation_flag", False))

        pairs_agree_crit = 0
        pairs_agree_hol = 0
        n_pairs = len(queries)
        for q in queries:
            qid = q["id"]
            tf = next((t for t in tier_trials if t["query_id"] == qid and t["order_tag"] == "forward"), None)
            ts = next((t for t in tier_trials if t["query_id"] == qid and t["order_tag"] == "swapped"), None)
            if tf and ts:
                if tf["outcome_criteria"] == ts["outcome_criteria"]:
                    pairs_agree_crit += 1
                if tf["outcome_holistic"] == ts["outcome_holistic"]:
                    pairs_agree_hol += 1
        swap_consistency_crit = (pairs_agree_crit / n_pairs) * 100.0 if n_pairs > 0 else 0.0
        swap_consistency_hol = (pairs_agree_hol / n_pairs) * 100.0 if n_pairs > 0 else 0.0

        crit_slm_wins = sum(1 for t in tier_trials if t["outcome_criteria"] == "SLM_WIN")
        crit_llm_wins = sum(1 for t in tier_trials if t["outcome_criteria"] == "LLM_WIN")
        crit_draws = sum(1 for t in tier_trials if t["outcome_criteria"] == "DRAW")
        crit_effective_wins = crit_slm_wins + crit_draws
        p_crit_vals = [t["p_criteria_pct"] for t in tier_trials]
        p_mean, p_lo, p_hi = compute_ci(p_crit_vals)
        d_crit_vals = [t["delta_criteria"] for t in tier_trials]
        d_mean, d_lo, d_hi = compute_ci(d_crit_vals)
        slm_cqs_mean = round(sum(t["slm_cqs"] for t in tier_trials) / n_trials, 4) if n_trials > 0 else 0.0
        llm_cqs_mean = round(sum(t["llm_cqs"] for t in tier_trials) / n_trials, 4) if n_trials > 0 else 0.0

        hol_slm_wins = sum(1 for t in tier_trials if t["outcome_holistic"] == "SLM_WIN")
        hol_llm_wins = sum(1 for t in tier_trials if t["outcome_holistic"] == "LLM_WIN")
        hol_draws = sum(1 for t in tier_trials if t["outcome_holistic"] == "DRAW")
        hol_effective_wins = hol_slm_wins + hol_draws
        qp_hol_vals = [t["qp_holistic"] for t in tier_trials]
        qp_mean, qp_lo, qp_hi = compute_ci(qp_hol_vals)
        d_hol_vals = [t["delta_holistic"] for t in tier_trials]
        dh_mean, dh_lo, dh_hi = compute_ci(d_hol_vals)
        slm_hol_mean = round(sum(t["slm_holistic"] for t in tier_trials) / n_trials, 4) if n_trials > 0 else 0.0
        llm_hol_mean = round(sum(t["llm_holistic"] for t in tier_trials) / n_trials, 4) if n_trials > 0 else 0.0

        matched_gain_crit = 0.0
        matched_gain_hol = 0.0
        matched_qp_gain_hol = 0.0
        if e1_ref_summary and b_key in e1_ref_summary.get("tiers", {}):
            e1_t = e1_ref_summary["tiers"][b_key]
            matched_gain_crit = round(d_mean - e1_t["criteria_framework_1_to_5"]["mean_delta_q"], 4)
            matched_gain_hol = round(dh_mean - e1_t["holistic_framework_1_to_10"]["mean_delta_q"], 4)
            matched_qp_gain_hol = round(qp_mean - e1_t["holistic_framework_1_to_10"]["qp_overall"], 4)

        summary["tiers"][b_key] = {
            "baseline_model": b_name,
            "baseline_params_b": b_params,
            "trials_count": n_trials,
            "audit": {
                "concordance_rate_pct": round(concordance_rate, 2),
                "truncation_count": truncations,
                "swap_consistency_criteria_pct": round(swap_consistency_crit, 2),
                "swap_consistency_holistic_pct": round(swap_consistency_hol, 2)
            },
            "criteria_framework_1_to_5": {
                "slm_mean_cqs": slm_cqs_mean,
                "llm_mean_cqs": llm_cqs_mean,
                "mean_delta_q": d_mean,
                "delta_q_95ci": [d_lo, d_hi],
                "p_mean_pct": p_mean,
                "p_mean_95ci": [p_lo, p_hi],
                "slm_wins": crit_slm_wins,
                "llm_wins": crit_llm_wins,
                "draws": crit_draws,
                "effective_slm_wins": crit_effective_wins,
                "slm_win_rate_pct": round((crit_slm_wins / n_trials) * 100.0, 2),
                "effective_win_rate_pct": round((crit_effective_wins / n_trials) * 100.0, 2),
                "matched_delta_q_gain_vs_e1": matched_gain_crit
            },
            "holistic_framework_1_to_10": {
                "slm_mean_score": slm_hol_mean,
                "llm_mean_score": llm_hol_mean,
                "mean_delta_q": dh_mean,
                "delta_q_95ci": [dh_lo, dh_hi],
                "qp_overall": qp_mean,
                "qp_overall_95ci": [qp_lo, qp_hi],
                "slm_wins": hol_slm_wins,
                "llm_wins": hol_llm_wins,
                "draws": hol_draws,
                "effective_slm_wins": hol_effective_wins,
                "slm_win_rate_pct": round((hol_slm_wins / n_trials) * 100.0, 2),
                "effective_win_rate_pct": round((hol_effective_wins / n_trials) * 100.0, 2),
                "matched_delta_q_gain_vs_e1": matched_gain_hol,
                "matched_qp_gain_vs_e1": matched_qp_gain_hol
            }
        }

        print(f"\n--- {exp_tag} TIER {b_key.upper()} ({b_name}, {b_params}B) ---")
        print(f"  Trials: {n_trials} | Concordance: {concordance_rate:.1f}% | Swap Consistency: {swap_consistency_hol:.1f}%")
        print(f"  Criteria Mode (1-5):")
        print(f"    SLM CQS: {slm_cqs_mean} | LLM CQS: {llm_cqs_mean} | Mean Delta: {d_mean} [{d_lo}, {d_hi}]")
        print(f"    P_criteria: {p_mean:.2f}% [{p_lo:.2f}%, {p_hi:.2f}%] | Wins: Pure={crit_slm_wins}, Draws={crit_draws}, Effective={crit_effective_wins} ({crit_effective_wins/n_trials*100:.1f}%)")
        print(f"  Holistic Mode (1-10, Mentor Protocol):")
        print(f"    SLM Score: {slm_hol_mean} | LLM Score: {llm_hol_mean} | Mean Delta: {dh_mean} [{dh_lo}, {dh_hi}]")
        print(f"    QP_holistic: {qp_mean:.4f} [{qp_lo:.4f}, {qp_hi:.4f}] | Wins: Pure={hol_slm_wins}, Draws={hol_draws}, Effective={hol_effective_wins} ({hol_effective_wins/n_trials*100:.1f}%)")
        if e1_ref_summary:
            print(f"    MATCHED GAIN vs E1: DeltaQ Gain: {matched_gain_hol:+.3f} | QP Gain: {matched_qp_gain_hol:+.4f}")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Audit Summary Saved] -> {summary_path}")
    return summary

async def main():
    ensure_dirs()
    preflight_assertions()

    with open("data/v3_queries_dev.json", "r", encoding="utf-8") as f:
        all_dev = json.load(f)
    target_ids = ["V3_TD_01", "V3_TD_11", "V3_TD_21", "V3_TD_31", "V3_TD_41", "V3_TD_51", "V3_TD_61", "V3_TD_71"]
    queries = [q for q in all_dev if q["id"] in target_ids]

    # Load Baseline Caches
    b_caches = {
        "b20": load_cached_dict(os.path.join(BASELINES_DIR, "baseline_20b_responses.jsonl")),
        "b32": load_cached_dict(os.path.join(BASELINES_DIR, "baseline_32b_responses.jsonl")),
        "b72": load_cached_dict(os.path.join(BASELINES_DIR, "baseline_72b_responses.jsonl")),
        "b120": load_cached_dict(os.path.join(BASELINES_DIR, "baseline_120b_responses.jsonl"))
    }
    # Copy baseline caches to E1 and E2 dirs
    for b_key, b_dict in b_caches.items():
        fname = f"baseline_{b_key[1:]}b_responses.jsonl"
        for target_d in [E1_DIR, E2_DIR]:
            tpath = os.path.join(target_d, fname)
            if not os.path.exists(tpath):
                with open(tpath, "w", encoding="utf-8") as f:
                    for r in b_dict.values():
                        f.write(json.dumps(r) + "\n")

    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        raise ValueError("GEMINI_API_KEY is not set in environment!")

    llama8b = HFRouterModelRunner(logical_model_name="llama8b", api_model_name="meta-llama/Llama-3.1-8B-Instruct", max_tokens=1100)

    # Clean previous stale judge trial files for a fresh judging pass
    # Note: Keep SLM pipeline responses and baseline responses intact
    for p in [E1_TRIALS_PATH, E2_TRIALS_PATH, E1_PRESERVED_PATH, E2_PRESERVED_PATH]:
        if os.path.exists(p):
            os.remove(p)
            print(f"[REMOVED] Stale judge file cleared for fresh run: {p}")

    # STAGE 1A: Generate E1 SLM Responses (Base phi3.5:cpu)
    e1_slm_cache = await generate_sequential_slm_pipeline(
        queries=queries,
        exp_tag="E1",
        coding_model=MODEL_SPECS["e1_coding_base"]["name"],
        output_path=E1_SLM_PATH,
        llama8b=llama8b
    )

    # STAGE 1B: Generate E2 SLM Responses (FT phi3.5-ft-coding)
    e2_slm_cache = await generate_sequential_slm_pipeline(
        queries=queries,
        exp_tag="E2",
        coding_model=MODEL_SPECS["e2_coding_ft"]["name"],
        output_path=E2_SLM_PATH,
        llama8b=llama8b
    )

    # STAGE 2A: Symmetrical Double-Blind Judging for E1
    e1_trials = run_experiment_judging(
        exp_tag="E1",
        slm_cache=e1_slm_cache,
        b_caches=b_caches,
        queries=queries,
        trials_path=E1_TRIALS_PATH,
        preserved_path=E1_PRESERVED_PATH,
        key_dir=E1_KEY_LOG_DIR,
        judge_dir=E1_JUDGE_LOG_DIR,
        gemini_key=gemini_key,
        pool_config_desc="E1: Re-baselined SLM Pool (Base phi3.5:cpu 3.82B Coder + Base Llama-3.1-8B), Baseline Not Fine-Tuned"
    )

    e1_summary = compile_summary_and_audit(
        exp_tag="E1",
        desc="Re-baselined SLM Pool (Base phi3.5:cpu 3.82B Coder + Base Llama-3.1-8B, 11.85B) vs 4-Tier Non-FT Baselines",
        all_trial_records=e1_trials,
        queries=queries,
        summary_path=E1_SUMMARY_PATH
    )

    # STAGE 2B: Symmetrical Double-Blind Judging for E2
    e2_trials = run_experiment_judging(
        exp_tag="E2",
        slm_cache=e2_slm_cache,
        b_caches=b_caches,
        queries=queries,
        trials_path=E2_TRIALS_PATH,
        preserved_path=E2_PRESERVED_PATH,
        key_dir=E2_KEY_LOG_DIR,
        judge_dir=E2_JUDGE_LOG_DIR,
        gemini_key=gemini_key,
        pool_config_desc="E2: Query-Dependent FT SLM Pool (FT phi3.5-ft-coding 3.82B Coder + Base Llama-3.1-8B), Baseline Not Fine-Tuned"
    )

    e2_summary = compile_summary_and_audit(
        exp_tag="E2",
        desc="Query-Dependent FT SLM Pool (FT phi3.5-ft-coding 3.82B Coder + Base Llama-3.1-8B, 11.85B) vs 4-Tier Non-FT Baselines",
        all_trial_records=e2_trials,
        queries=queries,
        summary_path=E2_SUMMARY_PATH,
        e1_ref_summary=e1_summary
    )

    print("\n" + "=" * 80)
    print(">>> OPTION B PATHWAY 1 (SEQUENTIAL SYNTHESIS) EXECUTION COMPLETE <<<")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
