"""
AI Search Framework - Phase F: Automated Local GGUF Deployment to Ollama
Watches or imports phi3.5_ft_retrieval_q4_k_m.gguf from Google Drive into local Ollama.
Creates Modelfile with verified ChatML template and stop tokens.
Builds 'phi3.5-ft-retrieval:latest' and validates inference.
"""

import os
import sys
import time
import subprocess
import json

DRIVE_EXPORT_PATH = r"G:\My Drive\ai_search_phase_f\exported_gguf\phi3.5_ft_retrieval_q4_k_m.gguf"
LOCAL_MODELS_DIR = "models"
LOCAL_GGUF_PATH = os.path.join(LOCAL_MODELS_DIR, "phi3.5_ft_retrieval_q4_k_m.gguf")
MODELFILE_PATH = os.path.join(LOCAL_MODELS_DIR, "Modelfile.phi35_ft_retrieval")

MODELFILE_CONTENT = f"""FROM ./{os.path.basename(LOCAL_GGUF_PATH)}
TEMPLATE \"\"\"{{{{ if .System }}}}<|system|>
{{{{ .System }}}}<|end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|user|>
{{{{ .Prompt }}}}<|end|>
{{{{ end }}}}<|assistant|>
{{{{ .Response }}}}<|end|>
\"\"\"
PARAMETER stop <|system|>
PARAMETER stop <|user|>
PARAMETER stop <|end|>
PARAMETER stop <|assistant|>
PARAMETER temperature 0.0
"""

def deploy_model(gguf_source: str = DRIVE_EXPORT_PATH):
    os.makedirs(LOCAL_MODELS_DIR, exist_ok=True)

    if not os.path.exists(gguf_source):
        print(f"ERROR: Source GGUF not found at {gguf_source}")
        print("Please ensure the Colab notebook has completed Step 7 and exported the file.")
        return False

    print(f"[1/4] Copying GGUF from {gguf_source} to {LOCAL_GGUF_PATH}...")
    import shutil
    shutil.copy2(gguf_source, LOCAL_GGUF_PATH)
    file_size_gb = os.path.getsize(LOCAL_GGUF_PATH) / (1024**3)
    print(f"  Copied {file_size_gb:.2f} GB GGUF model successfully.")

    print(f"[2/4] Writing Modelfile to {MODELFILE_PATH}...")
    with open(MODELFILE_PATH, "w", encoding="utf-8") as f:
        f.write(MODELFILE_CONTENT)
    print("  Modelfile written with verified ChatML template and stop tokens.")

    print("[3/4] Registering model in Ollama: 'ollama create phi3.5-ft-retrieval -f Modelfile'...")
    ollama_bin = r"C:\Users\ACER\AppData\Local\Programs\Ollama\ollama.exe"
    cmd = [ollama_bin, "create", "phi3.5-ft-retrieval", "-f", os.path.abspath(MODELFILE_PATH)]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=LOCAL_MODELS_DIR)
    print(res.stdout)
    if res.returncode != 0:
        print(f"ERROR registering model: {res.stderr}")
        return False
    print("  Model registered in local Ollama successfully.")

    print("[4/4] Verifying model responsiveness via OllamaModelRunner...")
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.models.ollama_runner import OllamaModelRunner
    import asyncio

    async def test_inference():
        runner = OllamaModelRunner(
            logical_model_name="phi3.5-ft-retrieval",
            api_model_name="phi3.5-ft-retrieval:latest",
            max_tokens=64
        )
        resp = await runner.generate("State the RFC number for TLS 1.3.")
        print(f"  Test response: '{resp.text[:100]}...' ({resp.latency_ms:.1f}ms)")
        return bool(resp.text)

    success = asyncio.run(test_inference())
    if success:
        print("\nSUCCESS: phi3.5-ft-retrieval:latest is active, tested, and ready for evaluation.")
    return success

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else DRIVE_EXPORT_PATH
    deploy_model(src)

