import urllib.request
import json

candidates = [
    # Coding <=5B
    "Qwen/Qwen2.5-Coder-3B-Instruct",
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    # Math <=5B
    "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    # Reasoning / Logic <=5B
    "microsoft/Phi-3.5-mini-instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    # General / QA / Retrieval / Science <=5B
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "google/gemma-2-2b-it"
]

print("=== Verifying Hugging Face Model Checkpoints ===")
for model_id in candidates:
    url = f"https://huggingface.co/api/models/{model_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "curl/7.88.1"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            print(f"  [EXISTS] {model_id} | id: {data.get('id')} | tag: {data.get('pipeline_tag')}")
    except urllib.error.HTTPError as e:
        print(f"  [HTTP {e.code}] {model_id}")
    except Exception as e:
        print(f"  [ERROR] {model_id}: {e}")

