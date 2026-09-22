import urllib.request
import json
import sys
import time

def pull_model(model_name: str):
    print(f"Starting pull of {model_name} from Ollama...")
    url = "http://localhost:11434/api/pull"
    req = urllib.request.Request(
        url,
        data=json.dumps({"name": model_name}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        last_status = ""
        for line in resp:
            if line:
                d = json.loads(line.decode("utf-8"))
                status = d.get("status", "")
                completed = d.get("completed", 0)
                total = d.get("total", 0)
                if total > 0:
                    pct = (completed / total) * 100.0
                    sys.stdout.write(f"\r{status}: {pct:.1f}% ({completed}/{total} bytes)")
                    sys.stdout.flush()
                elif status != last_status:
                    print(f"\n{status}")
                    last_status = status
    print(f"\nSuccessfully finished pulling {model_name}!")

if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "phi3.5"
    pull_model(model)

