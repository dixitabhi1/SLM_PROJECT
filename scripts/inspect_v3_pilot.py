import json

with open('results/v3_pilot/slm_pipeline_responses.jsonl', encoding='utf-8') as f:
    slm = {r['query_id']: r for r in (json.loads(l) for l in f if l.strip())}
with open('results/v3_pilot/llm_baseline_responses.jsonl', encoding='utf-8') as f:
    base = {r['query_id']: r for r in (json.loads(l) for l in f if l.strip())}

print(f"{'Query ID':16s} | {'SLM Chars':10s} | {'SLM Latency':12s} | {'Base Chars':10s} | {'Base Latency':12s}")
print("-" * 72)
for qid in sorted(slm.keys()):
    s = slm[qid]
    b = base[qid]
    s_chars = len(s["response_text"])
    b_chars = len(b["response_text"])
    s_lat = f"{s.get('wall_clock_latency_ms', 0)/1000:6.1f}s"
    b_lat = f"{b.get('latency_ms', 0)/1000:4.1f}s"
    print(f"{qid:16s} | {s_chars:10d} | {s_lat:12s} | {b_chars:10d} | {b_lat:12s}")

