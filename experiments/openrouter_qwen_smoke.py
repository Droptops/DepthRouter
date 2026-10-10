"""OpenRouter Qwen black-box smoke benchmark; no internal routing claims."""
import json
import os
import sys
import time
import urllib.error
import urllib.request

MODEL = os.getenv("OPENROUTER_MODEL", "qwen/qwen3-8b")
KEY = os.environ.get("OPENROUTER_API_KEY")
if not KEY:
    raise SystemExit("OPENROUTER_API_KEY missing; no request made")
prompts = [
    "Return only the integer result of 17 * 19.",
    "Return only the next number: 2, 3, 5, 8, 13, 21,",
    "In one sentence explain the difference between a cache hit and a cache miss.",
]
results = []
for i, prompt in enumerate(prompts):
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 128,
    }).encode()
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"},
        method="POST",
    )
    start = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            body = json.load(response)
        choices = body.get("choices", [])
        answer = choices[0].get("message", {}).get("content", "") if choices else ""
        results.append({"prompt_id": i, "response": answer, "latency_seconds": round(time.monotonic()-start, 3), "usage": body.get("usage", {})})
    except urllib.error.HTTPError as error:
        print("OpenRouter HTTP error:", error.code, file=sys.stderr)
        raise SystemExit(1)
output = {"experiment": "openrouter_qwen_black_box_smoke_v1", "model": MODEL, "scope": "output-only; not a sparse-routing or cache-traffic measurement", "results": results}
with open("openrouter_qwen_smoke.json", "w", encoding="utf-8") as file:
    json.dump(output, file, indent=2)
print(json.dumps({"model": MODEL, "requests": len(results), "result_file": "openrouter_qwen_smoke.json"}, indent=2))
