"""Agentic-loop demo: multi-step tool calling against a local deepseaport server.

The harness owns the tools; the model only decides calls. Loop:
  1. POST /v1/chat/completions with tools
  2. if finish_reason == tool_calls: run tools locally, append results, repeat
  3. else: print final answer.

Usage: python scripts/agentic_demo.py "weather in Beijing and 12*34?"
"""

import json
import sys
import urllib.request

BASE = "http://127.0.0.1:5001"
API_KEY = "sk-local-dev"

TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {"type": "object", "properties": {
            "location": {"type": "string", "description": "City name"}},
            "required": ["location"]}}},
    {"type": "function", "function": {
        "name": "calc",
        "description": "Evaluate an arithmetic expression.",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string", "description": "e.g. 12*34"}},
            "required": ["expression"]}}},
]


def local_tool(name: str, args: dict):
    if name == "get_weather":
        return {"location": args.get("location"), "temp": "24C", "note": "demo data"}
    if name == "calc":
        expr = str(args.get("expression", ""))
        allowed = set("0123456789+-*/(). %")
        if not set(expr) <= allowed:
            return {"error": "unsafe expression"}
        return {"result": eval(expr, {"__builtins__": {}}, {})}  # noqa: S307 demo only
    return {"error": f"unknown tool {name}"}


def post(payload: dict) -> dict:
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.load(resp)


def main() -> int:
    question = sys.argv[1] if len(sys.argv) > 1 else "What is 12*34 and weather in Beijing?"
    messages = [{"role": "user", "content": question}]
    for step in range(6):
        resp = post({"model": "deepseek-flash", "messages": messages, "tools": TOOLS})
        msg = resp["choices"][0]["message"]
        print(f"--- step {step} finish={resp['choices'][0]['finish_reason']} ---")
        if msg.get("tool_calls"):
            messages.append({"role": "assistant", "content": msg.get("content") or "",
                             "tool_calls": msg["tool_calls"]})
            for call in msg["tool_calls"]:
                fn = call["function"]
                result = local_tool(fn["name"], json.loads(fn["arguments"]))
                print(f"call {call['id']}: {fn['name']}{fn['arguments']} -> {result}")
                messages.append({"role": "tool", "tool_call_id": call["id"],
                                 "name": fn["name"], "content": json.dumps(result)})
        else:
            print(msg.get("content"))
            return 0
    print("max steps reached")
    return 1


if __name__ == "__main__":
    sys.exit(main())
