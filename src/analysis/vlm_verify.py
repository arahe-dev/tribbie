"""Score Claude Opus (via `claude -p`) at locating objects from camera images vs PhysX ground truth."""

import base64
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATES = ROOT / "results" / "vlm_states"
LOCS = ["table", "teal", "magenta", "gold", "floor", "not_visible"]
SCHEMA = {
    "type": "object",
    "properties": {c: {"type": "string", "enum": LOCS} for c in ("red", "blue", "green")},
    "required": ["red", "blue", "green"],
    "additionalProperties": False,
}
PROMPT = ("A simulated robot cell has a pick table, three open bins (teal, magenta, gold) and a floor area. "
          "There is a red cube, a blue ball and a green puck somewhere. Using only the images, say where each object is: "
          "on the table, inside the teal/magenta/gold bin, on the floor, or not_visible if you cannot see it.")
EXE = shutil.which("claude") or str(Path.home() / ".local" / "bin" / "claude.exe")
ENV = {k: v for k, v in os.environ.items() if not (k.startswith("CLAUDE") or k.startswith("ANTHROPIC"))}


def ask(state, views):
    content = []
    for v in views:
        data = base64.b64encode((STATES / state["files"][v]).read_bytes()).decode()
        content += [{"type": "text", "text": f"[{v} camera]"}, {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}]
    content.append({"type": "text", "text": PROMPT})
    msg = {"type": "user", "message": {"role": "user", "content": content}}
    cmd = [EXE, "-p", "--model", "opus", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
           "--tools", "", "--strict-mcp-config", "--no-session-persistence", "--json-schema", json.dumps(SCHEMA), "--effort", "low"]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, input=json.dumps(msg) + "\n", capture_output=True, text=True, encoding="utf-8", env=ENV, timeout=240)
    ans = None
    for line in p.stdout.splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "result":
            ans = ev.get("structured_output")
    return {"state": state["state"], "views": views, "truth": state["truth"], "answer": ans, "seconds": round(time.perf_counter() - t0, 2)}


def main():
    states = json.loads((STATES / "states.json").read_text())
    jobs = [(s, v) for s in states for v in (["table", "bins", "overview"], ["overview"])]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda j: ask(*j), jobs))
    out = {}
    for cond in ("table+bins+overview", "overview"):
        rows = [r for r in results if "+".join(r["views"]) == cond]
        tot = ok = 0
        per_loc = {}
        for r in rows:
            for c, t in r["truth"].items():
                a = (r["answer"] or {}).get(c)
                tot += 1
                ok += int(a == t)
                d = per_loc.setdefault(t, [0, 0]); d[0] += int(a == t); d[1] += 1
        out[cond] = {"accuracy": ok / max(tot, 1), "objects": tot, "per_true_location": {k: f"{v[0]}/{v[1]}" for k, v in per_loc.items()},
                     "mean_seconds": sum(r["seconds"] for r in rows) / max(len(rows), 1)}
    (ROOT / "results" / "vlm_verify.json").write_text(json.dumps({"summary": out, "calls": results}, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
