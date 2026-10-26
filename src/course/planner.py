"""Claude Opus as the high-level planner, called through the Claude Code CLI (`claude -p`).

Opus sees camera images (table, bins, overview) plus the instruction and the
history so far, and returns one structured decision: which object to deliver to
which bin next, or that the instruction is complete. It never touches the
motors; the learned latent policy executes each decision. Calls run in a
background thread so the simulation keeps advancing in real time while Opus
thinks, and every call's wall-clock latency is recorded.
"""

from __future__ import annotations

import base64
import io
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

SYSTEM_PROMPT = """You are the task planner for a mobile manipulator robot in a simulated warehouse cell.
The robot has a learned low-level controller that can execute exactly one kind of command:
deliver ONE object, identified by its colour, from the pick table into ONE bin (teal, magenta or gold).
It drives around obstacles and drops the object into the bin on its own. You decide what to deliver next.

You receive camera images: a top-down view of the pick table, a view of the three bins, and a wide overview.
Judge the scene only from these images and the history. Objects are simple solids (cube, ball, tall cylinder,
puck, bar, capsule) in distinct colours. If a previous delivery failed (object not in the requested bin),
say so and decide whether to retry. Never invent objects you cannot see. When the instruction is fully
satisfied, or nothing further can be done, return done=true."""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "observation": {"type": "string", "description": "One sentence: what you see on the table and in the bins."},
        "previous_result": {"type": "string", "enum": ["none", "succeeded", "failed", "unclear"]},
        "done": {"type": "boolean"},
        "object_color": {"type": "string", "enum": ["red", "orange", "green", "blue", "purple", "white", "none"]},
        "bin": {"type": "string", "enum": ["teal", "magenta", "gold", "none"]},
        "say": {"type": "string", "description": "Short narration for the video caption (max 15 words)."},
    },
    "required": ["observation", "previous_result", "done", "object_color", "bin", "say"],
    "additionalProperties": False,
}


def find_claude() -> str:
    exe = shutil.which("claude")
    if exe:
        return exe
    local = Path.home() / ".local" / "bin" / ("claude.exe" if os.name == "nt" else "claude")
    if local.exists():
        return str(local)
    raise FileNotFoundError("Claude Code CLI not found (install it and run `claude auth login`).")


def _png_b64(img: np.ndarray, max_side: int = 768) -> str:
    im = Image.fromarray(img)
    scale = max_side / max(im.size)
    if scale < 1.0:
        im = im.resize((int(im.width * scale), int(im.height * scale)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class OpusPlanner:
    def __init__(self, model: str = "opus", effort: str = "low", log_path: str | None = None):
        self.exe = find_claude()
        self.model = model
        self.effort = effort
        self.log_path = Path(log_path) if log_path else None
        self.calls = []
        self._thread = None
        self._result = None
        self._started = None
        # The desktop app injects its own session variables; a plain terminal has none of these.
        self.env = {k: v for k, v in os.environ.items() if not (k.startswith("CLAUDE") or k.startswith("ANTHROPIC"))}
        for k in ("PYTHONPATH", "PYTHONHOME"):
            self.env.pop(k, None)

    # -------------------------------------------------------------- request
    def _build_message(self, instruction, history, images, note):
        content = []
        for label, img in images:
            content.append({"type": "text", "text": f"[{label}]"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _png_b64(img)}})
        lines = [f"Instruction from the operator: {instruction}"]
        if history:
            lines.append("History of your decisions and what the robot's controller reported:")
            for h in history:
                lines.append(f"- {h}")
        else:
            lines.append("No deliveries have been attempted yet.")
        if note:
            lines.append(note)
        lines.append("Decide the next delivery (or done). Respond only through the structured output.")
        content.append({"type": "text", "text": "\n".join(lines)})
        return {"type": "user", "message": {"role": "user", "content": content}}

    def _run(self, msg):
        cmd = [self.exe, "-p", "--model", self.model, "--input-format", "stream-json", "--output-format", "stream-json",
               "--verbose", "--tools", "", "--strict-mcp-config", "--no-session-persistence",
               "--append-system-prompt", SYSTEM_PROMPT, "--json-schema", json.dumps(OUTPUT_SCHEMA), "--effort", self.effort]
        t0 = time.perf_counter()
        record = {"started": time.time()}
        try:
            proc = subprocess.run(cmd, input=json.dumps(msg) + "\n", capture_output=True, text=True, timeout=240,
                                  env=self.env, encoding="utf-8", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            result = None
            for line in proc.stdout.splitlines():
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ev.get("type") == "system" and ev.get("subtype") == "init":
                    record["model"] = ev.get("model")
                if ev.get("type") == "result":
                    result = ev
            if result is None or result.get("is_error") or not result.get("structured_output"):
                record["error"] = (result or {}).get("result") or proc.stderr[-400:] or "no result"
                decision = None
            else:
                decision = result["structured_output"]
                record["api_ms"] = result.get("duration_api_ms")
        except Exception as exc:  # noqa: BLE001
            record["error"] = repr(exc)
            decision = None
        record["wall_s"] = round(time.perf_counter() - t0, 2)
        record["decision"] = decision
        self.calls.append(record)
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        self._result = (decision, record)

    def request(self, instruction, history, images, note=""):
        """Start an asynchronous planning call."""
        if self.busy():
            raise RuntimeError("planner call already in flight")
        self._result = None
        self._started = time.perf_counter()
        msg = self._build_message(instruction, history, images, note)
        self._thread = threading.Thread(target=self._run, args=(msg,), daemon=True)
        self._thread.start()

    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def elapsed(self) -> float:
        return 0.0 if self._started is None else time.perf_counter() - self._started

    def poll(self):
        """Return (decision, record) once the call finished, else None."""
        if self._thread is None or self.busy():
            return None
        out, self._result, self._thread = self._result, None, None
        return out
