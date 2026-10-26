"""Claude Opus as a step-by-step robot programmer (via `claude -p`).

Opus receives camera images plus an execution log and writes a short program of
primitive calls. The executor runs it; learned skills report their own completion.
After the program ends, or any step fails, Opus is called again to verify the
result from the images and write the next program.
"""

from __future__ import annotations

import json

from course.planner import OpusPlanner

PROGRAM_PROMPT = """You are programming a mobile manipulator robot (holonomic base + one arm) in a simulated cell.
Layout: the pick table is NORTH of the start position. Three open bins (teal, magenta, gold) are far to the EAST,
behind a few obstacles. The robot's camera looks along its current heading.

You control it ONLY through these primitives:
  face(north|east|west)       rotate the base in place to that heading (do this before driving)
  goto(table|teal|magenta|gold)  drive to the docking spot of the table or of a bin (path planner)
  pick(<color>)                take that object from the table; the robot must be docked at the table facing north (learned skill)
  drop()                      release the held object into the bin in front; robot must be docked at a bin facing east (learned skill)
  done()                      the instruction is complete or cannot be completed

Recipe for one delivery from the table: pick(c) -> face(east) -> goto(bin) -> drop()
Coming back from a bin: face(west) -> goto(table) -> face(north)
Learned skills can fail. After each program you get the execution log and new images: verify from the IMAGES
where the objects really are (on the table, in which bin, on the floor) before deciding what to do next.
Keep programs short (one delivery at a time). Never pick an object you cannot see on the table."""

PROGRAM_SCHEMA = {
    "type": "object",
    "properties": {
        "observation": {"type": "string", "description": "What the images show: objects on the table, contents of each bin."},
        "assessment": {"type": "string", "description": "Did the previous program achieve its goal? Why/why not."},
        "program": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": ["face", "goto", "pick", "drop", "done"]},
                    "arg": {"type": "string", "enum": ["north", "east", "west", "table", "teal", "magenta", "gold",
                                                        "red", "orange", "green", "blue", "purple", "white", "none"]},
                },
                "required": ["op", "arg"],
                "additionalProperties": False,
            },
        },
        "say": {"type": "string", "description": "One short caption for the video (max 14 words)."},
    },
    "required": ["observation", "assessment", "program", "say"],
    "additionalProperties": False,
}


class OpusProgrammer(OpusPlanner):
    def _run(self, msg):
        import os
        import subprocess
        import time
        cmd = [self.exe, "-p", "--model", self.model, "--input-format", "stream-json", "--output-format", "stream-json",
               "--verbose", "--tools", "", "--strict-mcp-config", "--no-session-persistence",
               "--append-system-prompt", PROGRAM_PROMPT, "--json-schema", json.dumps(PROGRAM_SCHEMA), "--effort", self.effort]
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

    def _build_message(self, instruction, history, images, note):
        msg = super()._build_message(instruction, history, images, note)
        text = msg["message"]["content"][-1]["text"]
        msg["message"]["content"][-1]["text"] = text.replace(
            "Decide the next delivery (or done). Respond only through the structured output.",
            "Write the next short program. Respond only through the structured output.")
        return msg
