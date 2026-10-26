"""Generate the README's light/dark SVG pairs (pale-on-obsidian, matching HertaSDK) from the result files.

Backgrounds are transparent and box fills are a faint tint of the text colour, so the
assets sit cleanly on GitHub's light, dark, dimmed, high-contrast and colour-blind themes.

  python scripts/make_readme_assets.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets"
FONT = "ui-sans-serif, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Consolas, monospace"

THEMES = {
    "dark": dict(bg="#0d1117", fg="#f0f3f6", sub="#c6cdd5", mute="#9aa4b2", box="#161b22", line="#9aa4b2"),
    "light": dict(bg="#ffffff", fg="#111111", sub="#3a3f45", mute="#555555", box="#f6f8fa", line="#555555"),
}


def svg(w, h, title, desc, body, t, ident):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" '
            f'aria-labelledby="{ident}-title {ident}-desc">\n'
            f'  <title id="{ident}-title">{title}</title>\n  <desc id="{ident}-desc">{desc}</desc>\n'
            f'  <defs><marker id="{ident}-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="{t["line"]}"/></marker></defs>\n'
            f'  <g font-family="{FONT}">\n{body}  </g>\n</svg>\n')


def box(x, y, w, h, t, rx=9, sw=1.6, fill=None, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    f = f'fill="{fill}"' if fill else f'fill="{t["fg"]}" fill-opacity="0.05"'
    return f'    <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" {f} stroke="{t["fg"]}" stroke-width="{sw}"{d}/>\n'


def text(x, y, s, t, size=15, weight=400, color="fg", anchor="middle", family=None):
    f = f' font-family="{family}"' if family else ""
    return f'    <text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" font-weight="{weight}" fill="{t[color]}"{f}>{s}</text>\n'


def arrow(x1, y1, x2, y2, t, ident):
    return f'    <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{t["line"]}" stroke-width="1.6" marker-end="url(#{ident}-arrow)"/>\n'


def wordmark(t):
    # three small linked circles: one task, three roles
    body = '    <g transform="translate(28,0)">\n'
    for cx, cy in ((34, 28), (52, 28), (43, 44)):
        body += f'      <circle cx="{cx}" cy="{cy}" r="9" fill="none" stroke="{t["fg"]}" stroke-width="2.5"/>\n'
    body += (f'      <text x="74" y="46" font-size="28" font-weight="650" letter-spacing="-0.5" fill="{t["fg"]}">Tribbie'
             f'<tspan font-weight="400" fill="{t["mute"]}"> · latent control</tspan></text>\n    </g>\n')
    return svg(440, 72, "Tribbie wordmark", "Tribbie: latent control", body, t, "twm")


def architecture(t, ident="arch"):
    b = ""
    b += box(250, 16, 220, 40, t)
    b += text(360, 41, "Operator instruction", t, weight=650)
    b += box(150, 80, 420, 74, t, rx=12, sw=2, fill="none")
    b += text(360, 104, "Claude Opus  ·  operator", t, weight=700)
    b += text(360, 125, "reads cameras · writes pick / face / goto / drop · verifies · retries", t, size=13, color="sub")
    b += text(360, 144, "~10 s per call", t, size=12, color="mute")
    b += box(150, 180, 420, 92, t, rx=12, sw=2, fill="none")
    b += text(360, 204, "16-D latent skills  ·  student", t, weight=700)
    b += text(360, 225, "two small RGB cameras → keypoints → z ∈ ℝ¹⁶ → 4 gated experts", t, size=13, color="sub")
    b += text(360, 244, "→ gripper velocity · grip · phase", t, size=13, color="sub")
    b += text(360, 263, "15 Hz · 1.5 ms per decision · 255k parameters", t, size=12, color="mute")
    b += box(150, 298, 420, 40, t)
    b += text(360, 323, "Fixed controllers · IK · heading · path planner", t, size=14, weight=650)
    b += arrow(360, 56, 360, 78, t, ident) + arrow(360, 154, 360, 178, t, ident) + arrow(360, 272, 360, 296, t, ident)
    # teacher, off to the side: training only
    b += box(600, 180, 104, 92, t, dash="5 4", fill="none")
    b += text(652, 210, "Teacher", t, size=14, weight=650)
    b += text(652, 230, "scripted,", t, size=12, color="mute")
    b += text(652, 246, "true poses", t, size=12, color="mute")
    b += text(652, 262, "training only", t, size=12, color="mute")
    b += f'    <line x1="600" y1="226" x2="572" y2="226" stroke="{t["line"]}" stroke-width="1.4" stroke-dasharray="4 3" marker-end="url(#{ident}-arrow)"/>\n'
    b += text(18, 117, "slow", t, size=12, color="mute", anchor="start")
    b += text(18, 230, "fast", t, size=12, color="mute", anchor="start")
    return svg(720, 356, "Tribbie architecture",
               "An operator instruction goes to Claude Opus, which calls 16-D latent skills, which drive fixed controllers. A scripted teacher with true poses is used only to train the skills.",
               b, t, ident)


def head_to_head(t, ident="h2h"):
    def summ(f):
        rs = json.loads((ROOT / "results" / f).read_text())
        ok = [r for r in rs if r["success"]]
        return len(ok), len(rs), (sum(r["seconds"] for r in ok) / len(ok) if ok else None), sum(r["opus_calls"] for r in rs) / len(rs)
    rows = [("Claude Opus alone", summ("h2h_direct.json")), ("Claude Opus + 16-D skill", summ("h2h_skill.json"))]
    b = text(24, 34, "Pick up the commanded object  ·  success read from PhysX", t, size=14, weight=650, anchor="start")
    for i, (label, (ok, n, sec, calls)) in enumerate(rows):
        y = 64 + i * 78
        b += text(24, y + 20, label, t, size=15, weight=650, anchor="start")
        b += box(260, y, 300, 30, t, rx=6, sw=1.2)
        if ok:
            b += f'    <rect x="260" y="{y}" width="{300 * ok / n:.1f}" height="30" rx="6" fill="{t["fg"]}"/>\n'
        b += text(574, y + 21, f"{ok}/{n}", t, size=18, weight=700, anchor="start")
        detail = (f"mean {sec:.0f} s when successful" if sec else "never succeeded") + f" · {calls:.1f} Opus calls per attempt"
        b += text(260, y + 50, detail, t, size=12, color="mute", anchor="start")
    return svg(640, 216, "Head-to-head result",
               "Claude Opus alone picked the object in 0 of 6 trials; Claude Opus with the 16-D latent skill picked it in 4 of 6, in about 18 seconds.",
               b, t, ident)


def main():
    OUT.mkdir(exist_ok=True)
    for mode, t in THEMES.items():
        (OUT / f"tribbie-wordmark-{mode}.svg").write_text(wordmark(t), encoding="utf-8")
        (OUT / f"architecture-{mode}.svg").write_text(architecture(t, f"arch{mode[0]}"), encoding="utf-8")
        (OUT / f"head-to-head-{mode}.svg").write_text(head_to_head(t, f"h2h{mode[0]}"), encoding="utf-8")
    print("wrote", sorted(p.name for p in OUT.glob("*.svg")))


if __name__ == "__main__":
    main()
