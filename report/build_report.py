"""Fill measured numbers into report.html and print it to PDF with headless Edge."""

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "report"


def summ(rs):
    ok = [r for r in rs if r["success"]]
    t = sum(r["seconds"] for r in ok) / len(ok) if ok else None
    return len(ok), len(rs), t, sum(r["opus_calls"] for r in rs) / len(rs)


def h2h():
    d = json.loads((ROOT / "results" / "h2h_direct.json").read_text())
    s = json.loads((ROOT / "results" / "h2h_skill.json").read_text())
    rows, res = [], {}
    for label, rs in (("Opus alone", d), ("Opus + 16-D skill", s)):
        ok, n, t, c = res[label] = summ(rs)
        rows.append(f"<tr><td>{label}</td><td>{ok}/{n}</td><td>{'never' if t is None else f'{t:.0f}&nbsp;s'}</td><td>{c:.1f}</td></tr>")
    a, b = res["Opus alone"], res["Opus + 16-D skill"]
    fail_d = [r for r in d if not r["success"]]
    text = (f"With the skill, Opus picked the object in {b[0]} of {b[1]} trials"
            + (f", in {b[2]:.0f}&nbsp;s on average" if b[2] else "")
            + f". Alone, it succeeded in {a[0]} of {a[1]}"
            + (f" (mean {a[2]:.0f}&nbsp;s when it did)" if a[2] else "")
            + (f"; the failures used {sum(r['opus_calls'] for r in fail_d) / len(fail_d):.1f} calls on average before it stopped or reached the 14-call limit" if fail_d else "")
            + ". The difference is not intelligence but bandwidth: one Opus call takes several seconds, so it sees the scene and moves "
            "the hand a handful of times per minute, while closing the fingers on a ball needs corrections every few tens of milliseconds. "
            "The 16-D policy supplies that loop; Opus supplies the choice of what to pick and whether to retry.")
    return "\n".join(rows), text


def main():
    html = (P / "report.html").read_text(encoding="utf-8")
    prev = (P / "report_filled.html").read_text(encoding="utf-8")
    abl = re.search(r"Phase accuracy</th><th>Action error</th></tr></thead>\s*<tbody>(.*?)</tbody>", prev, re.S).group(1)
    rows, text = h2h()
    html = html.replace("ABLATION_ROWS_REPORT", abl).replace("H2H_ROWS", rows).replace("H2H_TEXT", text)
    out = P / "report_filled.html"
    out.write_text(html, encoding="utf-8")
    edge = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    pdf = P / "project_report.pdf"
    subprocess.run([edge, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", out.as_uri()],
                   check=False, timeout=120)
    print("pdf", pdf, pdf.exists())


if __name__ == "__main__":
    main()
