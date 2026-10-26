"""Fill measured numbers into paper.html and print it to PDF with headless Edge."""

import json
import statistics as st
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "paper"


def ablation():
    names = [("both16", "Both cameras, 16-D latent"), ("head16", "Head camera only, 16-D"), ("wrist16", "Wrist camera only, 16-D"),
             ("both4", "Both cameras, 4-D latent"), ("both64", "Both cameras, 64-D latent")]
    rows, res = [], {}
    for key, label in names:
        runs = []
        for suffix in ("", "_s8", "_s9"):
            f = ROOT / "results" / "ablations" / f"{key}{suffix}.json"
            if f.exists():
                runs.append(json.loads(f.read_text())["val"])
        def ms(k, scale=1.0, fmt="{:.3f}"):
            v = [r[k] * scale for r in runs]
            sd = st.stdev(v) if len(v) > 1 else 0.0
            return fmt.format(st.mean(v)) + " &plusmn; " + fmt.format(sd), st.mean(v)
        a, am = ms("act_l1_scaled", 1, "{:.4f}")
        g, _ = ms("bin_acc", 100, "{:.1f}")
        p, pm = ms("phase_acc", 100, "{:.1f}")
        res[key] = (am, pm, len(runs))
        rows.append(f"<tr><td>{label}</td><td>{a}</td><td>{g}&nbsp;%</td><td>{p}&nbsp;%</td></tr>")
    b, h, w, l4, l64 = (res[k] for k in ("both16", "head16", "wrist16", "both4", "both64"))
    text = (f"The wrist camera carries most of the task information: removing it lowers held-out phase accuracy from "
            f"{b[1]:.1f}&nbsp;% to {h[1]:.1f}&nbsp;%, while removing the head camera leaves {w[1]:.1f}&nbsp;%. "
            f"A 4-D latent is too small ({l4[1]:.1f}&nbsp;%), and a 64-D latent is slightly better than 16-D "
            f"({l64[1]:.1f}&nbsp;% vs. {b[1]:.1f}&nbsp;%; action error {l64[0]:.4f} vs. {b[0]:.4f}). "
            "Gripper accuracy is above 99&nbsp;% in every variant, because the grip decision is easy once the phase is known. "
            "These are offline results on teacher data; the closed-loop experiments use the 16-D model with both cameras, "
            "because the head camera is still needed to drive and to find the target object on the table.")
    return "\n".join(rows), text


def vlm():
    r = json.loads((ROOT / "results" / "vlm_verify.json").read_text())
    sm = r["summary"]
    rows = []
    stats = {}
    for key, label in (("table+bins+overview", "Table + bins + overview"), ("overview", "Overview only")):
        v = sm[key]
        p = v["per_true_location"]
        bins = [sum(int(p.get(b, "0/0").split("/")[i]) for b in ("teal", "magenta", "gold")) for i in (0, 1)]
        calls = [c for c in r["calls"] if "+".join(c["views"]) == key]
        answered = correct = abstain = 0
        for c in calls:
            for obj, truth in c["truth"].items():
                a = (c["answer"] or {}).get(obj)
                if a == "not_visible":
                    abstain += 1
                else:
                    answered += 1
                    correct += int(a == truth)
        stats[key] = (v["accuracy"], answered, correct, abstain)
        rows.append(f"<tr><td>{label}</td><td>{v['accuracy'] * 100:.0f}&nbsp;%</td><td>{p.get('table', '-')}</td>"
                    f"<td>{bins[0]}/{bins[1]}</td><td>{p.get('floor', '-')}</td><td>{v['mean_seconds']:.0f}&nbsp;s</td></tr>")
    full, ov = stats["table+bins+overview"], stats["overview"]
    text = ("To measure how far the programmer can be trusted as a verifier, we rendered 24 scenes in which each object was "
            "placed on the table, inside one of the bins or on the floor, and asked Opus where each object was, given either the "
            "table, bins and overview views or the overview alone (Table&nbsp;2). With all three views it located "
            f"{full[0] * 100:.0f}&nbsp;% of objects; with the overview alone, {ov[0] * 100:.0f}&nbsp;%. Almost every error was an "
            f"abstention: with all views it answered <i>not visible</i> {full[3]} times and gave a wrong location only "
            f"{full[1] - full[2]} time, so {full[2]}/{full[1]} of its definite answers were correct. Objects inside bins were the "
            "hard case, because the bin walls hide them from oblique cameras; the overview camera alone almost never saw into a bin. "
            "Calls ran four at a time, so the time per call includes queueing.")
    return "\n".join(rows), text


def h2h():
    def summ(f):
        rs = json.loads((ROOT / "results" / f).read_text())
        ok = [r for r in rs if r["success"]]
        return len(ok), len(rs), (st.mean(r["seconds"] for r in ok) if ok else None), st.mean(r["opus_calls"] for r in rs)
    a, b = summ("h2h_direct.json"), summ("h2h_skill.json")
    t = lambda x: f", mean {x:.0f}&nbsp;s when successful" if x else ""
    return ("To test the hierarchy as a deployment rather than in pieces, we gave the same instruction (pick up the object of a "
            "given colour, positions jittered per trial) to two systems that share the robot, the side and wrist cameras, the VLM and its "
            "time limit. <i>VLM alone</i> returns, per call, a gripper displacement of at most 8&nbsp;cm and open/close, executed by "
            "the fixed IK layer. <i>VLM + skill</i> returns pick(colour) and the 16-D policy closes the loop at 15&nbsp;Hz. Success is "
            "read from PhysX (object held 5&nbsp;cm above the table) and times are wall clock, including VLM thinking. "
            f"The VLM alone succeeded in {a[0]}/{a[1]} trials{t(a[2])}, using {a[3]:.1f} calls per attempt; with the skill, "
            f"{b[0]}/{b[1]}{t(b[2])}, using {b[3]:.1f} calls. The gap is one of bandwidth rather than reasoning: a VLM that observes "
            "every few seconds cannot servo a gripper onto a rolling ball, while the latent policy corrects every 67&nbsp;ms. "
            "Six trials per condition bound the gap only roughly.")


def main():
    html = (P / "paper.html").read_text(encoding="utf-8")
    a_rows, a_text = ablation()
    v_rows, v_text = vlm()
    html = html.replace("ABLATION_ROWS", a_rows).replace("ABLATION_TEXT", a_text).replace("VLM_ROWS", v_rows).replace("VLM_TEXT", v_text).replace("H2H_TEXT", h2h())
    out = P / "paper_filled.html"
    out.write_text(html, encoding="utf-8")
    edge = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    pdf = P / "latent_vision_paper.pdf"
    subprocess.run([edge, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", out.as_uri()],
                   check=False, timeout=120)
    print("pdf", pdf, pdf.exists(), pdf.stat().st_size if pdf.exists() else 0)


if __name__ == "__main__":
    main()
