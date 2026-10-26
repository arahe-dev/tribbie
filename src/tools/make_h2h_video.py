"""Side-by-side comparison video (< 60 s): Claude Opus alone vs Claude Opus + 16-D latent skill.

  .venv\\Scripts\\python.exe src\\tools\\make_h2h_video.py --direct-trial 0 --skill-trial 1
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
W, H, FPS = 1280, 720, 15


def font(sz, bold=False):
    return ImageFont.truetype(os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "segoeuib.ttf" if bold else "segoeui.ttf"), sz)


def card(lines, kicker=None):
    im = Image.new("RGB", (W, H), (14, 18, 24))
    d = ImageDraw.Draw(im)
    y = 250 if kicker else 280
    if kicker:
        d.text((110, 200), kicker, font=font(22, True), fill=(90, 200, 255))
    for i, (txt, sz, col, bold) in enumerate(lines):
        d.text((106, y), txt, font=font(sz, bold), fill=col)
        y += int(sz * 1.45)
    return im


def results_card(direct, skill):
    def summ(rs):
        ok = sum(r["success"] for r in rs)
        okt = [r["seconds"] for r in rs if r["success"]]
        t = f"{sum(okt) / len(okt):.0f} s" if okt else "never"
        calls = sum(r["opus_calls"] for r in rs) / len(rs)
        return ok, len(rs), t, calls
    a, b = summ(direct), summ(skill)
    im = Image.new("RGB", (W, H), (14, 18, 24))
    d = ImageDraw.Draw(im)
    d.text((106, 120), "RESULTS  ·  pick up the commanded object, measured from PhysX", font=font(22, True), fill=(90, 200, 255))
    cols = [(106, "Claude Opus alone", (224, 123, 40), a), (690, "Claude Opus + 16-D skill", (47, 190, 120), b)]
    for x, title, col, (ok, n, t, calls) in cols:
        d.text((x, 190), title, font=font(30, True), fill=col)
        d.text((x, 250), f"{ok}/{n}", font=font(96, True), fill=(240, 244, 248))
        d.text((x, 375), "successful picks", font=font(22), fill=(170, 180, 194))
        d.text((x, 430), f"{t}", font=font(48, True), fill=(240, 244, 248))
        d.text((x, 495), "mean time when successful", font=font(22), fill=(170, 180, 194))
        d.text((x, 545), f"{calls:.1f}", font=font(48, True), fill=(240, 244, 248))
        d.text((x, 610), "Opus calls per attempt", font=font(22), fill=(170, 180, 194))
    return im


def frames(mode, trial):
    return sorted((ROOT / "artifacts" / f"h2h_{mode}" / f"trial{trial}").glob("f_*.jpg"))


def pane(path, done_text=None):
    im = Image.open(path).convert("RGB").resize((636, 358), Image.LANCZOS)
    if done_text:
        d = ImageDraw.Draw(im)
        d.rectangle((0, 290, 636, 358), fill=(14, 18, 24))
        d.text((16, 306), done_text, font=font(24, True), fill=(110, 220, 140) if "picked" in done_text else (255, 190, 90))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direct-trial", type=int, required=True)
    ap.add_argument("--skill-trial", type=int, required=True)
    ap.add_argument("--realtime-seconds", type=float, default=30.0)
    ap.add_argument("--max-seconds", type=float, default=56.0)
    ap.add_argument("--out", default="deliverables/claude_vs_claude_plus_16d.mp4")
    args = ap.parse_args()
    direct = json.loads((ROOT / "results" / "h2h_direct.json").read_text())
    skill = json.loads((ROOT / "results" / "h2h_skill.json").read_text())
    fd, fs = frames("direct", args.direct_trial), frames("skill", args.skill_trial)
    rd, rs = direct[args.direct_trial], skill[args.skill_trial]
    end_d = f"{'picked' if rd['success'] else 'not picked'} after {rd['seconds']:.0f} s, {rd['opus_calls']} Opus calls"
    end_s = f"{'picked' if rs['success'] else 'not picked'} after {rs['seconds']:.0f} s, {rs['opus_calls']} Opus calls"

    out_frames = []
    title = card([("Same robot. Same cameras. Same task.", 44, (240, 244, 248), True),
                  ("Left: Claude Opus moves the gripper itself, one call at a time.", 26, (200, 206, 214), False),
                  ("Right: Claude Opus calls pick(), a 16-D latent policy runs the grasp at 15 Hz.", 26, (200, 206, 214), False)],
                 kicker="HEAD TO HEAD  ·  pick up the commanded object")
    out_frames += [title] * int(3.5 * FPS)

    budget = int((args.max_seconds - 3.5 - 6.0) * FPS)
    rt = int(args.realtime_seconds * FPS)
    longest = max(len(fd), len(fs))
    # real-time part, then a constant speed-up so the longer trial fits the remaining budget
    speed = max(1, -(-(longest - rt) // max(1, budget - rt))) if longest > rt else 1
    t = 0
    while t < longest and len(out_frames) < int(3.5 * FPS) + budget:
        canvas = Image.new("RGB", (W, H), (14, 18, 24))
        d = ImageDraw.Draw(canvas)
        d.text((10, 118), "CLAUDE OPUS ALONE", font=font(24, True), fill=(224, 123, 40))
        d.text((650, 118), "CLAUDE OPUS + 16-D LATENT SKILL", font=font(24, True), fill=(47, 190, 120))
        canvas.paste(pane(fd[min(t, len(fd) - 1)], end_d if t >= len(fd) else None), (2, 160))
        canvas.paste(pane(fs[min(t, len(fs) - 1)], end_s if t >= len(fs) else None), (642, 160))
        d.text((10, 540), "Clocks in each panel show real elapsed time, including Claude's thinking.", font=font(20), fill=(150, 160, 174))
        if t >= rt and speed > 1:
            d.text((10, 580), f"Playing at {speed}x speed", font=font(30, True), fill=(240, 244, 248))
        else:
            d.text((10, 580), "Playing in real time", font=font(30, True), fill=(240, 244, 248))
        out_frames.append(canvas)
        t += 1 if t < rt else speed
    canvas = out_frames[-1].copy()
    d = ImageDraw.Draw(canvas)
    out_frames += [canvas] * int(1.5 * FPS)
    out_frames += [results_card(direct, skill)] * int(6.0 * FPS)

    tmp = ROOT / "tmp" / "h2h_compose.mp4"
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    w = imageio_ffmpeg.write_frames(str(tmp), (W, H), fps=FPS, codec="libx264", quality=9, pix_fmt_out="yuv420p", macro_block_size=8)
    w.send(None)
    import numpy as np
    for f in out_frames:
        w.send(np.asarray(f))
    w.close()
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(tmp), "-vf", "minterpolate=fps=30:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1,format=yuv420p",
                    "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-movflags", "+faststart", str(out)], check=True)
    print("wrote", out, f"{len(out_frames) / FPS:.1f} s, speed-up {speed}x after {args.realtime_seconds:.0f} s")


if __name__ == "__main__":
    main()
