"""Video overlays: what the policy sees, what it outputs, and how fast."""

from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_CACHE = {}


def font(size: int, bold: bool = False):
    key = (size, bold)
    if key not in _FONT_CACHE:
        names = ["segoeuib.ttf", "arialbd.ttf"] if bold else ["segoeui.ttf", "arial.ttf"]
        f = None
        for n in names:
            path = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", n)
            if os.path.exists(path):
                f = ImageFont.truetype(path, size)
                break
        _FONT_CACHE[key] = f or ImageFont.load_default()
    return _FONT_CACHE[key]


ACCENT = (90, 200, 255)
GOOD = (110, 220, 140)
WARN = (255, 190, 90)
TEXT = (235, 240, 245)
DIM = (160, 170, 185)
PANEL = (12, 16, 22, 190)


def panel(draw, box, radius=10):
    draw.rounded_rectangle(box, radius=radius, fill=PANEL)


def inset(base: Image.Image, img: np.ndarray, xy, size, label):
    im = Image.fromarray(img).resize((size, size), Image.Resampling.NEAREST)
    d = ImageDraw.Draw(base, "RGBA")
    x, y = xy
    panel(d, (x - 6, y - 26, x + size + 6, y + size + 6), 8)
    base.paste(im, (x, y))
    d.text((x, y - 23), label, font=font(15, True), fill=TEXT)


def bars(draw, xy, values, labels, width=150, height=12, color=ACCENT, gap=4):
    x, y = xy
    for i, (v, lab) in enumerate(zip(values, labels)):
        yy = y + i * (height + gap)
        draw.rectangle((x, yy, x + width, yy + height), fill=(40, 48, 58, 255))
        draw.rectangle((x, yy, x + int(width * float(np.clip(v, 0, 1))), yy + height), fill=color + (255,))
        draw.text((x + width + 8, yy - 3), lab, font=font(13), fill=DIM)


def latent_strip(draw, xy, latent, cell=14):
    x, y = xy
    for i, v in enumerate(latent):
        t = float(np.clip((v + 1) / 2, 0, 1))
        c = (int(40 + 200 * t), int(80 + 120 * (1 - abs(2 * t - 1))), int(240 - 200 * t))
        draw.rectangle((x + i * cell, y, x + (i + 1) * cell - 2, y + 18), fill=c + (255,))


def wrap(text, width_chars):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width_chars:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return lines


def policy_panel(base: Image.Image, xy, phase: str, gate, latent, ms: float, title="LATENT POLICY (camera-only)"):
    d = ImageDraw.Draw(base, "RGBA")
    x, y = xy
    w, h = 410, 150
    panel(d, (x, y, x + w, y + h))
    d.text((x + 12, y + 8), title, font=font(15, True), fill=ACCENT)
    d.text((x + 12, y + 30), f"phase (own prediction):  {phase}", font=font(15), fill=TEXT)
    d.text((x + 12, y + 52), f"decision latency:  {ms:5.1f} ms", font=font(15), fill=TEXT)
    d.text((x + 12, y + 78), f"{len(latent)}-D latent", font=font(13), fill=DIM)
    latent_strip(d, (x + 12, y + 96), latent)
    d.text((x + 250, y + 78), "expert gate", font=font(13), fill=DIM)
    bars(d, (x + 250, y + 96), gate, [f"E{i + 1}" for i in range(len(gate))], width=110, height=9, gap=3)
