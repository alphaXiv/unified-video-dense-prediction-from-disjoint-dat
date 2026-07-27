#!/usr/bin/env python3
"""Generate the report's SVG figures without optional plotting dependencies."""

from __future__ import annotations

import csv
import html
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "compact-unid"
IMAGES = REPORT / "images"
DATA = list(csv.DictReader((REPORT / "results.csv").open()))

BG = "#fbfaf7"
INK = "#172033"
MUTED = "#667085"
GRID = "#d9dde7"
BLUE = "#356ae6"
TEAL = "#129c8b"
ORANGE = "#ef8b2c"
RED = "#d85555"
PURPLE = "#7b61c9"


def esc(value) -> str:
    return html.escape(str(value))


def start(title: str, subtitle: str) -> list[str]:
    return [
        '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="540" viewBox="0 0 960 540">',
        f'<rect width="960" height="540" fill="{BG}"/>',
        f'<text x="58" y="52" fill="{INK}" font-family="Inter,Arial,sans-serif" font-size="25" font-weight="700">{esc(title)}</text>',
        f'<text x="58" y="79" fill="{MUTED}" font-family="Inter,Arial,sans-serif" font-size="14">{esc(subtitle)}</text>',
    ]


def text(x, y, value, size=13, color=INK, anchor="start", weight=400) -> str:
    return (
        f'<text x="{x}" y="{y}" fill="{color}" text-anchor="{anchor}" '
        f'font-family="Inter,Arial,sans-serif" font-size="{size}" font-weight="{weight}">{esc(value)}</text>'
    )


def line(x1, y1, x2, y2, color=GRID, width=1, dash="") -> str:
    attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"{attr}/>'


def finish(parts: list[str], name: str) -> None:
    parts.append("</svg>")
    IMAGES.mkdir(parents=True, exist_ok=True)
    (IMAGES / name).write_text("\n".join(parts) + "\n")


def mean(key: str) -> float:
    return sum(float(row[key]) for row in DATA) / len(DATA)


def headline() -> None:
    items = [
        ("Specialists", 1.000, 0.000, INK),
        ("Latent", 0.973, 0.015, BLUE),
        ("Latent + FT", 1.089, 0.058, TEAL),
        ("Direct joint", 1.026, 0.083, ORANGE),
        ("Frozen", 0.339, 0.009, RED),
        ("Pixel distill", 0.959, 0.025, PURPLE),
    ]
    p = start("Headline: specialist accuracy retained", "Eight long-schedule seeds; mean of normalized depth and segmentation retention; bars show 95% CI")
    x0, y0, w, h = 118, 108, 790, 350
    for tick in [0, 0.25, 0.5, 0.75, 1.0, 1.25]:
        y = y0 + h - tick / 1.25 * h
        p += [line(x0, y, x0 + w, y), text(x0 - 13, y + 5, f"{tick:.2f}", 12, MUTED, "end")]
    y_ref = y0 + h - 1.0 / 1.25 * h
    p += [line(x0, y_ref, x0 + w, y_ref, INK, 1.5, "6 5"), text(x0 + w - 4, y_ref - 8, "specialist parity", 12, INK, "end")]
    bw, gap = 88, 38
    for i, (label, value, ci, color) in enumerate(items):
        x = x0 + 34 + i * (bw + gap)
        bh = value / 1.25 * h
        y = y0 + h - bh
        p.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="6" fill="{color}"/>')
        if ci:
            ym = y0 + h - value / 1.25 * h
            yt = y0 + h - (value + ci) / 1.25 * h
            yb = y0 + h - (value - ci) / 1.25 * h
            p += [line(x + bw / 2, yt, x + bw / 2, yb, INK, 2), line(x + bw / 2 - 8, yt, x + bw / 2 + 8, yt, INK, 2), line(x + bw / 2 - 8, yb, x + bw / 2 + 8, yb, INK, 2)]
        p += [text(x + bw / 2, y - 10, f"{value * 100:.1f}%", 13, INK, "middle", 700), text(x + bw / 2, y0 + h + 26, label, 12, INK, "middle", 600)]
    p.append(text(58, 514, "Latent beats frozen decisively; its 5.3-point deficit to direct joint is uncertain. Projector fine-tuning is the robust recovery.", 14, INK, "start", 600))
    finish(p, "headline_retention.svg")


def memory() -> None:
    p = start("Peak training memory grows with pixel decoders", "Measured CUDA peak allocation at 256×256, batch 8; same shared backbone and task count")
    groups = [("2 tasks", 0.5382, 2.2840), ("8 tasks", 0.5382, 4.0421)]
    x0, y0, w, h = 130, 112, 700, 330
    for tick in range(5):
        y = y0 + h - tick / 4.5 * h
        p += [line(x0, y, x0 + w, y), text(x0 - 12, y + 5, f"{tick} GiB", 12, MUTED, "end")]
    for i, (label, latent, pixel) in enumerate(groups):
        cx = 300 + i * 350
        for j, (name, val, color) in enumerate([("Latent", latent, BLUE), ("Per-pixel", pixel, ORANGE)]):
            x = cx - 92 + j * 112
            bh = val / 4.5 * h
            y = y0 + h - bh
            p += [f'<rect x="{x}" y="{y}" width="88" height="{bh}" rx="7" fill="{color}"/>', text(x + 44, y - 10, f"{val:.2f} GiB", 14, INK, "middle", 700), text(x + 44, y0 + h + 25, name, 12, INK, "middle")]
        ratio = pixel / latent
        p += [text(cx - 16, y0 + h + 56, label, 15, INK, "middle", 700), text(cx - 16, y0 + h + 80, f"{ratio:.2f}× total peak", 13, MUTED, "middle", 600)]
    p.append(text(58, 520, "Incremental memory ratios were 5.53× (2 tasks) and 10.09× (8 tasks); the paper estimated 650/78 = 8.33×.", 14, INK, "start", 600))
    finish(p, "memory_scaling.svg")


def paired(name: str, left_key: str, right_key: str, left_label: str, right_label: str, left_color: str, right_color: str, footer: str, ymin=0.75, ymax=1.25) -> None:
    p = start(name, "Each line is one independent long-schedule seed; higher is better")
    x0, y0, w, h = 110, 110, 790, 330
    for tick in [0.8, 0.9, 1.0, 1.1, 1.2]:
        y = y0 + h - (tick - ymin) / (ymax - ymin) * h
        p += [line(x0, y, x0 + w, y), text(x0 - 12, y + 5, f"{tick:.1f}", 12, MUTED, "end")]
    yref = y0 + h - (1.0 - ymin) / (ymax - ymin) * h
    p.append(line(x0, yref, x0 + w, yref, INK, 1.5, "6 5"))
    for i, row in enumerate(DATA):
        x = x0 + 45 + i * 98
        a, b = float(row[left_key]), float(row[right_key])
        ya = y0 + h - (a - ymin) / (ymax - ymin) * h
        yb = y0 + h - (b - ymin) / (ymax - ymin) * h
        p += [line(x - 13, ya, x + 13, yb, "#9aa3b2", 2), f'<circle cx="{x - 13}" cy="{ya}" r="6" fill="{left_color}"/>', f'<circle cx="{x + 13}" cy="{yb}" r="6" fill="{right_color}"/>', text(x, y0 + h + 25, row["seed"], 12, INK, "middle")]
    p += [f'<circle cx="650" cy="76" r="6" fill="{left_color}"/>', text(664, 81, left_label, 12), f'<circle cx="770" cy="76" r="6" fill="{right_color}"/>', text(784, 81, right_label, 12), text(58, 516, footer, 14, INK, "start", 600)]
    finish(p, name.lower().replace(" ", "_").replace(":", "") + ".svg")


def pretraining() -> None:
    p = start("Diffusion pretraining improves absolute task accuracy", "Matched seed-0 fourfold schedule; pretrained U-Net versus identical randomly initialized architecture")
    depth = [("Specialist", 0.2145, 0.2708), ("Latent", 0.2137, 0.2611), ("Direct", 0.2183, 0.2671)]
    seg = [("Specialist", 0.02063, 0.01357), ("Latent", 0.01946, 0.00735), ("Direct", 0.01854, 0.01157)]
    panels = [(90, "NYU depth AbsRel ↓", depth, 0.30), (510, "ADE20K mIoU ↑", seg, 0.025)]
    for x0, title, values, ymax in panels:
        p.append(text(x0 + 165, 112, title, 16, INK, "middle", 700))
        y0, h = 140, 285
        for j, (label, pre, rnd) in enumerate(values):
            x = x0 + j * 115
            for k, (val, color) in enumerate([(pre, BLUE), (rnd, RED)]):
                xx = x + k * 43
                bh = val / ymax * h
                p.append(f'<rect x="{xx}" y="{y0 + h - bh}" width="36" height="{bh}" rx="4" fill="{color}"/>')
                p.append(text(xx + 18, y0 + h - bh - 7, f"{val:.3f}", 11, INK, "middle", 600))
            p.append(text(x + 38, y0 + h + 23, label, 11, INK, "middle"))
    p += [f'<rect x="349" y="474" width="16" height="16" rx="3" fill="{BLUE}"/>', text(372, 487, "pretrained", 12), f'<rect x="470" y="474" width="16" height="16" rx="3" fill="{RED}"/>', text(493, 487, "random init", 12), text(58, 520, "Random initialization worsened all six absolute comparisons; this supports a useful pretrained diffusion prior in the compact model.", 14, INK, "start", 600)]
    finish(p, "pretraining_control.svg")


if __name__ == "__main__":
    headline()
    memory()
    paired(
        "Seed variability: latent vs direct joint",
        "latent_retention",
        "direct_retention",
        "Latent",
        "Direct joint",
        BLUE,
        ORANGE,
        "Latent won only 2/8 seeds; paired mean difference −5.3 points (95% CI −12.6 to +2.0).",
    )
    paired(
        "Projector fine-tuning recovery",
        "latent_retention",
        "latent_ft_retention",
        "Before FT",
        "After FT",
        BLUE,
        TEAL,
        "Fine-tuning improved all 8 seeds by 11.6 points on average (95% CI +7.0 to +16.2).",
        ymin=0.9,
        ymax=1.25,
    )
    pretraining()
