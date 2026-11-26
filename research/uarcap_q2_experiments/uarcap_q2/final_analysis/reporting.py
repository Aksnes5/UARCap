"""Paper-ready figure/table generation from analysis artifacts. Missing inputs are marked
PENDING (never filled with fabricated values)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARMS = ("ARM_A0", "ARM_A1", "ARM_A7")
COLORS = {"ARM_A0": "#4C72B0", "ARM_A1": "#DD8452", "ARM_A7": "#55A868"}


def _save(fig, out: Path, stem: str, formats=("pdf", "svg", "png")):
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        p = out / f"{stem}.{fmt}"
        fig.savefig(p, bbox_inches="tight", dpi=220 if fmt == "png" else 300)
        paths.append(str(p))
    plt.close(fig)
    return paths


def _pending(out: Path, stem: str, reason: str):
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{stem}.PENDING.txt").write_text(f"PENDING: {reason}\n", encoding="utf-8")
    return []


def plot_e2_fact_changes(transitions: dict, out: Path) -> list[str]:
    names = list(transitions)
    fields = ["added_correct", "lost_correct", "added_error", "removed_error", "replacements", "label_flips"]
    x = np.arange(len(names)); w = 0.13
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, f in enumerate(fields):
        ax.bar(x + (i - 2.5) * w, [transitions[n].get(f, 0) for n in names], w, label=f.replace("_", " "))
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15); ax.set_ylabel("facts"); ax.legend(fontsize=7, ncol=2)
    ax.set_title("E2 paired fact changes (machine-assisted)")
    return _save(fig, out, "fig_e2_fact_changes")


def plot_core_coverage(desc: dict, model_names: list[str], out: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(7, 3.2)); x = np.arange(len(model_names)); w = 0.25
    for i, arm in enumerate(ARMS):
        vals = [desc[m][arm].get("core_coverage_mean") or 0 for m in model_names]
        ax.bar(x + (i - 1) * w, vals, w, label=arm, color=COLORS[arm])
    ax.set_xticks(x); ax.set_xticklabels(model_names); ax.set_ylim(0, 1); ax.set_ylabel("core-event coverage"); ax.legend()
    return _save(fig, out, "fig_core_coverage")


def plot_dimensions(desc: dict, model: str, out: Path) -> list[str]:
    dims = sorted({k for arm in ARMS for k in desc[model][arm].get("dimensions", {})})
    if not dims: return _pending(out, "fig_dimensions", "no dimension data")
    fig, ax = plt.subplots(figsize=(8, 3.2)); x = np.arange(len(dims)); w = 0.25
    for i, arm in enumerate(ARMS):
        ax.bar(x + (i - 1) * w, [desc[model][arm].get("dimensions", {}).get(d, 0) for d in dims], w, label=arm, color=COLORS[arm])
    ax.set_xticks(x); ax.set_xticklabels([d.replace("_", " ") for d in dims], rotation=20); ax.set_ylabel("facts"); ax.legend()
    return _save(fig, out, "fig_dimensions")


def plot_model_comparison(desc: dict, models: list[str], metric: str, out: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(7, 3.2)); x = np.arange(len(models)); w = 0.25
    for i, arm in enumerate(ARMS):
        ax.bar(x + (i - 1) * w, [desc[m][arm].get(metric) or 0 for m in models], w, label=arm, color=COLORS[arm])
    ax.set_xticks(x); ax.set_xticklabels(models); ax.set_ylabel(metric); ax.legend()
    return _save(fig, out, f"fig_compare_{metric}")


def plot_agreement_matrix(rel: dict, out: Path, stem="fig_judge_confusion") -> list[str]:
    conf = rel.get("confusion_gemma_to_llama") or {}
    if not conf: return _pending(out, stem, "no cross-judge data")
    labels = ["entailed", "contradicted", "unmentioned"]
    m = np.zeros((3, 3))
    for i, a in enumerate(labels):
        for j, b in enumerate(labels): m[i, j] = conf.get(f"{a}->{b}", 0)
    fig, ax = plt.subplots(figsize=(4.2, 3.6)); im = ax.imshow(m, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_yticks(range(3)); ax.set_xticklabels(labels, rotation=20); ax.set_yticklabels(labels)
    ax.set_xlabel("Llama"); ax.set_ylabel("Gemma")
    for i in range(3):
        for j in range(3): ax.text(j, i, int(m[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(im); ax.set_title(f"Judge confusion (κ={rel.get('cohen_kappa')})")
    return _save(fig, out, stem)


def plot_effect_sizes(stats: dict, out: Path, stem="fig_effect_sizes") -> list[str]:
    rows = []
    for name, res in stats.items():
        for pair, v in res.items():
            if isinstance(v, dict) and v.get("mean_diff") is not None:
                rows.append((f"{name}:{pair}", v["mean_diff"], v.get("ci95") or [None, None]))
    if not rows: return _pending(out, stem, "no paired effect estimates")
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.35 * len(rows))))
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], color="#4C72B0")
    for i, r in enumerate(rows):
        if r[2][0] is not None: ax.plot([r[2][0], r[2][1]], [i, i], color="black", linewidth=1)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=7); ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("paired difference (95% bootstrap CI)")
    return _save(fig, out, stem)


def plot_flow(flow: dict, out: Path, stem="fig_sample_flow") -> list[str]:
    fig, ax = plt.subplots(figsize=(6, 4)); ax.axis("off")
    steps = [(k, v) for k, v in flow.items()]
    y = 1.0
    for i, (label, count) in enumerate(steps):
        ax.text(0.5, y, f"{label}: {count}", ha="center", va="center", fontsize=9,
                bbox=dict(boxstyle="round", fc="#EAF2F8", ec="#4C72B0"))
        if i < len(steps) - 1: ax.annotate("", xy=(0.5, y - 0.12), xytext=(0.5, y - 0.02), arrowprops=dict(arrowstyle="->"))
        y -= 0.14
    return _save(fig, out, stem)


def plot_budget_curve(historical: list[dict], out: Path, stem="fig_budget_curve") -> list[str]:
    if not historical: return _pending(out, stem, "historical budget metrics not loaded")
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for key in ("cider", "support_rate"):
        pts = [(r["budget"], r[key]) for r in historical if r.get(key) is not None]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=key)
    ax.set_xlabel("retrieval budget"); ax.set_ylabel("metric"); ax.legend()
    return _save(fig, out, stem)


def export_table(rows: list[dict], out: Path, stem: str) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    csv_p = out / f"{stem}.csv"; json_p = out / f"{stem}.json"; md_p = out / f"{stem}.md"
    if not rows:
        (out / f"{stem}.PENDING.txt").write_text("PENDING: no data\n", encoding="utf-8")
        return {"status": "PENDING"}
    with csv_p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    json_p.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    keys = list(rows[0].keys())
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for r in rows: lines.append("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |")
    md_p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"status": "OK", "csv": str(csv_p), "json": str(json_p), "md": str(md_p)}
