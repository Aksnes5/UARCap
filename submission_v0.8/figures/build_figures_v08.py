"""Generate publication-grade v0.8 manuscript figures from immutable statistical artifacts.

All numerical values are read directly from:
  - reports/local_independent_review/results/
  - reports/v07_final/
No experimental numbers are hardcoded or altered.
"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
OLD = ROOT / "reports/local_independent_review/results"
NEW = ROOT / "reports/v07_final"

# Colorblind-safe (Okabe-Ito inspired) & grayscale-distinguishable palette
NAVY = "#1B4F72"      # Primary / Experiment A / CF (dark in grayscale)
VERMILION = "#C04D00" # Contrast / Experiment B / D (mid-dark in grayscale)
TEAL = "#0E7C66"      # Secondary accent
INK = "#17202A"       # Text and axes
MUTED = "#5D6D7E"     # Reference lines / annotations
GRID = "#E2E8EC"      # Subtle horizontal grid
BAND = "#EDF2F5"      # Practical-margin shaded band
BOX_FILL = "#F2F6F8"  # Pipeline stage fill
BOX_EDGE = "#8FA6B5"  # Pipeline stage border

SCALES = ("10k", "50k", "100k", "250k", "500k", "full")
LABELS = ("10K", "50K", "100K", "250K", "500K", "FULL")


def read_json(path: Path):
    return json.loads(path.read_bytes())


def read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def save_fig(fig, name: str):
    for fmt in ("pdf", "svg", "png"):
        path = OUT / f"{name}.{fmt}"
        metadata = {"CreationDate": None} if fmt == "pdf" else {"Date": None} if fmt == "svg" else None
        fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.08, metadata=metadata)
        if fmt == "svg":
            lines = path.read_bytes().splitlines()
            path.write_bytes(b"\n".join(line.rstrip() for line in lines) + b"\n")
    plt.close(fig)


def build_fig1_framework():
    fig, ax = plt.subplots(figsize=(8.8, 3.75))
    ax.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    ax.axis("off")

    # Top header tag
    ax.text(0.01, 0.955, "Frozen video-captioning interface (Fang et al., 2026)  |  Controlled external-retrieval variables",
            fontsize=8.5, color=MUTED, weight="bold", va="center")

    # Four pipeline stages
    centers = [0.122, 0.374, 0.626, 0.878]
    stages = [
        ("1. External Corpus", "VideoUFO nested scales\n10K to 1.09M clips\n(345,437 source videos)", "#EBF3F7", NAVY),
        ("2. Candidate Access", "Video-conditioned X-CLIP\nTop-20 caption groups\n(exact cosine similarity)", "#EBF3F7", NAVY),
        ("3. Evidence Selection", "CF (raw) vs. D (consensus)\nPrefix budget k in {1,3,5,7}\nText + matched clip features", "#FDF3EB", VERMILION),
        ("4. Caption Utility", "Frozen cross-gating fusion\n+ Vicuna-7B-v1.3 decoder\nOfficial PTB CIDEr", "#F2F4F4", INK),
    ]
    w, h, y0 = 0.214, 0.33, 0.565
    for i, (cx, (title, desc, fill, edge)) in enumerate(zip(centers, stages, strict=True)):
        ax.add_patch(FancyBboxPatch(
            (cx - w / 2, y0), w, h,
            boxstyle="round,pad=0.008,rounding_size=0.018",
            facecolor=fill, edgecolor=edge, linewidth=1.25
        ))
        ax.text(cx, y0 + h - 0.055, title, ha="center", va="center", fontsize=8.3, weight="bold", color=INK)
        ax.text(cx, y0 + 0.115, desc, ha="center", va="center", fontsize=7.3, color=INK, linespacing=1.34)
        if i < 3:
            ax.annotate(
                "", xy=(centers[i + 1] - w / 2 - 0.011, y0 + h / 2),
                xytext=(cx + w / 2 + 0.011, y0 + h / 2),
                arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1.2, "mutation_scale": 10}
            )

    # Subtle divider line
    ax.plot([0.012, 0.988], [0.495, 0.495], color=GRID, lw=1.0, ls="--")
    ax.text(0.012, 0.455, "Prespecified controlled follow-up interventions on MSR-VTT (FULL external memory)",
            fontsize=8.6, color=INK, weight="bold", va="center")

    # Two intervention cards
    cards = [
        (0.245, "Experiment A: Matched Net Utility (n = 2,690; 2,004 sources)",
         "Retain full fusion pathway & prompt scaffold at zero external input (A0).\n"
         "Compare A0 (0/0) vs. A1 (1/1) and A7 (7/7) D-ordered caption-clip pairs.",
         NAVY, "#EFF5F9", "-"),
        (0.035, "Experiment B: Fixed-Visual Text Extension (n = 2,618; 1,969 sources)",
         "Verify per-query identity of all three retrieved visual tensors & fused projections (B3).\n"
         "Append four higher- (B7HQ) or lower-similarity (B7LQ) text-only captions (length- & overlap-matched).",
         VERMILION, "#FCF4EE", "--"),
    ]
    for cy, head, body, edge, fill, ls in cards:
        ax.add_patch(FancyBboxPatch(
            (0.015, cy), 0.97, 0.17,
            boxstyle="round,pad=0.006,rounding_size=0.015",
            facecolor=fill, edgecolor=edge, linewidth=1.2, linestyle=ls
        ))
        ax.text(0.032, cy + 0.125, head, fontsize=8.4, weight="bold", color=edge, va="center")
        ax.text(0.032, cy + 0.052, body, fontsize=7.8, color=INK, va="center", linespacing=1.28)

    # Connectors from Stage 3 down to intervention strip
    ax.annotate("", xy=(0.627, 0.485), xytext=(0.627, y0 - 0.010),
                arrowprops={"arrowstyle": "-|>", "color": VERMILION, "lw": 1.15, "mutation_scale": 10})

    save_fig(fig, "fig1_framework")


def build_fig2_access():
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.35), layout="constrained")
    changes = read_csv(OLD / "top1_churn_downstream.csv")
    effects = {
        r["test"]: r for r in read_csv(OLD / "paired_contrasts.csv")
        if r["split"] == "val497" and r["family"] == "exploratory" and r["metric"] == "official_ptb"
    }
    x = np.arange(len(SCALES))

    styles = (
        ("CF (raw similarity)", "CF", NAVY, "o", "-", "full"),
        ("D (consensus reranking)", "D", VERMILION, "s", "--", "none"),
    )
    for i, (label, order, color, marker, ls, fillstyle) in enumerate(styles):
        access_rows = {r["scale_vs_10k"]: r for r in changes if r["budget"] == "3" and r["ordering"] == order}
        sim_deltas = [0.0] + [float(access_rows[s]["selected_xclip_delta"]) for s in SCALES[1:]]
        axes[0].plot(
            x, sim_deltas, marker=marker, color=color, linestyle=ls, lw=1.65, ms=5.8,
            markerfacecolor=color if fillstyle == "full" else "white", markeredgewidth=1.4, label=label
        )
        rr = [effects[f"{s}-10k/{order}/k3"] for s in SCALES[1:]]
        d = np.array([float(r["diff"]) for r in rr])
        lo = np.array([float(r["cluster_lo"]) for r in rr])
        hi = np.array([float(r["cluster_hi"]) for r in rr])
        offset = 0.09 if i else -0.09
        axes[1].errorbar(
            x[1:] + offset, d, yerr=[d - lo, hi - d],
            fmt=marker, color=color, ms=5.8, lw=1.35, capsize=3.2,
            markerfacecolor=color if fillstyle == "full" else "white", markeredgewidth=1.4, label=label
        )

    axes[0].set(
        xticks=x, xticklabels=LABELS,
        xlabel="Nested external corpus size (clips)",
        ylabel="Change in selected top-1 X-CLIP similarity",
        ylim=(-0.0025, 0.036)
    )
    axes[0].set_title("a  Retrieval access (relative to 10K)", loc="left", pad=8)

    axes[1].set(
        xticks=x, xticklabels=LABELS,
        xlabel="Nested external corpus size (clips)",
        ylabel="Paired CIDEr change vs. 10K (points)",
        xlim=(-0.45, 5.45), ylim=(-4.5, 5.5)
    )
    axes[1].set_title("b  Generation utility at k = 3 (95% source-cluster CI)", loc="left", pad=8)
    axes[1].plot(0.0, 0.0, "d", color=MUTED, ms=4.5)
    axes[1].text(0.0, 0.35, "Ref.", ha="center", va="bottom", fontsize=7.5, color=MUTED)

    for ax in axes:
        ax.axhline(0.0, color=MUTED, lw=0.85, ls=":")
        ax.grid(axis="y", color=GRID, lw=0.75)
        ax.legend(frameon=False, loc="upper left", fontsize=8.0)

    save_fig(fig, "fig2_access")


def build_fig3_budget(summary):
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.25), layout="constrained")
    specs = (
        ("test2690", "a  MSR-VTT (n = 2,690; trained at k = 7)", 7, (67.4, 72.3), [(0, 8), (0, 8), (0, 8), (-4, -16)]),
        ("msvd670", "b  MSVD (n = 670; trained at k = 3)", 3, (138.0, 160.5), [(0, 8), (18, 7), (12, 7), (0, 9)]),
    )
    ks = [1, 3, 5, 7]
    for ax, (key, title, train_k, ylim, offsets) in zip(axes, specs, strict=True):
        means = summary[key]["means"]["official_ptb"]
        ys = [means[f"D{k}"] for k in ks]
        adapt = means["Adaptive"]

        # Shade extrapolation region on MSVD (k > 3)
        if train_k == 3:
            ax.axvspan(3.0, 7.6, color="#F7EFEA", alpha=0.75, zorder=-2)
            ax.text(5.3, ylim[1] - 1.35, "Beyond MSVD\ntraining budget", ha="center", va="top",
                    fontsize=7.5, color=VERMILION, style="italic")

        ax.axvline(train_k, color=MUTED, lw=0.95, ls="--", zorder=-1)
        ax.plot(ks, ys, "o-", color=NAVY, lw=1.75, ms=6.2, label="Fixed external budget (D)")
        ax.axhline(adapt, color=TEAL, lw=1.1, ls=":", label=f"Adaptive baseline ({adapt:.2f})")

        for xx, yy, (dx, dy) in zip(ks, ys, offsets, strict=True):
            ax.annotate(f"{yy:.2f}", (xx, yy), xytext=(dx, dy), textcoords="offset points",
                        ha="center", fontsize=8.2, color=INK, weight="bold")

        ax.set(
            xlabel="Joint text–visual budget k",
            ylabel="Official PTB CIDEr (points)",
            xticks=ks, xlim=(0.45, 7.55), ylim=ylim
        )
        ax.set_title(title, loc="left", pad=8)
        ax.grid(axis="y", color=GRID, lw=0.75)
        ax.legend(frameon=False, loc="lower left", fontsize=7.8)

    save_fig(fig, "fig3_budget")


def build_fig4_followups():
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.1), layout="constrained")
    panels = (
        ("A", "a  Experiment A: Matched zero evidence\n(n = 2,690; 2,004 sources)", NAVY, "o", "full", "A0"),
        ("B", "b  Experiment B: Fixed visual evidence\n(n = 2,618; 1,969 sources)", VERMILION, "s", "none", "B3"),
    )
    for ax, (fam, title, color, marker, fillstyle, ref_arm) in zip(axes, panels, strict=True):
        data = read_json(NEW / f"FOLLOWUP_{fam}_FORMAL_RESULTS.json")["CIDEr_points"]
        arms = list(data.keys())
        vals = [data[a] for a in arms]
        ref_val = data[ref_arm]

        ax.axhline(ref_val, color=MUTED, lw=0.95, ls="--", zorder=-1)
        for i, (arm, y) in enumerate(zip(arms, vals, strict=True)):
            ax.plot(
                i, y, marker=marker, ms=7.2, color=color,
                markerfacecolor=color if fillstyle == "full" else "white", markeredgewidth=1.6
            )
            ax.annotate(f"{y:.2f}", (i, y), xytext=(0, 9), textcoords="offset points",
                        ha="center", fontsize=8.4, weight="bold", color=INK)

        ax.set(
            xticks=range(len(arms)), xticklabels=arms,
            xlim=(-0.55, 2.55), ylim=(67.5, 72.1),
            ylabel="Cohort PTB CIDEr (points)"
        )
        ax.set_title(title, loc="left", pad=8, fontsize=9.2)
        ax.grid(axis="y", color=GRID, lw=0.75)

    save_fig(fig, "fig4_followups")


def build_fig5_forest():
    tests = read_csv(NEW / "FOLLOWUP_FIVE_TEST_HOLM_RESULTS.csv")
    fig, (ax, tab) = plt.subplots(1, 2, figsize=(8.4, 3.85), width_ratios=[2.35, 1.45], layout="constrained")

    # Practical margin band [-1, +1]
    ax.axvspan(-1.0, 1.0, color=BAND, zorder=-2, label="±1.0 pt practical margin")
    for v in (-1.0, 1.0):
        ax.axvline(v, color="#8596A2", lw=0.85, ls="--", zorder=-1)
    ax.axvline(0.0, color=INK, lw=0.95, zorder=-1)

    ys = [4.5, 3.5, 2.1, 1.1, 0.1]
    ax.axhline(2.8, color=GRID, lw=1.0, ls="-")
    tab.axhline(2.8, color=GRID, lw=1.0, ls="-")

    plotted_a = plotted_b = False
    for y, r in zip(ys, tests, strict=True):
        d = float(r["effect"])
        lo = float(r["ci_low"])
        hi = float(r["ci_high"])
        is_a = r["family"] == "A"
        color = NAVY if is_a else VERMILION
        marker = "o" if is_a else "s"
        mfc = color if is_a else "white"
        lbl = None
        if is_a and not plotted_a:
            lbl = "Experiment A (n = 2,690)"
            plotted_a = True
        elif (not is_a) and not plotted_b:
            lbl = "Experiment B (n = 2,618)"
            plotted_b = True

        ax.errorbar(
            d, y, xerr=[[d - lo], [hi - d]],
            fmt=marker, color=color, markerfacecolor=mfc, markeredgewidth=1.4,
            ms=6.0, lw=1.45, capsize=3.5, label=lbl
        )
        tab.text(0.22, y, f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]", ha="center", va="center", fontsize=7.6, color=INK)
        tab.text(0.67, y, f"{float(r['p_family_holm']):.4f}", ha="center", va="center", fontsize=8.0,
                 color=INK, weight="bold" if float(r["p_family_holm"]) < 0.05 else "normal")
        tab.text(0.91, y, f"{float(r['p_five_holm']):.4f}", ha="center", va="center", fontsize=8.0,
                 color=INK, weight="bold" if float(r["p_five_holm"]) < 0.05 else "normal")

    ax.set(
        yticks=ys,
        yticklabels=[r["contrast"].replace("-", " − ") for r in tests],
        ylim=(-0.55, 5.45), xlim=(-3.35, 2.85),
        xlabel="Paired CIDEr difference (points)"
    )
    ax.set_title("Effect and pointwise 95% source-cluster CI", loc="left", pad=8)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.48, -0.19), fontsize=7.8, ncol=3, columnspacing=1.2)

    tab.set(xlim=(0.0, 1.04), ylim=ax.get_ylim())
    tab.axis("off")
    tab.set_title("Estimate and Holm-adjusted p", loc="center", pad=8)
    for xh, s in ((0.22, "Effect [95% CI]"), (0.67, "Family"), (0.91, "All five")):
        tab.text(xh, 5.18, s, ha="center", va="center", fontsize=7.8, weight="bold", color=INK)
    tab.plot([0.02, 1.02], [4.95, 4.95], color=MUTED, lw=0.8)

    save_fig(fig, "fig5_forest")


if __name__ == "__main__":
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.8,
        "axes.titlesize": 9.6,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.labelcolor": INK,
        "text.color": INK,
        "svg.hashsalt": "uarcap-v08",
    })
    build_fig1_framework()
    build_fig2_access()
    build_fig3_budget(read_json(OLD / "summary.json"))
    build_fig4_followups()
    build_fig5_forest()
    print("Generated all 5 v0.8 figures in", OUT)
