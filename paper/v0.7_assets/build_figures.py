"""Create manuscript figures from immutable historical summaries and the completed CPU scorer."""
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
BLUE, TEAL, INK, GRID = "#20577a", "#087f73", "#172c39", "#dbe2e5"
SCALES = ("10k", "50k", "100k", "250k", "500k", "full")
LABELS = ("10K", "50K", "100K", "250K", "500K", "FULL")


def read(path):
    return json.loads(path.read_bytes())


def rows(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def save(fig, name):
    for fmt in ("pdf", "svg", "png"):
        path = OUT / f"{name}.{fmt}"
        metadata = {"CreationDate":None} if fmt=="pdf" else {"Date":None} if fmt=="svg" else None
        fig.savefig(path, dpi=280, facecolor="white", metadata=metadata)
        if fmt=="svg":
            lines = path.read_bytes().splitlines()
            path.write_bytes(b"\n".join(line.rstrip() for line in lines)+b"\n")
    plt.close(fig)


def framework():
    fig, ax = plt.subplots(figsize=(8.1, 3.2), layout="constrained")
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    centers = [.125, .375, .625, .875]
    for i, (x, title, desc) in enumerate(zip(centers,
            ("External corpus", "Candidate access", "Evidence selection", "Caption utility"),
            ("Six nested sizes\n10K to 1.09M clips", "X-CLIP top 20\nclip–caption groups",
             "Order and truncate\ntext and visual inputs", "Fixed captioner\nPTB CIDEr"), strict=True)):
        ax.add_patch(FancyBboxPatch((x-.112,.55),.224,.34,boxstyle="round,pad=.004,rounding_size=.014",
                     facecolor="#eef5f6",edgecolor="#9bb5bf"))
        ax.text(x,.80,title,ha="center",va="center",weight="bold",color=INK)
        ax.text(x,.66,desc,ha="center",va="center")
        if i < 3:
            ax.annotate("", xy=(x+.137,.72),xytext=(x+.114,.72), arrowprops={"arrowstyle":"->","color":INK,"lw":1.1})
    for y, tag, desc, color in ((.36, "A   Net utility", "Keep fusion active. Compare zero, one and seven caption–clip pairs.", BLUE),
                                (.14, "B   Text extension", "Keep three visual tensors fixed. Append four higher- or lower-score captions.", TEAL)):
        ax.text(.02,y,tag,weight="bold",va="center",color=color)
        ax.text(.26,y,desc,va="center",fontsize=9.2)
    save(fig, "fig1_framework")


def access():
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.25), layout="constrained")
    changes = rows(OLD / "top1_churn_downstream.csv")
    effects = {r["test"]:r for r in rows(OLD/"paired_contrasts.csv") if r["split"]=="val497"
               and r["family"]=="exploratory" and r["metric"]=="official_ptb"}
    x = np.arange(6)
    for i,(order,color,marker) in enumerate((("CF",BLUE,"o"),("D",TEAL,"s"))):
        access_rows = {r["scale_vs_10k"]:r for r in changes if r["budget"]=="3" and r["ordering"]==order}
        axes[0].plot(x,[0]+[float(access_rows[s]["selected_xclip_delta"]) for s in SCALES[1:]],marker=marker,color=color,label=order)
        rr = [effects[f"{s}-10k/{order}/k3"] for s in SCALES[1:]]
        d,lo,hi = (np.array([float(r[k]) for r in rr]) for k in ("diff","cluster_lo","cluster_hi"))
        axes[1].errorbar(x[1:]+(.055 if i else -.055),d,[d-lo,hi-d],fmt=marker,color=color,capsize=3,label=order)
    for ax,ttl,ylabel in zip(axes,("a   Retrieval access","b   Caption generation"),
            ("Change in selected top-1 similarity","CIDEr difference (points)"),strict=True):
        ax.set(xticks=x,xticklabels=LABELS,xlabel="Nested corpus (clips)",ylabel=ylabel)
        ax.set_title(ttl,loc="left")
        ax.axhline(0,color="#697c86",lw=.8)
        ax.grid(axis="y",color=GRID,lw=.7)
        ax.legend(frameon=False)
    save(fig,"fig2_access")


def benchmark(summary):
    fig, axes = plt.subplots(1,2,figsize=(7.7,3.0),layout="constrained")
    for ax,key,name in zip(axes,("test2690","msvd670"),("a   MSR-VTT","b   MSVD"),strict=True):
        k = [1,3,5,7]
        y = [summary[key]["means"]["official_ptb"][f"D{i}"] for i in k]
        ax.plot(k,y,"o-",color=BLUE,lw=1.7)
        for xx,yy in zip(k,y,strict=True):
            ax.annotate(f"{yy:.2f}",(xx,yy),xytext=(0,7),textcoords="offset points",ha="center")
        ax.set(xlabel="Joint text–visual budget",ylabel="CIDEr (points)",xticks=k,xlim=(.4,7.6))
        ax.margins(y=.32)
        ax.set_title(name,loc="left")
        ax.grid(axis="y",color=GRID)
    save(fig,"fig3_budget")


def followups():
    fig, axes = plt.subplots(1,2,figsize=(7.7,2.8),layout="constrained")
    for ax,fam,name,color in zip(axes,"AB",("a   Matched zero evidence","b   Fixed visual evidence"),(BLUE,TEAL),strict=True):
        data = read(NEW/f"FOLLOWUP_{fam}_FORMAL_RESULTS.json")["CIDEr_points"]
        for i,(arm,y) in enumerate(data.items()):
            ax.plot(i,y,"o",ms=7,color=color)
            ax.annotate(f"{y:.2f}",(i,y),xytext=(0,8),textcoords="offset points",ha="center")
        ax.set(xticks=range(3),xticklabels=list(data),xlim=(-.5,2.5),ylim=(67.8,72),ylabel="Cohort CIDEr (points)")
        ax.set_title(name,loc="left")
        ax.grid(axis="y",color=GRID)
    save(fig,"fig4_followups")

    tests=rows(NEW/"FOLLOWUP_FIVE_TEST_HOLM_RESULTS.csv")
    fig,(ax,tab)=plt.subplots(1,2,figsize=(8,3.45),width_ratios=[2.5,1.2],layout="constrained")
    ax.axvspan(-1,1,color="#edf1f3",zorder=-2)
    for v in (-1,1):
        ax.axvline(v,color="#82939d",lw=.8,ls="--")
    ax.axvline(0,color=INK,lw=.9)
    ys=[4.4,3.4,2,1,0]
    for y,r in zip(ys,tests,strict=True):
        d,lo,hi=map(float,(r["effect"],r["ci_low"],r["ci_high"]))
        c=BLUE if r["family"]=="A" else TEAL
        ax.errorbar(d,y,xerr=[[d-lo],[hi-d]],fmt="o",color=c,ms=5,capsize=3)
        for x,key in ((.26,"p_family_holm"),(.77,"p_five_holm")):
            tab.text(x,y,f"{float(r[key]):.4f}",ha="center",va="center",color=INK)
    ax.set(yticks=ys,yticklabels=[r["contrast"].replace("-"," − ") for r in tests],ylim=(-.6,5),xlim=(-3.3,2.8),
           xlabel="Paired CIDEr difference (points)")
    ax.set_title("Effect and pointwise 95% source-cluster CI",loc="left")
    tab.set(xlim=(0,1),ylim=ax.get_ylim())
    tab.set_title("Holm-adjusted p")
    for x,s in ((.26,"Family"),(.77,"All five")):
        tab.text(x,4.96,s,ha="center",va="center",fontsize=8,color=INK,weight="bold")
    tab.axis("off")
    save(fig,"fig5_forest")


if __name__=="__main__":
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.titlesize":10,"axes.titleweight":"bold",
                         "axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42,"svg.fonttype":"none",
                         "axes.labelcolor":INK,"text.color":INK,"svg.hashsalt":"uarcap-v07"})
    framework()
    access()
    benchmark(read(OLD/"summary.json"))
    followups()
