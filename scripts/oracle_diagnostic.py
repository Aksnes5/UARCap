#!/usr/bin/env python3
"""CPU-only top-20 Oracle Retrieval Diagnostic (diagnostic only, not inference)."""
import json
import os
import sys
from statistics import median

ROOT = "/home/bn/videoufo_research"
EVAL = os.path.join(ROOT, "evcap_external/evaluation")
sys.path.insert(0, EVAL)
from pycocoevalcap.cider.cider import Cider  # noqa: E402

QIDS = [l.strip() for l in open(os.path.join(ROOT, "dev300_queries.txt")) if l.strip()]
GT = {}
for line in open(os.path.join(ROOT, "evcap_external/dev300/dev300.caption.tsv")):
    p = line.split("\t", 1)
    GT[p[0].split("/")[-1][:-4]] = [x["caption"] for x in json.loads(p[1])]


def score_candidates(cands_by_q):
    """Global-df per-candidate CIDEr; key = query::rank."""
    gts, res = {}, {}
    for q, caps in cands_by_q.items():
        for i, c in enumerate(caps):
            k = f"{q}::{i}"
            gts[k] = GT[q]
            res[k] = [c]
    scores = Cider().compute_score(gts, res)[1]
    return dict(zip(gts.keys(), scores))


def run(level):
    S = json.load(open(os.path.join(ROOT, f"external/external_{level}_sims.json")))
    R = json.load(open(os.path.join(ROOT, f"external/external_{level}_reranked.json")))
    cands = {q: S[f"datasets/MSRVTT-v2/videos/{q}.mp4"][0] for q in QIDS}
    sc = score_candidates(cands)
    rows = []
    for q in QIDS:
        caps = cands[q]
        vals = [sc[f"{q}::{i}"] for i in range(len(caps))]
        rk = R[f"datasets/MSRVTT-v2/videos/{q}.mp4"]
        rvals = [sc[f"{q}::{i}"] for i in range(len(caps)) if caps[i] in rk[0]]
        rvals = [v for v in rvals]
        oracle = max(vals)
        oi = vals.index(oracle)
        x7 = max(vals[:7])
        r7 = max([sc[f"{q}::{i}"] for i in range(len(caps)) if caps[i] in rk[0]] or [0])
        rows.append({
            "q": q, "xclip_top1": vals[0], "rerank_top1": sc.get(f"{q}::{caps.index(rk[0][0])}", 0) if rk[0] else 0,
            "oracle20": oracle, "oracle_rank": oi,
            "oracle_in_rerank7": caps[oi] in rk[0],
            "xclip_top7_oracle": x7, "rerank_top7_oracle": r7,
        })
    n = len(rows)
    def mean(k): return sum(r[k] for r in rows) / n
    ranks = [r["oracle_rank"] for r in rows]
    hit = lambda k: sum(1 for x in ranks if x < k) / n
    move_in = sum(1 for r in rows if r["oracle_in_rerank7"] and r["oracle_rank"] >= 7) / n
    move_out = sum(1 for r in rows if (not r["oracle_in_rerank7"]) and r["oracle_rank"] < 7) / n
    out = {
        "level": level, "queries": n,
        "mean_xclip_top1": round(mean("xclip_top1"), 4),
        "mean_rerank_top1": round(mean("rerank_top1"), 4),
        "mean_oracle20": round(mean("oracle20"), 4),
        "mean_xclip_top7_oracle": round(mean("xclip_top7_oracle"), 4),
        "mean_rerank_top7_oracle": round(mean("rerank_top7_oracle"), 4),
        "oracle_rank_mean": round(sum(ranks) / n, 2),
        "oracle_rank_median": median(ranks),
        "hit1": round(hit(1), 4), "hit5": round(hit(5), 4),
        "hit7": round(hit(7), 4), "hit20": round(hit(20), 4),
        "rerank_moves_oracle_in": round(move_in, 4),
        "rerank_moves_oracle_out": round(move_out, 4),
    }
    print(json.dumps(out, indent=2), flush=True)
    return out


if __name__ == "__main__":
    allout = {}
    for lv in ["10k", "100k", "500k", "full"]:
        allout[lv] = run(lv)
    json.dump(allout, open(os.path.join(ROOT, "reports/oracle_diagnostic.json"), "w"), indent=2)
