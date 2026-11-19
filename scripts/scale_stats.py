#!/usr/bin/env python3
"""Retrieval-side statistics for one corpus scale."""
import json
import sys
from collections import Counter

ROOT = "/home/bn/videoufo_research"


def main():
    level = sys.argv[1]
    sims = json.load(open(f"{ROOT}/external/external_{level}_sims.json"))
    diag = json.load(open(f"{ROOT}/external/external_{level}_top20.json"))
    rer = json.load(open(f"{ROOT}/external/external_{level}_reranked.json"))
    n = len(sims)
    top1 = [v[1][0] for v in sims.values()]
    top20 = [sum(v[1]) / len(v[1]) for v in sims.values()]
    usrc, rep, utop = [], 0, []
    for k, v in diag.items():
        first = [x["clip_ids"][0] for x in v]
        srcs = [c.rsplit(".", 1)[0] for c in first]
        c = Counter(srcs)
        usrc.append(len(c))
        rep += int(max(c.values()) > 1)
        utop.append(len({t for x in v for t in x["topics"]}))
    all_clips_20 = {c for v in diag.values() for x in v for c in x["clip_ids"]}
    all_first_7 = {lst[0] for v in rer.values() for lst in v[2]}
    out = {
        "level": level,
        "queries": n,
        "mean_top1_sim": round(sum(top1) / n, 4),
        "mean_top20_sim": round(sum(top20) / n, 4),
        "mean_unique_sources_top20": round(sum(usrc) / n, 2),
        "same_source_query_ratio": round(rep / n, 4),
        "mean_unique_topics_top20": round(sum(utop) / n, 2),
        "top20_unique_clips": len(all_clips_20),
        "top7_unique_clips": len(all_first_7),
    }
    json.dump(out, open(f"{ROOT}/reports/scale_stats_{level}.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
