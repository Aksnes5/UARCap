#!/usr/bin/env python3
"""Build val497 CF7/D7 cap+sim JSONs for a manifest scale.

Usage: build_val_grid_scale.py <scale>
Reads external/external_val_<scale>_{sims,top20,reranked}.json and writes
external/external_val_<scale>_{cf7,d7}_{cap,sims}.json + <scale>_grid_clips.txt.
"""
import json
import os
import sys

ROOT = "/home/bn/videoufo_research"
EXT = os.path.join(ROOT, "external")
SCALE = sys.argv[1]
S = json.load(open(os.path.join(EXT, f"external_val_{SCALE}_sims.json")))
D = json.load(open(os.path.join(EXT, f"external_val_{SCALE}_top20.json")))
R = json.load(open(os.path.join(EXT, f"external_val_{SCALE}_reranked.json")))

cf, cfs, dd, ds = {}, {}, {}, {}
for key, (caps, rels) in S.items():
    cf[key] = [caps[:7], rels[:7], [D[key][j]["clip_ids"] for j in range(7)]]
    cfs[key] = [caps[:7], rels[:7]]
    rc, rq, rl = R[key][0][:7], R[key][1][:7], R[key][2][:7]
    dd[key] = [rc, rq, rl]
    m = {c: s for c, s in zip(caps, rels)}
    ds[key] = [rc, [m.get(c, 0.0) for c in rc]]

for name, cap, sim in [("cf7", cf, cfs), ("d7", dd, ds)]:
    json.dump(cap, open(os.path.join(EXT, f"external_val_{SCALE}_{name}_cap.json"), "w"), indent=1)
    json.dump(sim, open(os.path.join(EXT, f"external_val_{SCALE}_{name}_sims.json"), "w"), indent=1)

sel = set()
for cap in (cf, dd):
    for v in cap.values():
        for lst in v[2]:
            if lst:
                sel.add(lst[0])
open(os.path.join(EXT, f"external_val_{SCALE}_grid_clips.txt"), "w").write("\n".join(sorted(sel)) + "\n")
print(SCALE, "queries", len(cf), "selected clips", len(sel))
