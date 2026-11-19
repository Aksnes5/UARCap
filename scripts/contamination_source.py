#!/usr/bin/env python3
"""Definitive source-level contamination audit using MSR-VTT YouTube URLs."""
import csv
import json
import os
import re

ROOT = "/home/bn/videoufo_research"
DATA = "/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/lj/datasets/MSRVTT-v2"
UFO = "/home/bn/datasets/VideoUFO/metadata/VideoUFO.csv"


def yt(u):
    m = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{6,})", u or "")
    return m.group(1) if m else None


ms = {}
for fn in ["train_val_videodatainfo.json", "test_videodatainfo.json"]:
    for v in json.load(open(os.path.join(DATA, fn)))["videos"]:
        ms[v["video_id"]] = {"split": v.get("split"), "yt": yt(v.get("url")),
                             "start": v.get("start time"), "end": v.get("end time")}

csv.field_size_limit(10 ** 9)
ufo = {}
with open(UFO, newline="", encoding="utf-8-sig") as f:
    r = csv.reader(f); h = next(r)
    i_id, i_s, i_e = h.index("ID"), h.index("Start Time"), h.index("End Time")
    for row in r:
        if not row:
            continue
        cid = row[i_id]
        src, _, seg = cid.rpartition(".")
        try:
            st, en = float(row[i_s]), float(row[i_e])
        except Exception:
            st = en = None
        ufo.setdefault(src, []).append((cid, st, en))

src_overlap = {s: sorted({v["yt"] for v in ms.values() if v["split"] == s and v["yt"] in ufo})
               for s in ("train", "val", "test")}
same_time = {}
for s in src_overlap:
    hits = []
    for v in ms.values():
        if v["split"] != s or v["yt"] not in ufo:
            continue
        for cid, st, en in ufo[v["yt"]]:
            if st is None or v["start"] is None:
                continue
            ov = max(0.0, min(float(v["end"]), en) - max(float(v["start"]), st))
            if ov >= 3.0:
                hits.append({"msrvtt": v["video_id"], "clip": cid, "overlap_s": round(ov, 1)})
    same_time[s] = hits

# retrieved val candidates
R = json.load(open(os.path.join(ROOT, "external/external_val_reranked.json")))
cand_q = {}
for key, v in R.items():
    q = key.split("/")[-1][:-4]
    for lst in v[2][:7]:
        if lst:
            cand_q.setdefault(lst[0], set()).add(q)
flag = []
for cid, qs in cand_q.items():
    src = cid.rpartition(".")[0]
    for q in qs:
        v = ms.get(q)
        if v and v["yt"] == src:
            flag.append({"msrvtt": q, "clip": cid, "same_source": True,
                         "same_time": any(h["msrvtt"] == q and h["clip"] == cid for h in same_time["val"])})

res = {"msrvtt_videos": len(ms), "ufo_sources": len(ufo),
       "source_overlap": {s: {"n": len(l), "examples": l[:10]} for s, l in src_overlap.items()},
       "same_time_overlap": {s: {"n": len(l), "examples": l[:10]} for s, l in same_time.items()},
       "val_retrieved_same_source": flag}
json.dump(res, open(os.path.join(ROOT, "reports/contamination_source.json"), "w"), indent=2)
print(json.dumps({k: {kk: (vv["n"] if isinstance(vv, dict) and "n" in vv else vv) for kk, vv in v.items()}
                  if isinstance(v, dict) and set(v) <= {"n", "examples"} else v for k, v in res.items()
                  if k != "val_retrieved_same_source"}, indent=2))
print("val retrieved same-source hits:", len(flag), flag[:5])
