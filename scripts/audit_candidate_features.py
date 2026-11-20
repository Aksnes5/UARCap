#!/usr/bin/env python3
"""Audit candidate-union feature cache for a retrieval prefix (val scale or track).

Usage: audit_candidate_features.py <prefix> <D|CF>
prefix example: external_val_50k (files external_val_50k_d7_cap.json) or external_test / external_msvd.
"""
import json
import os
import sys

ROOT = "/home/bn/videoufo_research"
EXT = os.path.join(ROOT, "external")
FD = os.path.join(ROOT, "features/videollama_faithful_v2")
PREFIX, TRACK = sys.argv[1], sys.argv[2]
capf = os.path.join(EXT, f"{PREFIX}_{TRACK.lower()}7_cap.json")
cap = json.load(open(capf))
union, per_q = set(), {}
for key, v in cap.items():
    ids = [lst[0] for lst in v[2] if lst]
    per_q[key] = ids
    union |= set(ids)
have = {f[:-3] for f in os.listdir(FD) if f.endswith(".pt")}
missing = sorted(c for c in union if c not in have)
cover = sum(len(v) for v in per_q.values())
out = {"prefix": PREFIX, "track": TRACK, "queries": len(cap),
       "candidate_slots": cover, "unique_clips": len(union),
       "cache_hit": len(union) - len(missing), "missing": len(missing),
       "hit_rate": round((len(union) - len(missing)) / max(1, len(union)), 4)}
json.dump(out, open(os.path.join(ROOT, f"reports/cache_audit_{PREFIX}_{TRACK}.json"), "w"), indent=2)
open(f"/tmp/missing_{PREFIX}_{TRACK}.txt", "w").write("\n".join(missing) + ("\n" if missing else ""))
print(json.dumps(out))
