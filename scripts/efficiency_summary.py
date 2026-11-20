#!/usr/bin/env python3
"""Aggregate runtime ledger and context-reduction numbers (CPU only)."""
import csv
import json
import os

ROOT = "/home/bn/videoufo_research"
rows = list(csv.DictReader(open(os.path.join(ROOT, "reports/runtime_ledger.csv"))))
out = {"runs": rows, "summary": {}}
by = {}
for r in rows:
    key = f"{r['scale']}:{r['track']}"
    by.setdefault(key, []).append(r)
for key, rs in by.items():
    tot = sum(float(r["gpu_hours"]) for r in rs if r["gpu_hours"])
    out["summary"][key] = {"runs": len(rs), "gpu_hours": round(tot, 3)}


def mean_k(prefix, simfile):
    S = json.load(open(os.path.join(ROOT, simfile)))
    ks = []
    for key in S:
        sims = S[key][1][:7]
        gaps = [sims[i] - sims[i + 1] for i in range(6)]
        i = max(range(6), key=lambda j: gaps[j]) + 1
        ks.append(1 if i <= 1 else 3 if i <= 3 else 5 if i <= 5 else 7)
    return sum(ks) / len(ks), {k: ks.count(k) for k in (1, 3, 5, 7)}


mk_val, dist_val = mean_k("val", "external/external_val_sims.json")
out["adaptive_val497"] = {"mean_k": round(mk_val, 3), "dist": dist_val,
                          "context_reduction_vs_k7": round(1 - mk_val / 7, 4)}
mk_test, dist_test = mean_k("test", "external/external_test_sims.json")
out["adaptive_test2990"] = {"mean_k": round(mk_test, 3), "dist": dist_test,
                            "context_reduction_vs_k7": round(1 - mk_test / 7, 4)}
mk_msvd, dist_msvd = None, None
p = os.path.join(ROOT, "external/external_msvd_sims.json")
if os.path.exists(p):
    mk_msvd, dist_msvd = mean_k("msvd", "external/external_msvd_sims.json")
    out["adaptive_msvd670"] = {"mean_k": round(mk_msvd, 3), "dist": dist_msvd,
                               "context_reduction_vs_k7": round(1 - mk_msvd / 7, 4)}
json.dump(out, open(os.path.join(ROOT, "reports/efficiency_summary.json"), "w"), indent=2)
print(json.dumps({k: v for k, v in out.items() if k != "runs"}, indent=2))
