#!/usr/bin/env python3
"""S5: candidate survival/churn across nested scale corpora, linked to D3 generation."""
import json
import os
import sys

ROOT = "/home/bn/videoufo_research"
sys.path.insert(0, os.path.join(ROOT, "evcap_external/evaluation"))
from pycocoevalcap.cider.cider import Cider  # noqa: E402

SCALES = ["10k", "50k", "100k", "250k", "500k", "full"]
FILES = {s: os.path.join(ROOT, f"external/external_val_{s}_sims.json") for s in SCALES[:-1]}
FILES["full"] = os.path.join(ROOT, "external/external_val_sims.json")
S = {s: json.load(open(FILES[s])) for s in SCALES}
Q = sorted(S["full"])


def top(s, q, k):
    return S[s][q][0][:k]


def rank_of(s, q):
    return {c: i for i, c in enumerate(S[s][q][0])}


GT = {}
for line in open("/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/lj/datasets/MSRVTT-v2/val.caption.tsv"):
    p = line.split("\t", 1)
    GT[p[0].split("/")[-1][:-4]] = [x["caption"] for x in json.loads(p[1])]
out = {}
pairs = list(zip(SCALES, SCALES[1:]))
for a, b in pairs:
    surv = {k: 0 for k in (1, 3, 7, 20)}
    jac = {k: 0.0 for k in (3, 7, 20)}
    disp, new, src_new = [], [], []
    for q in Q:
        for k in surv:
            surv[k] += len(set(top(a, q, k)) & set(top(b, q, k))) / len(top(a, q, k))
        for k in jac:
            A, B = set(top(a, q, k)), set(top(b, q, k))
            jac[k] += len(A & B) / len(A | B)
        ra, rb = rank_of(a, q), rank_of(b, q)
        common = [c for c in top(a, q, 20) if c in rb]
        if common:
            disp.append(sum(abs(ra[c] - rb[c]) for c in common) / len(common))
        new.append(len(set(top(b, q, 7)) - set(top(a, q, 7))) / 7)
        sa = {c.rsplit('.', 1)[0] for c in top(a, q, 7)}
        sb = {c.rsplit('.', 1)[0] for c in top(b, q, 7)}
        src_new.append(len(sb - sa) / 7)
    out[f"{a}->{b}"] = {
        "survival": {k: round(v / len(Q), 4) for k, v in surv.items()},
        "jaccard": {k: round(v / len(Q), 4) for k, v in jac.items()},
        "mean_rank_displacement_top20": round(sum(disp) / len(disp), 4),
        "new_candidate_rate_top7": round(sum(new) / len(new), 4),
        "new_source_rate_top7": round(sum(src_new) / len(src_new), 4),
    }

# link churn to D3 generation change
NAME = "eval_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review_generated_captions.json"


def caps(s):
    d = f"eval_val497_{s}_D3" if s != "full" else "eval_val497_D3"
    return {x["image_name"]: x["prediction"].strip() for x in json.load(open(os.path.join(ROOT, "evcap_external/outputs", d, NAME)))}


C = {s: caps(s) for s in SCALES}
g = {q: GT[q.split("/")[-1][:-4]] for q in Q}
cid = {s: dict(zip(Q, Cider().compute_score(g, {q: [C[s][q]] for q in Q})[1])) for s in SCALES}
for a, b in pairs:
    changed = sum(C[a][q] != C[b][q] for q in Q) / len(Q)
    delta = sum(cid[b][q] - cid[a][q] for q in Q) / len(Q)
    out[f"{a}->{b}"]["d3_output_change_rate"] = round(changed, 4)
    out[f"{a}->{b}"]["d3_cider_delta"] = round(delta, 4)
json.dump(out, open(os.path.join(ROOT, "reports/nested_scale_evidence_survival.json"), "w"), indent=2)
for k, v in out.items():
    print(k, v)
