"""Matched external vs in-domain source x budget interaction on val497 (CPU only)."""
import json, os, sys, math, random, statistics, re
import numpy as np
sys.path.insert(0,"/home/bn/videoufo_research/scripts/cpu_audit")
sys.path.insert(0,"/home/bn/videoufo_research/evcap_external/evaluation")
import cider_fixed as CF
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

ROOT="/home/bn/videoufo_research"; OUT=os.path.join(ROOT,"evcap_external/outputs")
NAME="eval_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review_generated_captions.json"
DATA="/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/lj/datasets/MSRVTT-v2"

def load(d): return json.load(open(os.path.join(OUT,d,NAME)))
V={d:load(d) for d in ["eval_val497_D3","eval_val497_D7","eval_val497_indomain_D3","eval_val497_indomain_D7"]}
sets=[{r["image_name"] for r in v} for v in V.values()]
Q=sorted(set.intersection(*sets))
print("queries per condition",[len(s) for s in sets],"intersection",len(Q))
refs={r["image_name"]:r["captions"] for r in V["eval_val497_D3"]}
gt_all={q:refs[q] for q in Q}
tok=PTBTokenizer()
gt=tok.tokenize({q:[{"caption":c} for c in gt_all[q]] for q in Q})
pred={d:{r["image_name"]:r["prediction"].strip() for r in v} for d,v in V.items()}
C={}
for d in pred:
    hp=tok.tokenize({q:[{"caption":pred[d][q]}] for q in Q}); hp={q:v[0] for q,v in hp.items()}
    C[d]=CF.corpus(gt,hp)

def yt(u):
    m=re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{6,})",u or ""); return m.group(1) if m else None
vd=json.load(open(os.path.join(DATA,"train_val_videodatainfo.json")))
src={v["video_id"]: (yt(v.get("url")) or v["video_id"]) for v in vd["videos"]}
grp={q: src.get(q.split("/")[-1][:-4], q) for q in Q}
print("n sources",len(set(grp.values())))

dE={q:C["eval_val497_D3"][q]-C["eval_val497_D7"][q] for q in Q}
dI={q:C["eval_val497_indomain_D3"][q]-C["eval_val497_indomain_D7"][q] for q in Q}
inter={q:dE[q]-dI[q] for q in Q}

def mean(d): return sum(d.values())/len(d)
def boot(d,n=10000,seed=42):
    rng=random.Random(seed); ks=list(d); out=[]
    for _ in range(n): out.append(sum(d[ks[rng.randrange(len(ks))]] for _ in range(len(ks)))/len(ks))
    out.sort(); return [out[250],out[9750]]
def signflip(d,n=20000,seed=42):
    rng=random.Random(seed); obs=mean(d); m=len(d); c=0
    for _ in range(n):
        s=sum(v if rng.random()<.5 else -v for v in d.values())/m
        if abs(s)>=abs(obs): c+=1
    return (c+1)/(n+1)
def cluster_boot(d,grp,n=10000,seed=42):
    rng=random.Random(seed); by={}
    for q,v in d.items(): by.setdefault(grp[q],[]).append(v)
    ss=list(by); out=[]
    for _ in range(n):
        vals=[]
        for _ in range(len(ss)):
            vals += by[ss[rng.randrange(len(ss))]]
        out.append(sum(vals)/len(vals))
    out.sort(); return [out[250],out[9750]]

res={"n_queries":len(Q),"n_sources":len(set(grp.values())),
     "delta_external":dict(mean=mean(dE),ci95=boot(dE),p_signflip=signflip(dE),
                            median=statistics.median(dE.values()),frac_positive=sum(v>0 for v in dE.values())/len(dE)),
     "delta_indomain":dict(mean=mean(dI),ci95=boot(dI),p_signflip=signflip(dI),
                            median=statistics.median(dI.values()),frac_positive=sum(v>0 for v in dI.values())/len(dI)),
     "interaction":dict(mean=mean(inter),ci95=boot(inter),p_signflip=signflip(inter),
                        median=statistics.median(inter.values()),
                        frac_positive=sum(v>0 for v in inter.values())/len(inter),
                        cluster_ci95=cluster_boot(inter,grp)),
     "means":{ "external_D3":mean({q:C['eval_val497_D3'][q] for q in Q}),
               "external_D7":mean({q:C['eval_val497_D7'][q] for q in Q}),
               "indomain_D3":mean({q:C['eval_val497_indomain_D3'][q] for q in Q}),
               "indomain_D7":mean({q:C['eval_val497_indomain_D7'][q] for q in Q})}}
json.dump(res,open(os.path.join(ROOT,"reports/source_budget_interaction.json"),"w"),indent=2)
print(json.dumps(res,indent=2))

# per-query CSV
import csv
with open(os.path.join(ROOT,"reports/source_budget_interaction_per_query.csv"),"w",newline="") as f:
    w=csv.writer(f); w.writerow(["image_name","source","c_ext_D3","c_ext_D7","c_in_D3","c_in_D7","delta_external","delta_indomain","interaction"])
    for q in Q:
        w.writerow([q,grp[q],C["eval_val497_D3"][q],C["eval_val497_D7"][q],
                    C["eval_val497_indomain_D3"][q],C["eval_val497_indomain_D7"][q],dE[q],dI[q],inter[q]])
