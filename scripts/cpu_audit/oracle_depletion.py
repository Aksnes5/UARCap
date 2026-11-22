"""Oracle candidate-pool depletion audit (CPU only, frozen scorer)."""
import json, os, sys, math, random, statistics, re
from collections import defaultdict
import numpy as np
sys.path.insert(0,"/home/bn/videoufo_research/scripts/cpu_audit")
sys.path.insert(0,"/home/bn/videoufo_research/evcap_external/evaluation")
import cider_fixed as CF
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

ROOT="/home/bn/videoufo_research"; E=os.path.join(ROOT,"external")
DATA="/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/lj/datasets/MSRVTT-v2"

GT={}
for line in open(os.path.join(DATA,"val.caption.tsv")):
    p=line.rstrip("\n").split("\t",1); GT[p[0].split("/")[-1][:-4]]=[x["caption"] for x in json.loads(p[1])]
RAW=json.load(open(os.path.join(E,"external_val_sims.json")))

def tokenize(texts): return [re.findall(r"[a-z0-9']+", t.lower()) for t in texts]
refs_docs=[]; refs_tok={}
for q,refs in GT.items():
    tk=tokenize(refs); refs_tok[q]=tk; refs_docs.extend(tk)
N=len(refs_docs); df={}
for d in refs_docs:
    for w in set(d): df[w]=df.get(w,0)+1
idf={w:math.log(N/c) for w,c in df.items()}
def vec(tokens):
    tf={}
    for w in tokens: tf[w]=tf.get(w,0)+1
    v={w:c*idf.get(w,math.log(N)) for w,c in tf.items()}
    n=math.sqrt(sum(x*x for x in v.values())) or 1.0
    return {w:x/n for w,x in v.items()}
def score(caption,q):
    cv=vec(tokenize([caption])[0]); out=[]
    for rt in refs_tok[q]:
        rv=vec(rt); out.append(sum(v*rv.get(w,0.0) for w,v in cv.items()))
    return float(np.mean(out)) if out else 0.0

def boot_ci(d, n=10000, seed=42):
    rng=random.Random(seed); d=list(d); m=len(d); out=[]
    for _ in range(n): out.append(sum(d[rng.randrange(m)] for _ in range(m))/m)
    out.sort(); return out[250],out[9750]
def signflip_p(d, n=20000, seed=42):
    rng=random.Random(seed); obs=sum(d)/len(d); c=0; m=len(d)
    for _ in range(n):
        s=sum(x if rng.random()<.5 else -x for x in d)/m
        if abs(s)>=abs(obs): c+=1
    return (c+1)/(n+1)

qs=list(RAW)
rank_scores=[[] for _ in range(20)]
band13=[]; band47=[]; band820=[]; d13_47=[]; gap34=[]; adj={i:[] for i in range(1,20)}
perq={}
oracle3_c={}; oracle7_c={}
for key in qs:
    q=key.split("/")[-1][:-4]; caps=RAW[key][0]
    sc=[score(c,q) for c in caps]
    order=sorted(range(20), key=lambda i:(-sc[i], i))
    vals=[sc[i] for i in order]
    for r,v in enumerate(vals): rank_scores[r].append(v)
    m13=statistics.mean(vals[:3]); m47=statistics.mean(vals[3:7]); m820=statistics.mean(vals[7:])
    band13.append(m13); band47.append(m47); band820.append(m820); d13_47.append(m13-m47)
    gap34.append(vals[2]-vals[3])
    for r in range(1,20): adj[r].append(vals[r-1]-vals[r])
    perq[key]=dict(vals=vals,m13=m13,m47=m47,m820=m820,depletion=m13-m47)

def qstats(xs):
    xs=sorted(float(x) for x in xs); n=len(xs)
    def q(p): return xs[min(n-1,int(p*n))]
    return dict(n=n,mean=statistics.mean(xs),median=statistics.median(xs),sd=statistics.pstdev(xs),
                q10=q(.1),q25=q(.25),q75=q(.75),q90=q(.9))
res={"n_queries":len(qs),"reference_docs":N,"df_size":len(df),
     "rank_mean":{r+1:qstats(rank_scores[r]) for r in range(20)},
     "band_1_3":qstats(band13),"band_4_7":qstats(band47),"band_8_20":qstats(band820),
     "paired_1_3_minus_4_7":dict(qstats(d13_47),ci95=list(boot_ci(d13_47)),p_signflip=signflip_p(d13_47),
                                 frac_positive=sum(x>0 for x in d13_47)/len(d13_47)),
     "adjacent_drop":{f"{r}->{r+1}":dict(qstats(adj[r]),frac_positive=sum(x>0 for x in adj[r])/len(adj[r])) for r in range(1,20)},
     "gap_3_to_4":dict(qstats(gap34),ci95=list(boot_ci(gap34)),p_signflip=signflip_p(gap34),
                       frac_rank4_below_rank3=sum(x>0 for x in gap34)/len(gap34)),
     "ratio_depletion":float((statistics.mean(band13)-statistics.mean(band47))/statistics.mean(band13)),
     "rank3_over_rank1":float(statistics.mean([v[2]/v[0] for v in [d["vals"] for d in perq.values()]])),
     "rank4_over_rank1":float(statistics.mean([v[3]/v[0] for v in [d["vals"] for d in perq.values()]])),
     "rank7_over_rank1":float(statistics.mean([v[6]/v[0] for v in [d["vals"] for d in perq.values()]])),
     "rank20_over_rank1":float(statistics.mean([v[19]/v[0] for v in [d["vals"] for d in perq.values()]])),
    }

# correlate depletion with Oracle3 - Oracle7 generated CIDEr
NAME="eval_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review_generated_captions.json"
OUT=os.path.join(ROOT,"evcap_external/outputs")
def load(d): return json.load(open(os.path.join(OUT,d,NAME)))
R3=load("eval_val497_oracle3"); R7=load("eval_val497_oracle7")
refs={r["image_name"]:r["captions"] for r in R3}
tok=PTBTokenizer()
gt=tok.tokenize({q:[{"caption":c} for c in refs[q]] for q in refs})
def ciders(R):
    hp=tok.tokenize({r["image_name"]:[{"caption":r["prediction"].strip()}] for r in R}); hp={q:v[0] for q,v in hp.items()}
    return CF.corpus(gt,hp)
C3=ciders(R3); C7=ciders(R7)
dC={q:C3[q]-C7[q] for q in C3}
common=[k for k in perq if k in dC]
dep=np.array([perq[k]["depletion"] for k in common]); dc=np.array([dC[k] for k in common])
ra=np.argsort(np.argsort(dep)); rb=np.argsort(np.argsort(dc))
res["oracle3_minus_oracle7"]={"n":len(common),"mean_cider_diff":float(dc.mean()),
    "spearman_compat_depletion_vs_cider_gain":float(np.corrcoef(ra,rb)[0,1]),
    "mean_cider_diff_high_depletion_half":float(dc[dep>=np.median(dep)].mean()),
    "mean_cider_diff_low_depletion_half":float(dc[dep<np.median(dep)].mean())}
json.dump(res,open(os.path.join(ROOT,"reports/oracle_pool_depletion.json"),"w"),indent=2)
print(json.dumps(res,indent=2))

# per-query rank compatibility CSV
import csv
with open(os.path.join(ROOT,"reports/oracle_rank_compatibility.csv"),"w",newline="") as f:
    w=csv.writer(f); w.writerow(["image_name"]+[f"rank{i}" for i in range(1,21)])
    for key in qs: w.writerow([key]+[round(v,8) for v in perq[key]["vals"]])
