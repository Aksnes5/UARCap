#!/usr/bin/env python3
import json,os,re,math,random,sys
import numpy as np
ROOT='/home/bn/videoufo_research'; E=ROOT+'/external'; O=ROOT+'/evcap_external/outputs'
N='eval_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review_generated_captions.json'
sys.path.insert(0,ROOT+'/evcap_external/evaluation')
from pycocoevalcap.cider.cider import Cider
# fixed-DF scorer (same as oracle_fixeddf)
import re as _re
GT={}
for l in open('/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/lj/datasets/MSRVTT-v2/val.caption.tsv'):
    p=l.split('\t',1); GT['datasets/MSRVTT-v2/videos/'+p[0].split('/')[-1][:-4]+'.mp4']=[x['caption'] for x in json.loads(p[1])]
def tok(t): return _re.findall(r"[a-z0-9']+", t.lower())
refs={q:tok(ref) for q,rr in GT.items() for ref in rr} if False else None
refdocs=[]; reftok={}
for q,rr in GT.items():
    reftok[q]=[tok(x) for x in rr]; refdocs+=reftok[q]
Nref=len(refdocs); df={}
for d in refdocs:
    for w in set(d): df[w]=df.get(w,0)+1
idf={w:math.log(Nref/c) for w,c in df.items()}
def vec(t):
    tf={}
    for w in t: tf[w]=tf.get(w,0)+1
    v={w:c*idf.get(w,math.log(Nref)) for w,c in tf.items()}; n=math.sqrt(sum(x*x for x in v.values())) or 1
    return {w:x/n for w,x in v.items()}
def score(c,q):
    cv=vec(tok(c)); return float(np.mean([sum(v*vec(r).get(w,0) for w,v in cv.items()) for r in reftok[q]]))
# oracle caps
OR={k:json.load(open(f'{E}/external_val_oracle{k}_cap.json')) for k in (1,3,7)}
Q=sorted(set.intersection(*[set(v) for v in OR.values()]))
nested=all(OR[3][q][0]==OR[7][q][0] and OR[1][q][0]==OR[7][q][0] for q in Q)
# compatibility of ranks 1..7 from oracle7 order
comp={q:[score(c,q) for c in OR[7][q][0]] for q in Q}
head=np.array([np.mean(comp[q][:3]) for q in Q]); tail=np.array([np.mean(comp[q][3:]) for q in Q])
# generation cider per config
def caps(n): return {x['image_name']:x['prediction'].strip() for x in json.load(open(f'{O}/eval_postfix_{n}/{N}'))}
names={'D3':'postfix_full_D3','D7':'postfix_full_D7','O1':'oracle1','O3':'oracle3','O7':'oracle7'}
P={k:caps(v) for k,v in names.items()}
g={q:GT[q] for q in Q}
C={k:dict(zip(Q,Cider().compute_score(g,{q:[P[k][q]] for q in Q})[1])) for k in P}
d37=np.array([C['O3'][q]-C['O7'][q] for q in Q])
decay=np.array([head[i]-tail[i] for i in range(len(Q))])
from scipy.stats import spearmanr
rho,p=spearmanr(decay,d37)
# source clusters
d=json.load(open('/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/lj/datasets/MSRVTT-v2/train_val_videodatainfo.json'))
def yt(u): 
    m=re.search(r'(?:v=|youtu\.be/)([A-Za-z0-9_-]{6,})',u or ''); return m.group(1) if m else None
src={f'datasets/MSRVTT-v2/videos/{v["video_id"]}.mp4':yt(v.get('url')) or v['video_id'] for v in d['videos']}
def cluster_test(a,b):
    diff={q:C[a][q]-C[b][q] for q in Q}; by={}
    for q in Q: by.setdefault(src.get(q,q),[]).append(diff[q])
    srcs=list(by); rng=random.Random(42); obs=float(np.mean(list(diff.values()))); bs=[]
    for _ in range(5000):
        samp=[]
        for _ in range(len(srcs)): samp+=by[srcs[rng.randrange(len(srcs))]]
        bs.append(float(np.mean(samp)))
    bs.sort(); return round(obs*100,2),round(bs[125]*100,2),round(bs[4874]*100,2),len(srcs)
res={'n':len(Q),'nested':nested,'head_compat':round(float(head.mean()),4),'tail_compat':round(float(tail.mean()),4),
 'mean_decay':round(float(decay.mean()),4),'spearman_decay_vs_O3minusO7':round(float(rho),4),'p':float(p),
 'per_rank_compat':{i+1:round(float(np.mean([comp[q][i] for q in Q])),4) for i in range(7)},
 'cluster':{'O3-D3':cluster_test('O3','D3'),'O7-O3':cluster_test('O7','O3'),'D3-D7':cluster_test('D3','D7'),'O7-D7':cluster_test('O7','D7')}}
json.dump(res,open(ROOT+'/reports/post_fix/AUDIT_ORACLE_ANALYSIS.json','w'),indent=2)
print(json.dumps(res,indent=2))
