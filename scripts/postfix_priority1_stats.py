#!/usr/bin/env python3
"""POST_FIX Priority-1 statistics: 2x2 factorial and external-vs-in-domain DiD."""
import json,os,random,sys
ROOT='/home/bn/videoufo_research'; O=ROOT+'/evcap_external/outputs'
N='eval_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review_generated_captions.json'
sys.path.insert(0,ROOT+'/evcap_external/evaluation')
from pycocoevalcap.cider.cider import Cider
GT={}
for l in open('/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/lj/datasets/MSRVTT-v2/val.caption.tsv'):
    p=l.split('\t',1); GT['datasets/MSRVTT-v2/videos/'+p[0].split('/')[-1][:-4]+'.mp4']=[x['caption'] for x in json.loads(p[1])]
def caps(n): return {x['image_name']:x['prediction'].strip() for x in json.load(open(f'{O}/eval_postfix_{n}/{N}'))}
names=['postfix_full_D3','postfix_full_D7','postfix_T7V3','postfix_T3V7','postfix_indomain_D1','postfix_indomain_D3','postfix_indomain_D5','postfix_indomain_D7']
P={n:caps(n) for n in names}
Q=sorted(set.intersection(*[set(v) for v in P.values()]))
g={q:GT[q] for q in Q}
C={n:dict(zip(Q,Cider().compute_score(g,{q:[P[n][q]] for q in Q})[1])) for n in names}
rng=random.Random(42)
def paired(d): 
    obs=sum(d)/len(d); bs=sorted(sum(d[rng.randrange(len(d))] for _ in range(len(d)))/len(d) for _ in range(10000))
    cnt=sum(1 for _ in range(10000) if abs(sum(x if rng.random()<0.5 else -x for x in d)/len(d))>=abs(obs))
    return {'mean':round(obs*100,2),'ci95':[round(bs[250]*100,2),round(bs[9750]*100,2)],'p':round((cnt+1)/10001,5)}
D3,D7,T7V3,T3V7='postfix_full_D3','postfix_full_D7','postfix_T7V3','postfix_T3V7'
def qd(a,b): return [C[a][q]-C[b][q] for q in Q]
res={'n':len(Q),'factorial':{
 'text_at_v3':paired(qd(T7V3,D3)),'text_at_v7':paired(qd(D7,T3V7)),
 'visual_at_t3':paired(qd(T3V7,D3)),'visual_at_t7':paired(qd(D7,T7V3))}}
text_main=[0.5*((C[T7V3][q]-C[D3][q])+(C[D7][q]-C[T3V7][q])) for q in Q]
vis_main=[0.5*((C[T3V7][q]-C[D3][q])+(C[D7][q]-C[T7V3][q])) for q in Q]
inter=[(C[D7][q]-C[T3V7][q])-(C[T7V3][q]-C[D3][q]) for q in Q]
res['factorial']['text_main']=paired(text_main); res['factorial']['visual_main']=paired(vis_main); res['factorial']['interaction']=paired(inter)
# external vs in-domain DiD
ext=[C[D3][q]-C[D7][q] for q in Q]
ind=[C['postfix_indomain_D3'][q]-C['postfix_indomain_D7'][q] for q in Q]
res['external_D3_minus_D7']=paired(ext); res['indomain_D3_minus_D7']=paired(ind); res['DiD']=paired([e-i for e,i in zip(ext,ind)])
json.dump(res,open(ROOT+'/reports/post_fix/POST_FIX_PRIORITY1_STATS.json','w'),indent=2)
print(json.dumps(res,indent=2))
