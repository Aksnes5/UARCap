#!/usr/bin/env python3
"""Corrected Text x Visual 2x2 factorial statistics (per-query, val497)."""
import json,os,random,sys
ROOT='/home/bn/videoufo_research'; OUT=ROOT+'/evcap_external/outputs'
NAME='eval_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review_generated_captions.json'
sys.path.insert(0,ROOT+'/evcap_external/evaluation')
from pycocoevalcap.cider.cider import Cider
DIRS={'T3V3':'eval_val497_D3','T7V3':'eval_val497_T7V3','T3V7':'eval_val497_T3V7','T7V7':'eval_val497_D7'}
P={k:{x['image_name']:x['prediction'].strip() for x in json.load(open(f'{OUT}/{d}/{NAME}'))} for k,d in DIRS.items()}
for k in P: assert len(P[k])==497 and len(set(P[k]))==497
Q=sorted(set.intersection(*[set(v) for v in P.values()])); assert len(Q)==497
rows=json.load(open(f'{OUT}/eval_val497_D3/{NAME}')); GT={x['image_name']:x['captions'] for x in rows}
g={q:GT[q] for q in Q}
C={k:dict(zip(Q,Cider().compute_score(g,{q:[P[k][q]] for q in Q})[1])) for k in P}
rng=random.Random(42)
def paired(d):
    obs=sum(d)/len(d)
    boots=sorted(sum(d[rng.randrange(len(d))] for _ in range(len(d)))/len(d) for _ in range(10000))
    cnt=sum(1 for _ in range(10000) if abs(sum(x if rng.random()<0.5 else -x for x in d)/len(d))>=abs(obs))
    return {'mean':round(obs,4),'ci95':[round(boots[250],4),round(boots[9750],4)],'p_perm':round((cnt+1)/10001,5)}
def qd(a,b): return [C[a][q]-C[b][q] for q in Q]
res={}
res['text_at_v3']=paired(qd('T7V3','T3V3'))
res['text_at_v7']=paired(qd('T7V7','T3V7'))
res['visual_at_t3']=paired(qd('T3V7','T3V3'))
res['visual_at_t7']=paired(qd('T7V7','T7V3'))
text_main=[0.5*((C['T7V3'][q]-C['T3V3'][q])+(C['T7V7'][q]-C['T3V7'][q])) for q in Q]
vis_main=[0.5*((C['T3V7'][q]-C['T3V3'][q])+(C['T7V7'][q]-C['T7V3'][q])) for q in Q]
inter=[(C['T7V7'][q]-C['T3V7'][q])-(C['T7V3'][q]-C['T3V3'][q]) for q in Q]
res['text_main_averaged']=paired(text_main)
res['visual_main_averaged']=paired(vis_main)
res['interaction_did']=paired(inter)
res['unscaled_text_contrast']=round(2*res['text_main_averaged']['mean'],4)
res['unscaled_visual_contrast']=round(2*res['visual_main_averaged']['mean'],4)
res['n']=len(Q)
json.dump(res,open(ROOT+'/reports/text_visual_2x2_factorial.json','w'),indent=2)
print(json.dumps(res,indent=2))
