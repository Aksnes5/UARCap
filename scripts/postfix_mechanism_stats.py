#!/usr/bin/env python3
import json,os,random,sys
ROOT='/home/bn/videoufo_research'; O=ROOT+'/evcap_external/outputs'
N='eval_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review_generated_captions.json'
sys.path.insert(0,ROOT+'/evcap_external/evaluation')
from pycocoevalcap.cider.cider import Cider
PATH={'D1':'eval_postfix_postfix_full_D1','D3':'eval_postfix_postfix_full_D3','D7':'eval_postfix_postfix_full_D7',
'P1':'eval_postfix_oracle1','P3':'eval_postfix_oracle3','P7':'eval_postfix_oracle7',
'posrev':'eval_postfix_posrev','postop':'eval_postfix_postop3last','mid':'eval_postfix_midtail7','low':'eval_postfix_lowtail7',
'good':'eval_postfix_goodtail7','hard':'eval_postfix_hardwrong7','easy':'eval_postfix_easywrong7'}
GT={}
for l in open('/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/lj/datasets/MSRVTT-v2/val.caption.tsv'):
    p=l.split('\t',1); GT['datasets/MSRVTT-v2/videos/'+p[0].split('/')[-1][:-4]+'.mp4']=[x['caption'] for x in json.loads(p[1])]
P={n:{x['image_name']:x['prediction'].strip() for x in json.load(open(f'{O}/{d}/{N}'))} for n,d in PATH.items()}
Q=sorted(set.intersection(*[set(v) for v in P.values()])); g={q:GT[q] for q in Q}
C={n:dict(zip(Q,Cider().compute_score(g,{q:[P[n][q]] for q in Q})[1])) for n in P}
rng=random.Random(42)
def pair(a,b):
    d=[C[a][q]-C[b][q] for q in Q]; obs=sum(d)/len(d)
    bs=sorted(sum(d[rng.randrange(len(d))] for _ in range(len(d)))/len(d) for _ in range(10000))
    cnt=sum(1 for _ in range(10000) if abs(sum(x if rng.random()<0.5 else -x for x in d)/len(d))>=abs(obs))
    return {'diff':round(obs*100,2),'ci95':[round(bs[250]*100,2),round(bs[9750]*100,2)],'p':round((cnt+1)/10001,5)}
pairs=[('P1','D1'),('P3','D3'),('P7','D7'),('P7','P3'),('posrev','D7'),('postop','D7'),
('mid','D7'),('low','D7'),('good','D7'),('hard','D7'),('easy','D7'),('hard','easy')]
out={f'{a}-{b}':pair(a,b) for a,b in pairs}
json.dump({'cider':{n:round(json.load(open(f'{O}/{d}/metrics.json'))['CIDEr']*100,2) for n,d in PATH.items()},'pairs':out},open(ROOT+'/reports/post_fix/POST_FIX_MECHANISM_STATS.json','w'),indent=2)
print(json.dumps(out,indent=2))
