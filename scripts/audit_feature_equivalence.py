#!/usr/bin/env python3
"""Compare legacy in-domain features vs FAITHFUL_V2 recomputation (20 videos)."""
import hashlib,json,os,sys
import numpy as np, torch
ROOT='/home/bn/videoufo_research'
LEG='/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/lj/EVCapCode/ext_data/video_query'
FD=os.path.join(ROOT,'features/videollama_faithful_v2')
C=json.load(open(f'{ROOT}/external/external_val_indomain_d7_cap.json'))
ids=sorted({l[0] for v in C.values() for l in v[2] if l})
sample=sorted(ids,key=lambda c: hashlib.md5(c.encode()).hexdigest())[:20]
rows=[];ok=0
for c in sample:
    vid=os.path.basename(c)[:-4]
    lp=os.path.join(LEG,vid+'.pt'); fp=os.path.join(FD,c+'.pt')
    if not (os.path.exists(lp) and os.path.exists(fp)): rows.append({'id':c,'status':'missing'}); continue
    a=torch.load(lp,map_location='cpu').float(); b=torch.load(fp,map_location='cpu').float()
    if a.shape!=b.shape: rows.append({'id':c,'status':'shape','legacy':list(a.shape),'faithful':list(b.shape)}); continue
    d=(a-b).abs()
    cos=float(torch.nn.functional.cosine_similarity(a.flatten(),b.flatten(),dim=0))
    rows.append({'id':c,'status':'compared','shape':list(a.shape),'max_abs':float(d.max()),'mean_abs':float(d.mean()),'cosine':cos}); ok+=1
json.dump({'sample':sample,'compared':ok,'rows':rows,'legacy_path':LEG,'faithful_path':FD},open(f'{ROOT}/reports/indomain_feature_equivalence.json','w'),indent=2)
print('compared',ok,'of',len(sample))
for r in rows[:5]: print(r)
