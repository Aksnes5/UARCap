#!/usr/bin/env python3
"""CHECK 3.5: frozen-set remove-one assertions for LOO variants."""
import json,os,sys
ROOT='/home/bn/videoufo_research'
C=json.load(open(f'{ROOT}/external/external_val_d7_cap.json'))
bad=[]
qids=sorted(C)[:5]
for q in qids:
    base=C[q]
    for i in range(7):
        cap=json.load(open(f'{ROOT}/external/external_val_loo_rm{i}_cap.json'))[q]
        if len(cap[0])!=6 or len(cap[2])!=6 or len(cap[1])!=6: bad.append((q,i,'count'))
        removed=base[0][i]
        if removed in cap[0]: bad.append((q,i,'removed_present'))
        expect=[j for j in range(7) if j!=i]
        if cap[0]!=[base[0][j] for j in expect]: bad.append((q,i,'order'))
        if cap[1]!=[base[1][j] for j in expect]: bad.append((q,i,'scores'))
        if cap[2]!=[base[2][j] for j in expect]: bad.append((q,i,'pairing'))
        if len(set(cap[0])-set(base[0])): bad.append((q,i,'new_candidate'))
json.dump({'checked_queries':qids,'violations':bad,'pass':not bad},open(f'{ROOT}/reports/loo_assertions.json','w'),indent=2)
print('PASS' if not bad else bad[:10])
