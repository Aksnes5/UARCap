#!/usr/bin/env python3
import json,os,sys
ROOT='/home/bn/videoufo_research'; FD=os.path.join(ROOT,'features/videollama_faithful_v2')
have={f[:-3] for f in os.listdir(FD) if f.endswith('.pt')}
sel=set()
for n in sys.argv[1:]:
    C=json.load(open(f'{ROOT}/external/external_val_{n}_cap.json'))
    for v in C.values(): sel|={l[0] for l in v[2] if l}
miss=sorted(sel-have)
open('/tmp/missing_union.txt','w').write('\n'.join(miss)+'\n')
print(len(sel),len(miss))
