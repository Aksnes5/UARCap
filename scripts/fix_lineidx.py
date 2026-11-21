#!/usr/bin/env python3
import os,struct,sys
for p in sys.argv[1:]:
    if not os.path.exists(p) or not p.endswith(('.tsv','.txt')): continue
    data=open(p,'rb').read(); offs=[]; pos=0
    for line in data.splitlines(keepends=True): offs.append(pos); pos+=len(line)
    with open(p+'.lineidx','w') as w:
        for o in offs: w.write(str(o)+'\n')
    with open(p+'.lineidx.8b','wb') as w:
        for o in offs: w.write(struct.pack('<q',o))
    base=p[:-4] if p.endswith('.tsv') else p
    import shutil
    shutil.copyfile(p+'.lineidx', base+'.lineidx'); shutil.copyfile(p+'.lineidx.8b', base+'.lineidx.8b')
    print('lineidx',p,len(offs))
