import glob,json,os
ROOT='/home/bn/videoufo_research'
ids=set()
for p in glob.glob(os.path.join(ROOT,'external/external_uar_*_cap.json')):
    d=json.load(open(p))
    for x in d.values():
        for lst in x[2]:
            if lst: ids.add(lst[0])
ids=sorted(ids)
open(os.path.join(ROOT,'external/external_uar_selected_clips.txt'),'w').write('\n'.join(ids)+'\n')
print('selected unique clips',len(ids))
