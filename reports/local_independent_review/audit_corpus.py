"""Read the six frozen clip manifests and count source overlap without accessing videos."""

from argparse import ArgumentParser
import csv
from pathlib import Path
import re
from subprocess import check_output
from hashlib import sha256

from audit_inputs import REF
from audit_statistics import SCALES, dump, read


def main():
    p=ArgumentParser(description=__doc__)
    for k in ('repo','metadata','results'):
        p.add_argument('--'+k,type=Path,required=True)
    args=p.parse_args()
    meta=read(args.metadata)
    source={}
    for v in meta['videos']:
        q=v['video_id']; m=re.search(r'(?:v=|youtu\.be/)([A-Za-z0-9_-]{6,})',v.get('url') or '')
        if m:
            source[q]=m[1]
    result={}; prior=set()
    for scale in SCALES:
        name=f'manifests/clips_{scale}.tsv'
        b=check_output(['git','-C',str(args.repo),'cat-file','blob',f'{REF}:{name}'])
        ids=set(); yt=set(); rows=0
        for item in csv.DictReader(b.decode('utf-8').splitlines(),delimiter='\t'):
            ids.add(item['clip_id']); yt.add(item['source_id']); rows+=1
        result[scale]={'manifest_sha256':sha256(b).hexdigest(),'rows':rows,'unique_clip_ids':len(ids),'source_count':len(yt),
                       'previous_clip_ids_missing':len(prior-ids),'source_overlap_with_msrvtt':{name:len({v for q,v in source.items()
                           if lower<=int(q.removeprefix('video'))<upper} & yt) for name,lower,upper in
                           (('train6513',0,6513),('val497',6513,7010),('test2990',7010,10000))}}
        prior=ids
    dump(args.results/'corpus_clip_manifest_audit.json',{'levels':result,'metadata_valid_source_ids':len(source),
          'grouping_limit':'No raw full caption CSV or production embedding arrays were used. Clip and source checks cannot establish caption normalization or visual near-duplicate absence.'})


if __name__=='__main__':
    main()
