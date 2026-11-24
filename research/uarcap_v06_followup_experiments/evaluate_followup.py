"""Score completed, trace-verified follow-ups using the independent Windows PTB scorer."""

import argparse
from pathlib import Path
import sys

from launch_pinned import require
from prepare_inputs import id_file, primitive_hash


FAMILIES={'A':{'arms':('A0','A1','A7'),'comparisons':(('A1','A0'),('A7','A0'))},
          'B':{'arms':('B3','B7HQ','B7LQ'),'comparisons':(('B7HQ','B3'),('B7LQ','B3'),('B7HQ','B7LQ'))}}
B_PINS=('frames','training','vision','feature_tensors','aggregate','qformer_query','gated_projection','query_projection','prompt_template_sha256')
A_PINS=('frames','training','qformer_query','query_projection','prompt_template_sha256')


def check_trace(traces, order, family):
    values={name:{row['query']:row for row in arr} for name,arr in traces.items()}
    root=FAMILIES[family]['arms'][0]
    for name,arr in traces.items():
        require([row['query'] for row in arr]==order and len(values[name])==len(order), 'Missing, duplicated or out-of-order trace')
    pins=B_PINS if family=='B' else A_PINS
    for q in order:
        base=values[root][q]
        for name in traces:
            require(all(values[name][q][k]==base[k] for k in pins), f'{name}:{q} failed trace parity')
    if family=='A':
        require(all(row['vision']['representatives']==[] and row['feature_tensors']==[] and row['text_count']==0 for row in traces[root]),
                'Nonempty A0 evidence')


def main():
    import json
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family',choices=FAMILIES,required=True)
    for k in ('runtime-manifest','full-ids','test-ids','audit-dir','vendor','java','jar','metadata','out'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--eligible',type=Path)
    a=p.parse_args()
    sys.path[:0]=[str(a.audit_dir.resolve()),str(a.vendor.resolve())]
    from audit_statistics import cluster_labels, dump, inference, load_predictions, read, scores, table

    spec=FAMILIES[a.family]
    files=read(a.runtime_manifest)
    require(set(files)==set(spec['arms']), 'Wrong arm set')
    receipts={arm:read(Path(files[arm]['receipt'])) for arm in spec['arms']}
    reference=receipts[spec['arms'][0]]
    require(all(rec['arm']==arm and all(rec[k]==reference[k] for k in ('common','runtime','code_sha256','engine_args'))
                for arm,rec in receipts.items()), 'Mixed run receipt, checkpoint, runtime or dataset')
    order=id_file(a.full_ids)
    test=id_file(a.test_ids)
    prereg=read(Path(__file__).with_name('CPU_PREFLIGHT_SUMMARY.json'))
    from launch_pinned import digest
    require(digest(a.full_ids)==prereg['source_sha256']['full_ids'] and digest(a.test_ids)==prereg['source_sha256']['test_ids'],
            'Wrong frozen cohorts')
    if a.family=='B':
        require(a.eligible is not None, 'B eligibility file required')
        test=read(a.eligible)
        require(primitive_hash(test)==prereg['eligible_ids_sha256'], 'B eligibility changed')
    traces={arm:[json.loads(line) for line in Path(files[arm]['trace']).read_text(encoding='utf-8').splitlines()] for arm in spec['arms']}
    check_trace(traces,order,a.family)
    balance={}
    if a.family=='B':
        eligible=set(test)
        for arm, rows in traces.items():
            require(all(row['text_count']==(7 if arm!='B3' and row['query'] in eligible else 3) for row in rows),
                    f'{arm}: observed text count differs from frozen policy')
        lens=[hi['prompt_tokens']-lo['prompt_tokens'] for hi,lo in zip(traces['B7HQ'],traces['B7LQ'],strict=True) if hi['query'] in eligible]
        import statistics
        balance={'HQ_minus_LQ_mean_prompt_tokens':statistics.mean(lens),
                 'HQ_LQ_median_absolute_token_difference':statistics.median(abs(x) for x in lens)}
    else:
        require(all(row['text_count']==k for arm,k in (('A0',0),('A1',1),('A7',7)) for row in traces[arm]),
                'Observed A text count differs from frozen policy')
    paths={arm:Path(files[arm]['predictions']) for arm in spec['arms']}
    for arm,path in paths.items():
        require([row['image_name'] for row in read(path)]==order, f'Missing, duplicated or out-of-order prediction in {arm}')
    want={Path(q).stem for q in test}
    configs={arm:{q:r for q,r in data.items() if q in want} for arm,data in load_predictions(paths).items()}
    keys,_,result=scores(configs,a.java,a.jar)
    require(set(keys)==want, 'Required scored queries are incomplete')
    source,meta=cluster_labels(a.metadata,keys)
    require(meta.get('sha256')=='3540d91f9fd5f9a56d75b00c6576f30814fd19395d027791a6d438e9dc297014' and
            meta.get('fallback_singletons')==0, 'Source metadata differs from audit witness')
    vals=result['official_ptb']
    tests={f'{x}-{y}':vals[x]-vals[y] for x,y in spec['comparisons']}
    results=inference(tests,source,seed=20261125,reps=10000)
    a.out.mkdir(parents=True,exist_ok=False)
    table(a.out/'paired_ptb.csv',results)
    dump(a.out/'metrics.json',{'sample_count':len(keys), 'clusters':meta,'CIDEr_points':{k:float(v.mean()) for k,v in vals.items()},
                               'prompt_balance':balance})


if __name__=='__main__':
    main()
