"""Compare reproduced metric cells and measure output length and overlap with frozen evidence."""

from argparse import ArgumentParser
import csv
from pathlib import Path

import numpy as np

from audit_retrieval import load_keyed, tokens
from audit_statistics import K, SCALES, TRACKS, dump, load_predictions, read, table


def ngrams(words,n):
    return list(zip(*(words[i:] for i in range(n))))


def style(predictions, evidence, k):
    rows=[]
    for q,r in predictions.items():
        out=tokens(r['prediction']); g=ngrams(out,3)
        row={'words':len(out)}
        if evidence is not None:
            names=('fixed_top1_trigram_match','supplied_union_trigram_match')
            for label,caps in zip(names,(evidence[q][0][:1],evidence[q][0][:k]),strict=True):
                pool={s for c in caps for s in ngrams(tokens(c),3)}
                row[label]=sum(s in pool for s in g)/len(g) if g else 0.
        rows.append(row)
    return {a:float(np.mean([r[a] for r in rows])) for a in rows[0]}


def main():
    p=ArgumentParser(description=__doc__)
    for x in ('bundle','repo-data','results'):
        p.add_argument('--'+x,type=Path,required=True)
    args=p.parse_args()
    out=args.results
    summary=read(out/'summary.json')
    with (args.repo_data/'reports/post_fix/POST_FIX_FINAL_RESULTS_LEDGER.csv').open(encoding='utf-8') as f:
        ledger={r['experiment_id']:r for r in csv.DictReader(f)}
    metrics=[]
    for k,v in summary['val497']['means']['official_ptb'].items():
        frozen=float(ledger[k]['CIDEr'])
        metrics.append({'config':k,'split':'val497','frozen_ledger_cider':frozen,'ptb_cider':v,'difference':v-frozen})
    for k,v in summary['val150_loo']['means']['official_ptb'].items():
        if k in ledger:
            frozen=float(ledger[k]['CIDEr'])
            metrics.append({'config':k,'split':'val150_loo','frozen_ledger_cider':frozen,'ptb_cider':v,'difference':v-frozen})
    for split,tag in (('test2690','test2690'),('msvd670','msvd')):
        for k,v in summary[split]['means']['official_ptb'].items():
            path=args.repo_data/'evcap_external/outputs'/f'eval_{tag}_{k}'/'metrics.json'
            if path.is_file():
                frozen=read(path)['CIDEr']*100
                metrics.append({'config':k,'split':split,'frozen_ledger_cider':frozen,'ptb_cider':v,'difference':v-frozen})
    table(out/'metric_reconciliation.csv',metrics)
    reports=args.repo_data/'reports/post_fix'
    old=read(reports/'POST_FIX_MECHANISM_STATS.json')['pairs']
    primary=read(reports/'POST_FIX_PRIORITY1_STATS.json')
    old_pairs={key:old[previous] for key,previous in (('Oracle3-D3','P3-D3'),('Oracle7-Oracle3','P7-P3'),('Oracle1-D1','P1-D1'),('Oracle7-D7','P7-D7'))}
    old_pairs.update({k:primary['factorial'][k] for k in ('text_main','visual_main')})
    old_pairs['source_x_budget']=primary['DiD']
    with (out/'paired_contrasts.csv').open(encoding='utf-8') as f:
        records=list(csv.DictReader(f))
    paired_observed={(r['test'],r['metric']):r for r in records if r['family'] in ('oracle4','mechanism')}
    table(out/'old_and_replayed_paired_effects.csv',[
        {'contrast':k,'historical_diff':r.get('diff',r.get('mean')),'historical_p':r['p'],
         **{f'{m}/{a}':paired_observed[k,m][a] for m in ('historical_whitespace','official_ptb') for a in ('diff','ci_lo','ci_hi','p','bh','cluster_lo','cluster_hi','cluster_p')}}
        for k,r in old_pairs.items()])
    data=args.repo_data/'external'
    output=[]
    for split, cap in (('val497','external_val_d7_cap.json'),('test2690','external_test_d7_cap.json'),('msvd670','external_msvd_d7_cap.json')):
        path=data/cap
        e=load_keyed(path) if path.exists() else None
        for k in K:
            if split=='val497':
                file=args.bundle/'03_SAMPLE_LEVEL_EVIDENCE/predictions'/f'postfix_full_D{k}.json'
            else:
                d=args.repo_data/'evcap_external/outputs'/f'eval_{"msvd" if split=="msvd670" else split}_D{k}'
                file=next(d.glob('*generated_captions.json'))
            r=load_predictions({'p':file})['p']
            output.append({'split':split,'k':k,'n':len(r),**style(r,e,k),
                          'copy_evidence_status':'AVAILABLE' if e is not None else 'UNVERIFIED_CAP_ARTIFACT_MISSING'})
    table(out/'output_behavior.csv',output)
    c=summary['val497']['means']['official_ptb']
    selection={f'{s}/{t}':{k:c[f'postfix_{s}_{t}{k}'] for k in K} for s in SCALES for t in TRACKS}
    best={name:max(y,key=y.get) for name,y in selection.items()}
    q={k:max(abs(r['difference']) for r in metrics if r['split']==k) for k in {r['split'] for r in metrics}}
    dump(out/'output_audit_summary.json',{'official_metric_max_abs_difference_by_split':q,'metric_rows':len(metrics),
         'best_budget_by_scale_and_track':best,'mean_gap_to_k7':float(np.mean([max(v.values())-v[7] for v in selection.values()])),
         'matched_graph_dimensions':len(selection)})


if __name__=='__main__':
    main()
