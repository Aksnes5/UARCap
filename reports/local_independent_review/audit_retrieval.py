"""Independently replay saved retrieval rankings and connect them to downstream query scores."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
import csv
from functools import lru_cache
from hashlib import sha256
from math import log, sqrt
from pathlib import Path
import re

import numpy as np
from scipy.stats import spearmanr

from audit_statistics import SCALES, TRACKS, cluster_labels, dump, read, table


def load_keyed(path):
    rows = read(path)
    keyed = {Path(k).stem: v for k, v in rows.items()}
    if len(keyed) != len(rows):
        raise ValueError(f'Duplicate IDs in {path.name}')
    return keyed


def weights_from_references(refs):
    docs = [set(tokens(s)) for rs in refs.values() for s in rs]
    dfs = Counter(w for doc in docs for w in doc)
    return {w:log(len(docs)/n) for w,n in dfs.items()}, log(len(docs)), len(docs)


def tokens(s):
    return re.findall(r"[a-z0-9']+", s.lower())


@lru_cache(maxsize=32000)
def rouge(a,b):
    a, b = a.lower().split(), b.lower().split()
    if not a or not b:
        return 0.0
    prev = [0]*(len(b)+1)
    for u in a:
        row = [0]
        for j,v in enumerate(b,1):
            row.append(prev[j-1]+1 if u == v else max(prev[j],row[-1]))
        prev = row
    if not prev[-1]:
        return 0.0
    p, r = prev[-1]/len(a),prev[-1]/len(b)
    return 2*p*r/(p+r)


def compare(old,new):
    return {'ordered_exact':old==new,
            'captions_exact':old[0]==new[0], 'sources_exact':old[2]==new[2],
            'similarities_exact':old[1]==new[1]}


def rank_compat(deltas, quality, groups):
    if len(deltas) != len(quality):
        raise ValueError('Inconsistent sample counts')
    rho, p = spearmanr(quality,deltas)
    lookup = defaultdict(list)
    for i,k in enumerate(groups or range(len(deltas))):
        lookup[k].append(i)
    idx = list(lookup.values())
    rng=np.random.default_rng(20261124)
    sim = []
    for _ in range(4000):
        sample = [i for j in rng.integers(0,len(idx),size=len(idx)) for i in idx[j]]
        sim.append(float(spearmanr(quality[sample], deltas[sample]).statistic))
    by_half = quality >= np.median(quality)
    return {'rho':float(rho),'p_query_asymptotic':float(p),'cluster_percentile_ci':np.quantile(sim,[.025,.975]).tolist(),
            'high_decay_mean_loss':float(deltas[by_half].mean()), 'low_decay_mean_loss':float(deltas[~by_half].mean()),
            'zero_delta_fraction':float((deltas==0).mean())}


def main():
    parser=ArgumentParser(description=__doc__)
    for key in ('bundle','repo-data','results','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--metadata',type=Path)
    args=parser.parse_args()
    bundle=args.bundle/'03_SAMPLE_LEVEL_EVIDENCE'
    external=args.repo_data/'external'
    out=args.output
    out.mkdir(exist_ok=True,parents=True)
    pred={name:{Path(r['image_name']).stem:r for r in read(bundle/'predictions'/f'{name}.json')}
          for name in ('oracle1','oracle3','oracle7','postfix_full_D1','postfix_full_D3','postfix_full_D7')}
    keys=sorted(pred['oracle3'])
    ref={q:pred['oracle3'][q]['captions'] for q in keys}
    idf, unseen, n=weights_from_references(ref)

    @lru_cache(maxsize=None)
    def vector(text):
        v={t:tf*idf.get(t,unseen) for t,tf in Counter(tokens(text)).items()}
        norm=sqrt(sum(x*x for x in v.values())) or 1
        return {t:x/norm for t,x in v.items()}

    def compatibility(cap,q):
        v=vector(cap)
        vecs=[vector(s) for s in ref[q]]
        return float(np.mean([sum(x*r.get(t,0) for t,x in v.items()) for r in vecs]))

    top=load_keyed(bundle/'retrieval/external_val_top20.json')
    sims=load_keyed(bundle/'retrieval/external_val_sims.json')
    configs=('oracle1','oracle3','oracle7','d7','goodtail7','hardwrong7','easywrong7')
    caps_by_config={k:load_keyed(external/f'external_val_{k}_cap.json') for k in configs}
    scores_by_config={k:load_keyed(external/f'external_val_{k}_sims.json') for k in configs}
    saved={k:{q:[row[0],[scores_by_config[k][q][1][scores_by_config[k][q][0].index(c)] for c in row[0]],row[2]]
               for q,row in caps_by_config[k].items()} for k in configs}
    source_hashes={f'{k}_{kind}':sha256((external/f'external_val_{k}_{kind}.json').read_bytes()).hexdigest()
                   for k in configs for kind in ('cap','sims')}
    if any(set(x)!=set(keys) for x in [top,sims,*saved.values()]):
        raise ValueError('Missing retrieval query')
    detail=[]
    rank_rows=[]
    for q in keys:
        caps, similarities = sims[q]
        if len(caps)!=20 or any((row['caption'],row['score'])!=(c,s) for row,c,s in zip(top[q],caps,similarities,strict=True)):
            raise ValueError(f'Candidate order mismatch {q}')
        ids=[r['clip_ids'] for r in top[q]]
        quality=[compatibility(c,q) for c in caps]
        order=sorted(range(20),key=lambda i:(-quality[i],i))
        rouge_quality=[sum(rouge(c,caps[j]) for j in range(20) if i!=j)/19 for i,c in enumerate(caps)]
        d=sorted(range(20),key=lambda i:(-rouge_quality[i],i))
        d_pool=d[3:]
        low=sorted(d_pool,key=quality.__getitem__)[:8]
        tails={'goodtail7':sorted(d_pool,key=lambda i:-quality[i])[:4],
               'hardwrong7':sorted(low,key=lambda i:-similarities[i])[:4],
               'easywrong7':sorted(low,key=lambda i:similarities[i])[:4]}

        def fields(indices):
            return [[seq[i] for i in indices] for seq in (caps,similarities,ids)]

        matches={k:compare(saved[k][q],fields(d[:3]+tails[k])) for k in tails}
        matches.update({k:compare(saved[k][q],fields(order[:7])) for k in configs[:3]})
        matches['d7']=compare(saved['d7'][q],fields(d[:7]))
        o_saved=saved['oracle7'][q]
        values=[compatibility(cap,q) for cap in o_saved[0]]
        head=float(np.mean(values[:3])); tail=float(np.mean(values[3:]))
        candidate_set={(c,s,tuple(a)) for c,s,a in zip(caps,similarities,ids,strict=True)}
        detail.append({'query':q,'head_compat':head,'tail_compat':tail,'decay':head-tail,
                       'all_archived_configs_within_raw_pool':all((c,s,tuple(i)) in candidate_set for k in configs
                       for c,s,i in zip(*saved[k][q],strict=True)),
                       'oracle_archives_exactly_equal':all(caps_by_config[k][q]==caps_by_config['oracle7'][q]
                         and scores_by_config[k][q]==scores_by_config['oracle7'][q] for k in ('oracle1','oracle3')),
                       'd_cap_scores_are_not_xclip':caps_by_config['d7'][q][1]!=saved['d7'][q][1],
                       **{f'{k}/{prop}':value for k,by_prop in matches.items() for prop,value in by_prop.items()}})
        rank_rows.extend({'query':q,'oracle_rank':j+1,'raw_rank':i+1,'clips':';'.join(ids[i]),
                          'raw_xclip_similarity':similarities[i],'reference_compatibility':quality[i]}
                         for j,i in enumerate(order))
    table(out/'oracle_query_integrity.csv',detail)
    table(out/'oracle_replayed_top20.csv',rank_rows)
    with (args.results/'val497_per_query.csv').open(encoding='utf-8') as stream:
        score={r['query']:r for r in csv.DictReader(stream)}
    groups, _=cluster_labels(args.metadata,keys)
    dec=np.array([r['decay'] for r in detail]); head=np.array([r['head_compat'] for r in detail]); tail=np.array([r['tail_compat'] for r in detail])
    correlation={}
    for metric in ('historical_whitespace','official_ptb'):
        loss=np.array([float(score[q][f'{metric}/oracle3'])-float(score[q][f'{metric}/oracle7']) for q in keys])
        correlation[metric]={'head_minus_tail':rank_compat(loss,dec,groups),'tail_only':rank_compat(loss,tail,groups),
                             'head_only':rank_compat(loss,head,groups)}
    points=np.array([float(score[q]['official_ptb/oracle3'])-float(score[q]['official_ptb/oracle7']) for q in keys])
    case_ids={'largest_3_over_7':keys[np.argmax(points)],'largest_7_over_3':keys[np.argmin(points)],
              'median_depletion':keys[np.argsort(dec)[len(dec)//2]]}
    samples={label:{'query':q,'oracle3_minus_7_points':float(points[keys.index(q)]),
                    'head_mean':float(head[keys.index(q)]),'tail_mean':float(tail[keys.index(q)]),
                    'oracle_top7_clips':saved['oracle7'][q][2],'D_top7_clips':saved['d7'][q][2],
                    'oracle_top7_scores':saved['oracle7'][q][1], 'D_top7_scores':saved['d7'][q][1],
                    'prediction_sha256':{k:sha256(v[q]['prediction'].encode()).hexdigest() for k,v in pred.items()}}
             for label,q in case_ids.items()}
    scale_rows=[]
    for t in TRACKS:
        tag=t.lower()
        small=load_keyed(external/f'external_val_10k_{tag}7_cap.json')
        small_scores=load_keyed(external/f'external_val_10k_{tag}7_sims.json')
        for s in SCALES[1:]:
            sep=f'{s}_' if s!='full' else ''
            high=load_keyed(external/f'external_val_{sep}{tag}7_cap.json')
            high_scores=load_keyed(external/f'external_val_{sep}{tag}7_sims.json')
            same=np.array([small[q][2][0]==high[q][2][0] for q in keys])
            x=np.array([high_scores[q][1][0]-small_scores[q][1][0] for q in keys])
            for k in (1,3,5,7):
                y=np.array([float(score[q][f'official_ptb/postfix_{s}_{t}{k}'])-float(score[q][f'official_ptb/postfix_10k_{t}{k}']) for q in keys])
                rho=spearmanr(x,y).statistic
                scale_rows.append({'scale_vs_10k':s,'ordering':t,'budget':k,'top1_changed_queries':int((~same).sum()),
                                   'top1_unchanged_queries':int(same.sum()),'selected_xclip_delta':float(x.mean()),
                                   'mean_cider_delta_changed':float(y[~same].mean()) if (~same).any() else None,
                                   'mean_cider_delta_unchanged':float(y[same].mean()) if same.any() else None,
                                   'rho_selected_xclip_delta_vs_cider_delta':float(rho) if np.isfinite(rho) else None})
    table(out/'top1_churn_downstream.csv',scale_rows)
    names=[k for k,v in detail[0].items() if isinstance(v,bool)]
    dump(out/'oracle_and_rank_summary.json',{'queries':len(keys),'reference_sentences':n,'df_terms':len(idf),
         'archived_cap_sha256':source_hashes,'matching_query_counts':{k:sum(row[k] for row in detail) for k in names},
         'replayed_compatibility_means_by_rank':[float(np.mean([r['reference_compatibility'] for r in rank_rows if r['oracle_rank']==k])) for k in range(1,21)],
         'head_1_to_3':float(head.mean()),'tail_4_to_7':float(tail.mean()),'decay':float(dec.mean()),
         'correlation':correlation,'sample_cases':samples})


if __name__=='__main__':
    main()
