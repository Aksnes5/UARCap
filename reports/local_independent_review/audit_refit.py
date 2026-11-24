"""Sensitivity of the two Oracle contrasts to re-estimating CIDEr IDF in each cluster bootstrap draw."""

from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
import csv

import numpy as np
from scipy.sparse import csr_matrix

from audit_statistics import cluster_labels, dump, load_predictions, ptb_lookup, shared


def grams(text):
    tokens = text.split()
    return [Counter(zip(*(tokens[j:] for j in range(n)))) for n in range(1,5)]


def sparse(rows, vocabulary):
    lengths = [len(row) for row in rows]
    inds = [vocabulary[t] for row in rows for t in row]
    counts = [value for row in rows for value in row.values()]
    return csr_matrix((counts, inds, np.r_[0,np.cumsum(lengths)]), shape=(len(rows),len(vocabulary)))


def compiled(gts, systems):
    ref = [grams(s) for group in gts for s in group]
    n = len(gts)
    byref = np.repeat(np.arange(n), [len(row) for row in gts])
    nrefs = np.bincount(byref)
    document = [set() for _ in gts]
    for i,row in zip(byref,ref,strict=True):
        document[i].update(term for part in row for term in part)
    system_grams = [[grams(s) for s in predictions] for predictions in systems]
    vocab = {k:i for i,k in enumerate(sorted(set().union(*document,*(set(term for row in g for part in row for term in part) for g in system_grams))))}
    terms = sparse([dict.fromkeys(row,1.) for row in document],vocab).T.tocsr()
    refnorm = sparse([{t:c*c for t,c in part.items()} for parts in ref for part in parts],vocab)
    rl = np.array([sum(parts[1].values()) for parts in ref])
    models = []
    for cur in system_grams:
        norms = sparse([{t:c*c for t,c in part.items()} for parts in cur for part in parts],vocab)
        hl = np.asarray([sum(parts[1].values()) for parts in cur])
        penalty = np.exp(-(hl[byref]-rl)**2/72.)
        overlap = sparse([{term:min(count, r[term])*r[term] for term,count in h.items() if term in r}
                          for j,parts in zip(byref,ref,strict=True) for h,r in zip(cur[j],parts,strict=True)],vocab)
        models.append((norms,overlap,penalty))

    def evaluate(weights):
        idf_squared = (np.log(weights.sum())-np.log(np.maximum(1.,terms @ weights)))**2
        ref_len = np.sqrt((refnorm @ idf_squared).reshape(-1,4))
        scores = []
        for norm,overlap,penalty in models:
            norm_values = np.sqrt((norm @ idf_squared).reshape(-1,4))
            denom = ref_len * norm_values[byref]
            value = (overlap @ idf_squared).reshape(-1,4)
            score = np.divide(value,denom,out=np.zeros_like(value),where=denom>0).mean(axis=1)*penalty
            scores.append(np.bincount(byref,weights=score,minlength=n) / nrefs * 1000.)
        return np.array(scores)
    return evaluate


def main():
    parser=ArgumentParser(description=__doc__)
    for k in ('bundle','results','output','java','jar','metadata'):
        parser.add_argument('--'+k,type=Path,required=True)
    parser.add_argument('--reps',type=int,default=999)
    args=parser.parse_args()
    p=args.bundle/'03_SAMPLE_LEVEL_EVIDENCE/predictions'
    names=['postfix_full_D3','oracle3','oracle7']
    configs=load_predictions({k:p/f'{k}.json' for k in names})
    keys,refs=shared(configs)
    lookup=ptb_lookup(configs,refs,args.java,args.jar)
    prep=lambda s:lookup[s.replace('\n',' ')]
    evaluate=compiled([[prep(s) for s in refs[q]] for q in keys],[[prep(configs[k][q]['prediction']) for q in keys] for k in names])
    equal=evaluate(np.ones(len(keys)))
    with (args.results/'val497_per_query.csv').open(encoding='utf-8') as f:
        original=np.asarray([[float(row['official_ptb/'+k]) for k in names] for row in csv.DictReader(f)]).T
    error=float(np.abs(equal-original).max())
    if error>1e-8:
        raise ValueError(f'Independent CIDEr reimplementation differs by {error}')
    labels,meta=cluster_labels(args.metadata,keys)
    _,cidx=np.unique(labels,return_inverse=True)
    rng=np.random.default_rng(20261124)
    nc=len(set(labels))
    rc=rng.multinomial(nc,np.full(nc,1/nc),size=args.reps)
    paired=lambda s:np.array([s[1]-s[0],s[2]-s[1]])
    fixed=paired(equal)
    sampled=[]
    for cluster_weights in rc:
        w=cluster_weights[cidx]
        sampled.append(np.c_[paired(evaluate(w)) @ w / w.sum(),fixed @ w / w.sum()])
    ci=np.quantile(sampled,[.025,.975],axis=0)
    dump(args.output,{'reps':args.reps,'n_clusters':meta['n_clusters'],'max_query_score_error_against_frozen_evaluator':error,
                      'contrasts':{k:{'refit_idf_cluster_ci':ci[:,i,0].tolist(),'fixed_idf_cluster_ci_same_draws':ci[:,i,1].tolist()}
                                   for i,k in enumerate(('Oracle3-D3','Oracle7-Oracle3'))}})


if __name__=='__main__':
    main()
