"""Replay frozen caption decisions on CPU. Scores are CIDEr points (x100)."""

import argparse
from collections import defaultdict, Counter
import csv
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np


SCALES = ('10k', '50k', '100k', '250k', '500k', 'full')
TRACKS = ('CF', 'D')
K = (1, 3, 5, 7)
PUNCT = set("'' ' `` ` -LRB- -RRB- -LCB- -RCB- . ? ! , : - -- ... ;".split())


def read(path):
    return json.loads(path.read_bytes())


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def table(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        out = csv.DictWriter(f, list(dict.fromkeys(k for row in rows for k in row)))
        out.writeheader()
        out.writerows(rows)


def load_predictions(paths):
    configs = {}
    for name, path in paths.items():
        data = read(path)
        keyed = {Path(item['image_name']).stem: item for item in data}
        if len(keyed) != len(data) or any(not row['prediction'].strip() or not row['captions'] for row in data):
            raise ValueError(f'Duplicate queries or empty outputs/references in {name}')
        configs[name] = keyed
    return configs


def shared(configs):
    first = next(iter(configs.values()))
    keys = sorted(first)
    refs = {q: first[q]['captions'] for q in keys}
    if any(set(group) != set(keys) or any(group[q]['captions'] != refs[q] for q in keys) for group in configs.values()):
        raise ValueError('Predictions do not share exact query IDs and references')
    return keys, refs


def ptb_lookup(configs, refs, java, jar):
    texts = list(dict.fromkeys([t.replace('\n', ' ') for rs in refs.values() for t in rs] +
                                [r['prediction'].replace('\n', ' ') for config in configs.values() for r in config.values()]))
    with tempfile.TemporaryDirectory(prefix='uarcap_ptb_') as tmp:
        p = Path(tmp) / 'sentences.txt'
        p.write_bytes(('\n'.join(texts) + '\n').encode('utf-8'))
        cmd = [str(java), '-Dfile.encoding=UTF-8', '-cp', str(jar.resolve()), 'edu.stanford.nlp.process.PTBTokenizer',
               '-preserveLines', '-lowerCase', str(p)]
        result = subprocess.run(cmd, capture_output=True, check=True).stdout.decode('utf-8').split('\n')
    if result and result[-1] == '':
        result.pop()
    if len(texts) != len(result):
        raise ValueError(f'PTB line count mismatch: {len(texts)} / {len(result)}')
    return dict(zip(texts, [' '.join(w for w in line.rstrip().split(' ') if w not in PUNCT) for line in result], strict=True))


def scores(configs, java, jar):
    from pycocoevalcap.cider.cider import Cider
    keys, refs = shared(configs)
    lookup = ptb_lookup(configs, refs, java, jar)
    prep = lambda s: lookup[s.replace('\n', ' ')]
    ptb_refs = {q: [prep(t) for t in rs] for q, rs in refs.items()}
    result = {'historical_whitespace': {}, 'official_ptb': {}}
    for name, data in configs.items():
        raw = {q: [data[q]['prediction'].strip()] for q in keys}
        ptb = {q: [prep(data[q]['prediction'])] for q in keys}
        for label, g, p in (('historical_whitespace', refs, raw), ('official_ptb', ptb_refs, ptb)):
            result[label][name] = np.asarray(Cider().compute_score(g, p)[1]) * 100
    return keys, refs, result


def adjust(p_values):
    p = np.asarray(p_values)
    order = np.argsort(p, kind='stable')
    result = np.empty_like(p)
    result[order] = np.minimum(1, np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p)+1))[::-1])[::-1])
    return result


def cluster_labels(metadata, keys):
    if metadata is None:
        return None, {'status': 'UNVERIFIED_METADATA_MISSING'}
    obj = read(metadata)
    raw = {r['video_id']: r.get('url') for r in obj['videos']}
    groups = {}
    missing = []
    for q in keys:
        hit = re.search(r'(?:v=|youtu\.be/)([A-Za-z0-9_-]{6,})', raw.get(q) or '')
        if hit:
            groups[q] = hit[1]
        else:
            missing.append(q)
            groups[q] = q
    if len(missing) == len(keys):
        return None, {'status': 'UNVERIFIED_METADATA_NO_MATCH'}
    count = Counter(groups.values())
    return [groups[q] for q in keys], {'status': 'MATCHED_FROM_PUBLIC_MIRROR', 'n_clusters': len(count),
            'fallback_singletons': len(missing), 'max_cluster_size': max(count.values()),
            'sha256': sha256(metadata.read_bytes()).hexdigest()}


def holm(p_values):
    p = np.asarray(p_values)
    order = np.argsort(p, kind='stable')
    result = np.empty_like(p)
    result[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return result


def inference(contrasts, clusters, seed=20261124, reps=10000):
    x = np.column_stack(list(contrasts.values()))
    n = x.shape[0]
    rng = np.random.default_rng(seed)
    bs = np.empty((reps, x.shape[1]))
    flips = np.empty_like(bs)
    cb = cp = None
    if clusters is not None:
        _, cindex = np.unique(clusters, return_inverse=True)
        sizes = np.bincount(cindex)
        totals = np.zeros((len(sizes), x.shape[1]))
        np.add.at(totals, cindex, x)
        cb, cp = np.empty_like(bs), np.empty_like(bs)
    for start in range(0, reps, 250):
        end = min(start+250, reps)
        m = end-start
        weights = rng.multinomial(n, np.repeat(1/n, n), size=m)
        bs[start:end] = (weights @ x) / n
        flips[start:end] = ((rng.integers(0, 2, size=(m, n))*2-1) @ x) / n
        if clusters is not None:
            nc = len(sizes)
            weights = rng.multinomial(nc, np.repeat(1/nc, nc), size=m)
            cb[start:end] = (weights @ totals)/(weights @ sizes)[:, None]
            cp[start:end] = ((rng.integers(0, 2, size=(m, nc))*2-1) @ totals)/n
    obs = x.mean(axis=0)
    ci = np.quantile(bs, [.025, .975], axis=0)
    p = ((np.abs(flips) >= np.abs(obs) - 1e-12).sum(axis=0)+1)/(reps+1)
    sds = x.std(axis=0, ddof=1)
    qc = pc = None
    if cb is not None:
        qc = np.quantile(cb, [.025, .975], axis=0)
        pc = ((np.abs(cp) >= np.abs(obs) - 1e-12).sum(axis=0)+1)/(reps+1)
    return [{'test': name, 'n': n, 'diff': float(obs[i]), 'ci_lo': float(ci[0,i]), 'ci_hi': float(ci[1,i]),
             'p': float(p[i]), 'bh': float(adjust(p)[i]), 'holm': float(holm(p)[i]), 'paired_dz': float(obs[i]/sds[i]) if sds[i] else None,
             **({'cluster_lo': float(qc[0,i]), 'cluster_hi': float(qc[1,i]), 'cluster_p': float(pc[i]),
                 'cluster_bh': float(adjust(pc)[i]), 'cluster_holm': float(holm(pc)[i])} if qc is not None else {})}
            for i, name in enumerate(contrasts)]


def grid(c):
    f = lambda s,t,k: c[f'postfix_{s}_{t}{k}']
    return {f'{s}/{t}/k{a}-k{b}': f(s,t,a)-f(s,t,b) for s in SCALES for t in TRACKS
            for a,b in ((1,3),(1,5),(1,7),(3,7))} | {f'{s}/CF-D/k{k}': f(s,'CF',k)-f(s,'D',k) for s in SCALES for k in K}


def exploratory(c):
    f = lambda s,t,k: c[f'postfix_{s}_{t}{k}']
    tests = {f'{s}-10k/{t}/k{k}': f(s,t,k)-f('10k',t,k) for s in SCALES[1:] for t in TRACKS for k in K}
    b = {(s,t): f(s,t,3)-f(s,t,7) for s in SCALES for t in TRACKS}
    tests['mean_3minus7_all_scale_track'] = np.mean(list(b.values()),axis=0)
    tests['mean_3minus7_D_only'] = np.mean([d for (s,t), d in b.items() if t == 'D'],axis=0)
    tests['mean_3minus7_CF_only'] = np.mean([d for (s,t), d in b.items() if t == 'CF'],axis=0)
    tests['budget_x_order_mean'] = np.mean([b[s,'CF']-b[s,'D'] for s in SCALES],axis=0)
    for t in TRACKS:
        tests[f'budget_x_scale_full_minus10k_{t}'] = b['full',t]-b['10k',t]
    tests['budget_x_scale_x_order_endpoints'] = tests['budget_x_scale_full_minus10k_CF'] - tests['budget_x_scale_full_minus10k_D']
    return tests


def mechanisms(c):
    d = lambda k: c[f'postfix_full_D{k}']
    ind = lambda k: c[f'postfix_indomain_D{k}']
    oracle = {f'Oracle{k}-D{k}': c[f'oracle{k}']-d(k) for k in (1,3,7)}
    oracle['Oracle7-Oracle3'] = c['oracle7']-c['oracle3']
    t3v7, t7v3 = c['postfix_T3V7'], c['postfix_T7V3']
    tx3, tx7, vxt3, vxt7 = t7v3-d(3), d(7)-t3v7, t3v7-d(3), d(7)-t7v3
    others = {'text_at_v3': tx3, 'text_at_v7': tx7, 'visual_at_t3': vxt3, 'visual_at_t7': vxt7,
              'text_main': (tx3+tx7)/2, 'visual_main': (vxt3+vxt7)/2, 'text_x_visual': tx7-tx3,
              'external_D3-D7': d(3)-d(7), 'indomain_D3-D7': ind(3)-ind(7),
              'source_x_budget': (d(3)-d(7))-(ind(3)-ind(7))}
    others.update({f'{s}-D7': c[s]-d(7) for s in ('posrev','postop3last','midtail7','lowtail7','goodtail7','hardwrong7','easywrong7')})
    others['hardwrong7-easywrong7'] = c['hardwrong7']-c['easywrong7']
    return oracle, others


def old_grid(path, out):
    data = read(path)['tests']
    q = adjust([r['p'] for r in data])
    for r, corrected in zip(data,q,strict=True):
        r['q_recomputed_from_printed_p'] = float(corrected)
    table(out/'bh_frozen_p_values.csv', data)
    return {'tests': len(data),'raw_p_lt_05':sum(r['p'] < .05 for r in data), 'min_bh':float(min(q)),
            'bh_lt_05':int((q < .05).sum()), 'max_difference_from_stored_rounded_q':max(abs(r['q_bh']-j) for r,j in zip(data,q))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('bundle', 'repo-data', 'output', 'java', 'jar', 'vendor'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--metadata', type=Path)
    parser.add_argument('--reps', type=int, default=10000)
    args = parser.parse_args()
    sys.path.insert(0, str(args.vendor.resolve()))
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    official = args.repo_data/'reports/post_fix'
    result = {'historical_bh':old_grid(official/'AUDIT_MAIN_GRID_STATS.json',out), 'resamples':args.reps}
    raw = args.bundle/'03_SAMPLE_LEVEL_EVIDENCE/predictions'
    sets = {'val497':{p.stem:p for p in raw.glob('*.json') if not (p.stem.startswith(('loo','smoke')))}}
    for split, prefix in (('test2690','eval_test2690_'), ('msvd670','eval_msvd_')):
        sets[split] = {d.name.removeprefix(prefix):next(d.glob('*generated_captions.json'))
                       for d in (args.repo_data/'evcap_external/outputs').glob(prefix+'*') if list(d.glob('*generated_captions.json'))}
    sets['val150_loo'] = {p.stem:p for p in sorted(raw.glob('loo*.json'))}
    all_results = []
    for split, paths in sets.items():
        print('Scoring', split, len(paths), flush=True)
        configs = load_predictions(paths)
        if split == 'val150_loo':
            keys150, _ = shared(configs)
            full = load_predictions({'D7':raw/'postfix_full_D7.json'})['D7']
            configs['D7'] = {q:full[q] for q in keys150}
        keys, refs, by_method = scores(configs,args.java,args.jar)
        expected = {'val497':497,'test2690':2690,'msvd670':670,'val150_loo':150}[split]
        if len(keys)!=expected:
            raise ValueError(f'{split}: expected {expected}, actual {len(keys)}')
        if split == 'msvd670':
            matches = [re.fullmatch(r'([\w-]{11})_\d+_\d+', key, flags=re.ASCII) for key in keys]
            clusters = [m[1] for m in matches] if all(matches) else None
            provenance = {'status':'SOURCE_ID_FROM_FROZEN_MSVD_FILENAME' if clusters else 'UNVERIFIED_FILENAME_PATTERN',
                          'n_clusters':len(set(clusters)) if clusters else None}
        else:
            clusters, provenance = cluster_labels(args.metadata, keys)
        result[split] = {'n':len(keys),'ref_count':sum(map(len,refs.values())), 'clusters':provenance,'means':{},'families':{}}
        for name,c in by_method.items():
            result[split]['means'][name] = {k:float(v.mean()) for k,v in sorted(c.items())}
            if split=='val497':
                oracle, other = mechanisms(c)
                all_families = {'grid72':grid(c),'exploratory':exploratory(c),'oracle4':oracle,'mechanism':other,
                                'grid_plus_oracle76':grid(c)|oracle}
            elif split == 'val150_loo':
                removals = {f'remove_rank{i+1}':c[f'loo{i}']-c['D7'] for i in range(7)}
                effects = list(removals.values())
                all_families = {'loo':removals | {'head_minus_tail_removal':np.mean(effects[:3],axis=0)-np.mean(effects[3:],axis=0)}}
            else:
                all_families = {'benchmark':{f'{a}-{b}':c[a]-c[b] for a,b in (('D1','D3'),('D1','D7'),('D3','D7'),('D5','D7'),('Adaptive','D1'),('Adaptive','D3'),('B','D1')) if a in c and b in c}}
            for family,tests in all_families.items():
                parsed = inference(tests,clusters,reps=args.reps)
                all_results.extend({'split':split, 'metric':name, 'family':family, **r} for r in parsed)
                ps = [r['p'] for r in parsed]
                result[split]['families'].setdefault(name,{})[family] = {'count':len(ps),'raw_p_lt_05':sum(p < .05 for p in ps),'min_bh':min(r['bh'] for r in parsed),'bh_lt_05':sum(r['bh'] < .05 for r in parsed)}
        table(out/f'{split}_per_query.csv', [{'query': q, **{f'{m}/{a}':float(v[i]) for m,c in by_method.items() for a,v in sorted(c.items())}} for i,q in enumerate(keys)])
    table(out/'paired_contrasts.csv',all_results)
    dump(out/'summary.json',result)


if __name__ == '__main__':
    main()
