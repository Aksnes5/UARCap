"""Build two frozen interventions from archived retrieval inputs, without reading GT or predictions."""

import argparse
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import re


BAND = 8
MIN_SCORE_GAP = 0.005
MAX_HEAD_OVERLAP_GAP = 0.08
MAX_INTERNAL_OVERLAP_GAP = 0.08
MIN_ELIGIBLE = 1000
MIN_COVERAGE = 0.50


def primitive_hash(value):
    return sha256(json.dumps(value, ensure_ascii=False, separators=(',', ':'), sort_keys=True).encode()).hexdigest()


def words(s):
    return re.findall(r"[a-z0-9']+", s.lower())


def jaccard(a, b):
    return len(a & b) / len(a | b) if a or b else 0.0


def source(clip):
    return clip.rsplit('.', 1)[0]


@dataclass(frozen=True)
class Item:
    text: str
    clips: tuple
    score: float

    @classmethod
    def from_pool(cls, record):
        if not record['clip_ids']:
            raise ValueError('Candidate without representative clip')
        return cls(record['caption'], tuple(record['clip_ids']), float(record['score']))

    @property
    def youtube(self):
        return source(self.clips[0])


@dataclass(frozen=True)
class Tail:
    indices: tuple
    length: int
    relevance: float
    head_overlap: float
    redundancy: float


def tails(items, ids, head):
    h_src = {items[i].youtube for i in head}
    word_lists = {i: words(items[i].text) for i in (*ids, *head)}
    w = {i:set(t) for i,t in word_lists.items()}
    overlaps = {i: sum(jaccard(w[i],w[h]) for h in head)/len(head) for i in ids}
    similarity = {(i,j): jaccard(w[i],w[j]) for i,j in combinations(ids,2)}
    for subset in combinations(ids,4):
        source_ids = {items[i].youtube for i in subset}
        if len(source_ids) != 4 or source_ids & h_src:
            continue
        yield Tail(subset, sum(len(word_lists[i]) for i in subset), sum(items[i].score for i in subset)/4,
                   sum(overlaps[i] for i in subset)/4, sum(similarity[p] for p in combinations(subset,2))/6)


def choose(items, head):
    remainder = sorted(set(range(len(items)))-set(head), key=lambda i:(-items[i].score,i))
    high, low = list(tails(items,remainder[:BAND],head)), list(tails(items,remainder[-BAND:],head))
    if not high or not low:
        return None, 'INSUFFICIENT_SOURCE_DIVERSE_TAILS'
    best = None
    for h in high:
        for l in low:
            gap=h.relevance-l.relevance
            if gap<MIN_SCORE_GAP:
                continue
            length_limit=max(5, .10*(h.length+l.length)/2)
            dl, dh, dr = abs(h.length-l.length), abs(h.head_overlap-l.head_overlap), abs(h.redundancy-l.redundancy)
            if dl>length_limit or dh>MAX_HEAD_OVERLAP_GAP or dr>MAX_INTERNAL_OVERLAP_GAP:
                continue
            key=(dl/length_limit+dh/MAX_HEAD_OVERLAP_GAP+dr/MAX_INTERNAL_OVERLAP_GAP, -gap, h.indices,l.indices)
            if best is None or key<best[0]:
                best=(key,h,l)
    return ((best[1],best[2]),'ELIGIBLE') if best else (None,'NO_BALANCED_SEPARATED_TAIL_PAIR')


def anchor(items, cap, sims):
    if len(items)!=20 or len(set((i.text,i.clips) for i in items))!=20:
        raise ValueError('Invalid top-20 group list')
    if len(cap[0])<3 or len(cap[2])<3 or len(sims[0])!=len(set(sims[0])):
        raise ValueError('Visual indices cannot be uniquely pinned')
    by_key={(i.text,i.clips):n for n,i in enumerate(items)}
    sims_by_text=dict(zip(*sims,strict=True))
    result=[]
    for text, clips in zip(cap[0][:3],cap[2][:3],strict=True):
        idx=by_key.get((text,tuple(clips)))
        if idx is None or sims_by_text.get(text)!=items[idx].score:
            raise ValueError('Archived D head differs from top-20 caption/clip/score triple')
        result.append(idx)
    if len(set(result))!=3:
        raise ValueError('Repeated head candidate')
    return tuple(result)


def visual_contract(prefix):
    return {'representatives':[i.clips[0] for i in prefix], 'similarities':[i.score for i in prefix],
            'lookup_text_hashes':[sha256(i.text.encode()).hexdigest() for i in prefix]}


def pack(prefix, tail=()):
    return [[i.text for i in (*prefix,*tail)], [i.score for i in (*prefix,*tail)],
            [list(i.clips) for i in prefix]+[[] for _ in tail]]


def id_file(path):
    ids=path.read_text(encoding='utf-8-sig').splitlines()
    if any(not q for q in ids) or len(set(ids))!=len(ids):
        raise ValueError(f'Invalid query manifest: {path.name}')
    return ids


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n',encoding='utf-8')


def run(top, cap, sims, full, subset, out):
    if len(full)!=2990 or len(subset)!=2690 or set(subset)-set(full):
        raise ValueError('Run/test cohort counts differ from frozen split')
    if any(set(obj)!=set(full) for obj in (top,cap,sims)):
        raise ValueError('Candidate and frozen manifests differ')
    empty_cap, empty_sims = {}, {}
    text3, text_hq, text_lq, visual_sims = {}, {}, {}, {}
    contracts, eligible, balances = {}, [], []
    status=Counter()
    subset_set=set(subset)
    for q in full:
        empty_cap[q],empty_sims[q]=[[],[],[]],[[],[]]
        items=[Item.from_pool(r) for r in top[q]]
        head=anchor(items,cap[q],sims[q])
        base=tuple(items[i] for i in head)
        contracts[q]=visual_contract(base)
        text3[q]=pack(base)
        visual_sims[q]=[[i.text for i in base],[i.score for i in base]]
        pair,reason=choose(items,head) if q in subset_set else (None,'PILOT_EXCLUDED')
        status[reason]+=1
        if pair is None:
            text_hq[q]=text_lq[q]=text3[q]
            continue
        h,l=pair
        eligible.append(q)
        text_hq[q]=pack(base,tuple(items[i] for i in h.indices))
        text_lq[q]=pack(base,tuple(items[i] for i in l.indices))
        balances.append([h.relevance-l.relevance,abs(h.length-l.length),abs(h.head_overlap-l.head_overlap),abs(h.redundancy-l.redundancy)])
    out.mkdir(parents=True,exist_ok=True)
    objects={'A_k0_cap':empty_cap,'A_k0_sims':empty_sims,'B_text3_cap':text3,'B_text7_hq_cap':text_hq,
             'B_text7_lq_cap':text_lq,'B_shared_visual_sims':visual_sims,'B_visual_contract':contracts,'B_eligible_queries':eligible}
    for name,item in objects.items():
        dump(out/f'{name}.json',item)
    metrics={key:{'mean':sum(col)/len(col),'min':min(col),'max':max(col)} for key,col in
             zip(('relevance_gap','absolute_word_delta','absolute_head_jaccard_delta','absolute_internal_jaccard_delta'),zip(*balances),strict=True)} if balances else {}
    return {'run_queries':len(full),'pilot_excluded_queries':len(subset),'cohort_counts':dict(status),
            'B_primary_eligible':len(eligible),'B_coverage':len(eligible)/len(subset),
            'B_coverage_gate_passed':len(eligible)>=MIN_ELIGIBLE and len(eligible)/len(subset)>=MIN_COVERAGE,
            'B_matching_metrics':metrics,'inputs_fingerprints':{k:primitive_hash(v) for k,v in objects.items()},
            'eligible_ids_sha256':primitive_hash(eligible),'formal_caps_or_predictions_published':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('top20','cap','sims','full-ids','test-ids','out','report'):
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    load=lambda x:json.loads(x.read_bytes())
    inputs={k:sha256(getattr(a,k).read_bytes()).hexdigest() for k in ('top20','cap','sims','full_ids','test_ids')}
    report=run(load(a.top20),load(a.cap),load(a.sims),id_file(a.full_ids),id_file(a.test_ids),a.out)
    dump(a.report,{'source_sha256':inputs,**report})


if __name__=='__main__':
    main()
