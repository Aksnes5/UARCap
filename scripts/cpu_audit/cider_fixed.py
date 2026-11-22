"""Exact re-implementation of pycocoevalcap CiderScorer permitting a fixed
document-frequency / ref_len corpus (for IDF-scope forensics only)."""
import math
from collections import defaultdict
import numpy as np

def precook(s, n=4):
    words = s.split()
    counts = defaultdict(int)
    for k in range(1, n + 1):
        for i in range(len(words) - k + 1):
            counts[tuple(words[i:i + k])] += 1
    return counts

def cook_refs(refs, n=4):
    return [precook(r, n) for r in refs]

def compute_df(crefs):
    df = defaultdict(float)
    for refs in crefs:
        for ngram in set(ng for ref in refs for ng in ref):
            df[ngram] += 1
    return df

def compute_cider(ctest, crefs, df, ref_len, n=4, sigma=6.0):
    def counts2vec(cnts):
        vec = [defaultdict(float) for _ in range(n)]
        length = 0
        norm = [0.0 for _ in range(n)]
        for (ngram, term_freq) in cnts.items():
            dfv = math.log(max(1.0, df[ngram]))
            m = len(ngram) - 1
            vec[m][ngram] = float(term_freq) * (ref_len - dfv)
            norm[m] += vec[m][ngram] ** 2
            if m == 1:
                length += term_freq
        norm = [math.sqrt(x) for x in norm]
        return vec, norm, length

    def sim(vh, vr, nh, nr, lh, lr):
        delta = float(lh - lr)
        val = np.zeros(n)
        for m in range(n):
            for (ngram, count) in vh[m].items():
                val[m] += min(vh[m][ngram], vr[m][ngram]) * vr[m][ngram]
            if nh[m] != 0 and nr[m] != 0:
                val[m] /= (nh[m] * nr[m])
            val[m] *= np.e ** (-(delta ** 2) / (2 * sigma ** 2))
        return val

    scores = []
    for test, refs in zip(ctest, crefs):
        vec, norm, length = counts2vec(test)
        score = np.zeros(n)
        for ref in refs:
            vr, nr, lr = counts2vec(ref)
            score += sim(vec, vr, norm, nr, length, lr)
        scores.append(np.mean(score) / len(refs) * 10.0)
    return scores

def corpus(id2refs, id2hyp, corpus_refs=None, corpus_n=None, n=4):
    """corpus_refs: list of ref-lists used to estimate DF; defaults to id2refs."""
    ids = list(id2refs)
    crefs = [cook_refs(id2refs[i], n) for i in ids]
    if corpus_refs is None:
        df = compute_df(crefs)
        rl = math.log(float(len(crefs)))
    else:
        cdf = compute_df([cook_refs(r, n) for r in corpus_refs])
        df = cdf
        rl = math.log(float(len(corpus_refs)))
    ctest = [precook(id2hyp[i], n) for i in ids]
    return dict(zip(ids, compute_cider(ctest, crefs, df, rl, n)))

def score_one(hyp, refs, df, ref_len, n=4, sigma=6.0):
    return compute_cider([precook(hyp, n)], [cook_refs(refs, n)], df, ref_len, n, sigma)[0]
