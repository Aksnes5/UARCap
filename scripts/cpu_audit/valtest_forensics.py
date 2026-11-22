import json, os, sys, math, statistics, random
sys.path.insert(0, "/home/bn/videoufo_research/scripts/cpu_audit")
sys.path.insert(0, "/home/bn/videoufo_research/evcap_external/evaluation")
import cider_fixed as CF
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

ROOT = "/home/bn/videoufo_research"
OUT = os.path.join(ROOT, "evcap_external/outputs")
NAME7 = "eval_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review_generated_captions.json"

def load_gen(d):
    return json.load(open(os.path.join(OUT, d, NAME7)))

val = load_gen("eval_val497_D3")
test = load_gen("eval_test2690_D3")
testfull = load_gen("eval_test2990_D1")

def split_model(rows):
    gts = {r["image_name"]: r["captions"] for r in rows}
    hyp = {r["image_name"]: r["prediction"].strip() for r in rows}
    tok = PTBTokenizer()
    g = tok.tokenize({k: [{"caption": c} for c in v] for k, v in gts.items()})
    h = tok.tokenize({k: [{"caption": v}] for k, v in hyp.items()})
    g = {k: v for k, v in g.items()}
    h = {k: v[0] for k, v in h.items()}
    return gts, g, h

gv_raw, gv, hv = split_model(val)
gt_raw, gt, ht = split_model(test)
gfull_raw, gfull, hfull = split_model(testfull)

def summ(xs):
    xs = sorted(xs)
    n = len(xs)
    return dict(n=n, mean=statistics.mean(xs), median=statistics.median(xs),
                sd=statistics.pstdev(xs), q10=xs[int(.10*n)], q25=xs[int(.25*n)],
                q75=xs[int(.75*n)], q90=xs[int(.90*n)], min=xs[0], max=xs[-1],
                frac_lt_01=sum(x < .1 for x in xs)/n, frac_gt_1=sum(x > 1 for x in xs)/n)

res = {"split_stats": {}, "idf_scope": {}, "reference_structure": {}, "retrieval_compat": {}}

# native recomputation
Cv = CF.corpus(gv, hv)
Ct = CF.corpus(gt, ht)
Cfull = CF.corpus(gfull, hfull)
res["native_recompute"] = {
    "val497_D3": summ(Cv.values()), "test2690_D3": summ(Ct.values()),
    "test2990_D1": summ(Cfull.values())}

# IDF scope: score val preds with test-ref DF and combined DF, and vice versa
combined_refs = list(gv.values()) + list(gt.values())
df_comb = CF.compute_df([CF.cook_refs(r) for r in combined_refs]); rl_comb = math.log(len(combined_refs))
Ct_df = CF.compute_df([CF.cook_refs(r) for r in gt.values()]); rl_t = math.log(len(gt))
Cv_df = CF.compute_df([CF.cook_refs(r) for r in gv.values()]); rl_v = math.log(len(gv))

def cross(g, h, df, rl):
    out = {}
    for q in g:
        out[q] = CF.score_one(h[q], g[q], df, rl)
    return out

res["idf_scope"] = {
  "val_preds_valDF": summ(Cv.values()),
  "val_preds_testDF": summ(cross(gv, hv, Ct_df, rl_t).values()),
  "val_preds_combinedDF": summ(cross(gv, hv, df_comb, rl_comb).values()),
  "test_preds_testDF": summ(Ct.values()),
  "test_preds_valDF": summ(cross(gt, ht, Cv_df, rl_v).values()),
  "test_preds_combinedDF": summ(cross(gt, ht, df_comb, rl_comb).values()),
}

# reference structure
def rouge_l(a, b):
    x, y = a.split(), b.split()
    if not x or not y:
        return 0.0
    dp = [[0]*(len(y)+1) for _ in range(len(x)+1)]
    for i in range(1, len(x)+1):
        for j in range(1, len(y)+1):
            dp[i][j] = dp[i-1][j-1]+1 if x[i-1] == y[j-1] else max(dp[i-1][j], dp[i][j-1])
    l = dp[-1][-1]
    if l == 0: return 0.0
    p, r = l/len(x), l/len(y)
    return 2*p*r/(p+r)

def ref_structure(g, df, rl):
    nrefs = [len(r) for r in g]
    reflen = [statistics.mean(len(x.split()) for x in r) for r in g.values()]
    consensus = []
    ceiling_mean, ceiling_max = [], []
    for q, refs in g.items():
        pairs = [rouge_l(refs[i], refs[j]) for i in range(len(refs)) for j in range(i+1, len(refs))]
        consensus.append(statistics.mean(pairs))
        los = [CF.score_one(refs[i], refs[:i]+refs[i+1:], df, rl) for i in range(len(refs))]
        ceiling_mean.append(statistics.mean(los)); ceiling_max.append(max(los))
    return dict(n_refs_per_query=summ(nrefs), ref_len_tokens=summ(reflen),
                ref_pairwise_rouge_l=summ(consensus),
                loo_ref_cider_mean=summ(ceiling_mean), loo_ref_cider_max=summ(ceiling_max))

res["reference_structure"]["val497"] = ref_structure(gv, Cv_df, rl_v)
res["reference_structure"]["test2690"] = ref_structure(gt, Ct_df, rl_t)

# retrieval candidate compatibility (fixed combined DF)
def compat(g, df, rl, capfile, subset=None):
    d = json.load(open(os.path.join(ROOT, "external", capfile)))
    top1, top3, top7, top20max = [], [], [], []
    for q, refs in g.items():
        if q not in d: continue
        caps = d[q][0]
        if not caps: continue
        sc = sorted([CF.score_one(c, refs, df, rl) for c in caps], reverse=True)
        top1.append(sc[0]); top3.append(statistics.mean(sc[:3]))
        top7.append(statistics.mean(sc[:7])); top20max.append(sc[0])
    return dict(top1_best=summ(top1), top3_mean=summ(top3), top7_mean=summ(top7))

res["retrieval_compat"]["val497"] = compat(gv, df_comb, rl_comb, "external_val_d7_cap.json")
res["retrieval_compat"]["test2690"] = compat(gt, df_comb, rl_comb, "external_test_d7_cap.json")

json.dump(res, open(os.path.join(ROOT, "reports/cpu_audit_valtest_forensics.json"), "w"), indent=2)
print(json.dumps(res, indent=2))
