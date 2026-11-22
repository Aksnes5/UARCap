"""CPU-only style / length / copying audit of frozen generation outputs."""
import json, os, re, sys, csv, statistics
from collections import defaultdict
import numpy as np
sys.path.insert(0, "/home/bn/videoufo_research/scripts/cpu_audit")
sys.path.insert(0, "/home/bn/videoufo_research/evcap_external/evaluation")
import cider_fixed as CF
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

ROOT = "/home/bn/videoufo_research"
OUT = os.path.join(ROOT, "evcap_external/outputs")
NAME = "eval_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review_generated_captions.json"
NAMEM = "eval_evcap_top3_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_msvd_generated_captions.json"

WORD = re.compile(r"[a-z0-9']+")
def words(s): return WORD.findall(s.lower())
def ngrams(ws, n): return [tuple(ws[i:i+n]) for i in range(len(ws)-n+1)]

def rouge_l(a, b):
    x, y = words(a), words(b)
    if not x or not y: return 0.0
    dp = [[0]*(len(y)+1) for _ in range(len(x)+1)]
    for i in range(1, len(x)+1):
        for j in range(1, len(y)+1):
            dp[i][j] = dp[i-1][j-1]+1 if x[i-1] == y[j-1] else max(dp[i-1][j], dp[i][j-1])
    l = dp[-1][-1]
    if l == 0: return 0.0
    p, r = l/len(x), l/len(y)
    return 2*p*r/(p+r)

def qstats(xs):
    xs = sorted(float(x) for x in xs); n = len(xs)
    def q(p): return xs[min(n-1, int(p*n))]
    return dict(n=n, mean=statistics.mean(xs), median=statistics.median(xs),
                sd=statistics.pstdev(xs), q10=q(.10), q25=q(.25), q75=q(.75), q90=q(.90),
                iqr=q(.75)-q(.25), min=xs[0], max=xs[-1])

def spearman(a,b):
    ra=np.argsort(np.argsort(a)); rb=np.argsort(np.argsort(b))
    return float(np.corrcoef(ra,rb)[0,1])

def per_query(preds, ev):
    P = {}
    for q,p in preds.items():
        supplied = ev.get(q, [])
        top1 = supplied[0] if supplied else ""
        utok=set(); u_ng={n:set() for n in (1,2,3,4)}
        for c in supplied:
            w=words(c); utok.update(w)
            for n in (1,2,3,4): u_ng[n].update(ngrams(w,n))
        pw=words(p); pset=set(pw); t1=set(words(top1))
        d=dict(wc=len(p.split()), ptb_tok=len(pw),
               ov_top1=len(pset&t1)/max(1,len(pset)),
               jac_top1=len(pset&t1)/max(1,len(pset|t1)),
               ov_union=len(pset&utok)/max(1,len(pset)),
               jac_union=len(pset&utok)/max(1,len(pset|utok)),
               rouge_top1=rouge_l(p,top1),
               rouge_max=max([rouge_l(p,c) for c in supplied] or [0.0]),
               exact_copy=float(any(p.lower().strip(". ")==c.lower().strip(". ") for c in supplied)))
        for n in (1,2,3,4):
            png=ngrams(pw,n); copied=sum(1 for g in png if g in u_ng[n])
            d[f"copy{n}_tok"]=copied/max(1,len(png))
            types=set(png); d[f"copy{n}_type"]=sum(1 for g in types if g in u_ng[n])/max(1,len(types))
        for n in (3,4):
            d[f"has_copy{n}"]=float(any(g in u_ng[n] for g in ngrams(pw,n)))
        P[q]=d
    return P

def main():
    datasets = {
        "MSR-VTT-test2690": ("eval_test2690_D{k}", NAME, os.path.join(ROOT,"external/external_test_d7_cap.json")),
        "MSVD670":          ("eval_msvd_D{k}",     NAMEM, os.path.join(ROOT,"external/external_msvd_d7_cap.json")),
        "MSR-VTT-val497":   ("eval_val497_D{k}",   NAME, os.path.join(ROOT,"external/external_val_d7_cap.json")),
    }
    res={}; rows_csv=[]
    for dsname,(pat,nm,capf) in datasets.items():
        capall=json.load(open(capf)); res[dsname]={}
        for k in (1,3,5,7):
            d=os.path.join(OUT,pat.format(k=k))
            rows=json.load(open(os.path.join(d,nm)))
            preds={r["image_name"]:r["prediction"].strip() for r in rows}
            refs={r["image_name"]:r["captions"] for r in rows}
            ev={q:capall[q][0][:k] for q in preds if q in capall}
            PQ=per_query(preds,ev)
            m={key:qstats([PQ[q][key] for q in PQ]) for key in PQ[next(iter(PQ))]}
            # per-query CIDEr
            tok=PTBTokenizer()
            gt=tok.tokenize({q:[{"caption":c} for c in refs[q]] for q in preds})
            hp=tok.tokenize({q:[{"caption":preds[q]}] for q in preds}); hp={q:v[0] for q,v in hp.items()}
            C=CF.corpus(gt,hp); m["cider_per_query"]=qstats([C[q] for q in preds])
            cid=np.array([C[q] for q in preds]); order=sorted(preds)
            corr={}
            for key in ["wc","ptb_tok","ov_union","jac_union","rouge_top1","copy3_tok","copy4_tok","has_copy4","exact_copy"]:
                corr[key+"_vs_cider_spearman"]=spearman(np.array([PQ[q][key] for q in order]),cid)
            m["corr"]=corr
            met=json.load(open(os.path.join(d,"metrics.json"))); m["frozen"]=met
            res[dsname][f"D{k}"]=m
            rows_csv.append(dict(dataset=dsname,condition=f"D{k}",k=k,n=m["wc"]["n"],
                wc_mean=round(m["wc"]["mean"],4),wc_median=m["wc"]["median"],wc_sd=round(m["wc"]["sd"],4),
                wc_q25=m["wc"]["q25"],wc_q75=m["wc"]["q75"],wc_iqr=m["wc"]["iqr"],
                ptb_tok_mean=round(m["ptb_tok"]["mean"],4),
                ov_top1=round(m["ov_top1"]["mean"],4),jac_top1=round(m["jac_top1"]["mean"],4),
                ov_union=round(m["ov_union"]["mean"],4),jac_union=round(m["jac_union"]["mean"],4),
                rouge_top1=round(m["rouge_top1"]["mean"],4),rouge_max=round(m["rouge_max"]["mean"],4),
                exact_copy=round(m["exact_copy"]["mean"],4),
                copy1_tok=round(m["copy1_tok"]["mean"],4),copy2_tok=round(m["copy2_tok"]["mean"],4),
                copy3_tok=round(m["copy3_tok"]["mean"],4),copy4_tok=round(m["copy4_tok"]["mean"],4),
                has_copy3=round(m["has_copy3"]["mean"],4),has_copy4=round(m["has_copy4"]["mean"],4),
                cider=round(met["CIDEr"]*100,4),spice=round(met["SPICE"]*100,4),
                corr_len_cider=round(m["corr"]["wc_vs_cider_spearman"],4),
                corr_copy3_cider=round(m["corr"]["copy3_tok_vs_cider_spearman"],4),
                corr_ov_cider=round(m["corr"]["ov_union_vs_cider_spearman"],4)))
    json.dump(res,open(os.path.join(ROOT,"reports/style_length_copying_full.json"),"w"),indent=2)
    with open(os.path.join(ROOT,"reports/style_length_copying_summary.csv"),"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows_csv[0])); w.writeheader(); w.writerows(rows_csv)
    for ds in res:
        print("==",ds)
        for c in ["D1","D3","D5","D7"]:
            m=res[ds][c]
            print(" %s wc %.2f med %.1f iqr %.1f | ov_u %.3f jac_u %.3f rl_t1 %.3f | copy3 %.3f has4 %.3f exact %.3f | CIDEr %.2f SPICE %.2f | corr_len %.3f copy3 %.3f" % (
                c,m["wc"]["mean"],m["wc"]["median"],m["wc"]["iqr"],m["ov_union"]["mean"],m["jac_union"]["mean"],
                m["rouge_top1"]["mean"],m["copy3_tok"]["mean"],m["has_copy4"]["mean"],m["exact_copy"]["mean"],
                m["frozen"]["CIDEr"]*100,m["frozen"]["SPICE"]*100,
                m["corr"]["wc_vs_cider_spearman"],m["corr"]["copy3_tok_vs_cider_spearman"]))

if __name__=="__main__": main()
