#!/usr/bin/env python3
"""E2-Metric-Bridge: traditional caption metrics vs fact-level changes on the SAME 600
frozen CaReBench videos and the SAME 3,600 generated captions.

Post-hoc exploratory analysis added after initial results; it does not alter protocol
primary endpoints or human-calibration gates. Traditional metric values are real; fact
labels remain machine-assisted/uncalibrated.

Run inside the frozen repro_evcap venv (pycocoevalcap + Java SPICE available):
  /media/.../repro_evcap/venv/bin/python scripts/e2_metric_bridge.py
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
O = Path("/home/bn/uarcap_q2_runtime/outputs")
FORMAL = O / "formal_candidate_unvalidated"
OUT = O / "final_analysis/metric_bridge"


def utcnow(): return datetime.now(timezone.utc).isoformat()
def read_jsonl(p): return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def sha_file(p):
    h = sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8 * 1024 * 1024), b""): h.update(c)
    return h.hexdigest()


def yaml_check_core_short(candidates):
    rows = []
    for c in candidates:
        full = (c.get("reference_caption") or "").strip()
        short = (c.get("core_event_candidate") or "").strip()
        # R-Short must be a human-original opening span; exact (whitespace-normalized) substring check.
        norm_full = " ".join(full.split()); norm_short = " ".join(short.split())
        rows.append({"query_id": c["query_id"], "full": full, "short": short,
                     "short_is_human_substring": bool(short) and norm_short in norm_full,
                     "short_char_ratio": round(len(short) / max(1, len(full)), 4),
                     "short_words": len(short.split()), "full_words": len(full.split())})
    return rows


def compute_metrics(refs, preds, skip_spice=False):
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    if refs.keys()!=preds.keys() or not refs:
        raise ValueError("Metrics require identical nonempty reference and prediction IDs")
    qids = sorted(refs)
    refs = {q:refs[q] for q in qids}
    preds = {q:preds[q] for q in qids}
    res = {q: {} for q in qids}
    corpus = {}
    b4, bs = Bleu(4).compute_score(refs, preds)
    corpus.update({"BLEU_1": float(b4[0]), "BLEU_4": float(b4[3])})
    if len(bs)!=4 or any(len(column)!=len(qids) for column in bs):
        raise ValueError("BLEU scorer returned an incompatible score matrix")
    for q,b1,b4_item in zip(qids,bs[0],bs[3]):
        res[q]["BLEU_1"]=float(b1)
        res[q]["BLEU_4"]=float(b4_item)
    for key, scorer in (("METEOR", Meteor()), ("ROUGE_L", Rouge())):
        score, scores = scorer.compute_score(refs, preds)
        corpus[key] = float(score)
        if len(scores)!=len(qids):
            raise ValueError(f"{key} scorer returned wrong number of items")
        for q, s in zip(qids, scores): res[q][key] = float(s)
    score, scores = Cider().compute_score(refs, preds)
    corpus["CIDEr"] = float(score)
    if len(scores)!=len(qids):
        raise ValueError("CIDEr scorer returned wrong number of items")
    for q, s in zip(qids, scores): res[q]["CIDEr"] = float(s)
    corpus["SPICE"] = None; corpus["SPICE_status"] = "SKIPPED"
    if not skip_spice:
        try:
            t0 = time.time()
            score, scores = Spice_fn()(refs, preds)
            corpus["SPICE"] = float(score); corpus["SPICE_status"] = "OK"; corpus["SPICE_elapsed_sec"] = round(time.time()-t0, 1)
            for q, s in zip(qids, scores):
                val = s["All"]["f"] if isinstance(s, dict) and isinstance(s.get("All"), dict) else s
                res[q]["SPICE"] = float(val)
        except Exception as e:
            corpus["SPICE_status"] = f"BLOCKED: {type(e).__name__}: {str(e)[:300]}"
    return corpus, res


def Spice_fn():
    from pycocoevalcap.spice.spice import Spice
    return Spice().compute_score


def main():
    global FORMAL, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-spice", action="store_true")
    ap.add_argument("--formal", type=Path, default=FORMAL)
    ap.add_argument("--extension-root", type=Path)
    ap.add_argument("--model", choices=("videollama","qwen"))
    ap.add_argument("--out-dir", type=Path, required=True)
    args=ap.parse_args()
    FORMAL=args.formal
    OUT=args.out_dir
    if OUT.resolve()==(O/"final_analysis/metric_bridge").resolve():
        raise ValueError("Use versioned output for the repaired metric join")
    OUT.mkdir(parents=True,exist_ok=True)
    candidates = read_jsonl(FORMAL / "formal_candidates.jsonl")
    cand = {c["query_id"]: c for c in candidates}
    short_rows = yaml_check_core_short(candidates)
    valid_short = {r["query_id"] for r in short_rows if r["short_is_human_substring"] and r["short"]}
    ref_full = {q: c["reference_caption"] for q, c in cand.items()}
    ref_short = {q: c["core_event_candidate"] for q, c in cand.items() if q in valid_short}
    preds = {"formal_videollama": read_jsonl(FORMAL / "videollama/predictions.jsonl"),
             "formal_qwen": read_jsonl(FORMAL / "qwen/predictions.jsonl")}
    evals = {"formal_videollama": read_jsonl(FORMAL / "videollama_eval/a2_ref_fact_eval.jsonl"),
             "formal_qwen": read_jsonl(FORMAL / "qwen_eval/a2_ref_fact_eval.jsonl")}
    cross = {"formal_videollama": {(r["query_id"], r["norm_key"]): r for r in read_jsonl(FORMAL / "videollama_machine_cross/A2_CROSS_JUDGE_LABELS.jsonl")},
             "formal_qwen": {(r["query_id"], r["norm_key"]): r for r in read_jsonl(FORMAL / "qwen_machine_cross/A2_CROSS_JUDGE_LABELS.jsonl")}}

    if args.model:
        for table in (preds, evals, cross):
            for key in list(table):
                if key != "formal_" + args.model:
                    del table[key]
    if args.extension_root:
        for label, slug in (("formal_videollama","videollama"),("formal_qwen","qwen")):
            if label not in preds:
                continue
            folder=args.extension_root/slug
            preds[label]+=read_jsonl(folder/"predictions.jsonl")
            evals[label]+=read_jsonl(folder/"eval/a2_ref_fact_eval.jsonl")
            path=folder/"cross/A2_CROSS_JUDGE_LABELS.jsonl"
            if path.is_file():
                cross[label].update({(r["query_id"],r["norm_key"]):r for r in read_jsonl(path)})
    arms=("ARM_A0","ARM_A1","ARM_A3","ARM_A5","ARM_A7") if args.extension_root else ("ARM_A0","ARM_A1","ARM_A7")

    def fact_map(model):
        out = {}
        for r in evals[model]:
            if r.get("parse_status") != "OK": continue
            out[(r["query_id"], r["arm"])] = {"n_entail": r.get("n_entail_ref", 0), "n_contra": r.get("n_contra_ref", 0),
                                              "n_unment": r.get("n_unment_ref", 0), "n_claims": r.get("n_claims", 0),
                                              "r_core": r.get("r_core_ref"), "caption_words": len(str(r.get("caption", "")).split())}
        return out

    def llama_fact_map(model):
        out = {}
        for r in evals[model]:
            if r.get("parse_status") != "OK": continue
            q, arm = r["query_id"], r["arm"]; c = Counter()
            import re
            for cl in r.get("claims", []):
                key = " ".join(t for t in re.sub(r"[^a-z0-9'\s]", " ", cl["claim"].lower()).split() if t not in {"a","an","the"})
                o = cross[model].get((q, key))
                if o: c[o["llama_label"]] += 1
            if sum(c.values())==len(r["claims"]):
                out[(q, arm)] = {"n_entail": c["entailed"], "n_contra": c["contradicted"], "n_unment": c["unmentioned"]}
        return out

    all_rows = []
    summaries = []
    paired = {}
    for model, plist in preds.items():
        fmap = fact_map(model); lmap = llama_fact_map(model)
        by_arm = defaultdict(dict)
        for p in plist:
            if p["query_id"] in by_arm[p["arm"]] or not p.get("ok") or not p.get("caption"):
                raise ValueError("Duplicate or invalid prediction")
            by_arm[p["arm"]][p["query_id"]] = p
        for ref_mode, refs in (("R-Full", ref_full), ("R-Short", ref_short)):
            for arm in arms:
                if set(by_arm[arm])!=set(cand):
                    raise ValueError("Budget arm is missing a frozen query")
                pred_dict = {q: by_arm[arm][q]["caption"] for q in refs}
                corpus, per = compute_metrics({q: [refs[q]] for q in pred_dict}, {q: [pred_dict[q]] for q in pred_dict}, args.skip_spice)
                summaries.append({"model": model, "ref_mode": ref_mode, "arm": arm, "n_videos": len(pred_dict), **corpus, "CIDEr_points_x100":corpus["CIDEr"]*100})
                for q, sc in per.items():
                    f = fmap.get((q, arm)) or {}
                    l = lmap.get((q, arm)) or {}
                    all_rows.append({"model": model, "ref_mode": ref_mode, "query_id": q, "arm": arm,
                                     "caption": pred_dict[q], "reference": refs[q], "CIDEr_points_x100":sc["CIDEr"]*100,
                                     **{k: sc.get(k) for k in ("BLEU_1","BLEU_4","METEOR","ROUGE_L","CIDEr","SPICE")},
                                     **{k: f.get(k) for k in ("n_entail","n_contra","n_unment","n_claims","r_core","caption_words")},
                                     "llama_n_entail": l.get("n_entail"), "llama_n_contra": l.get("n_contra")})
        # paired changes at video level
        for ref_mode in ("R-Full", "R-Short"):
            idx = {(r["query_id"], r["arm"]): r for r in all_rows if r["model"] == model and r["ref_mode"] == ref_mode}
            for a, b in [*(("ARM_A0",a) for a in arms[1:]),*zip(arms[1:],arms[2:])]:
                qs = sorted({q for q, arm in idx if arm == a} & {q for q, arm in idx if arm == b})
                if not qs: continue
                def corr_field(kind, field):
                    pairs = [(idx[(q,b)][kind] - idx[(q,a)][kind], idx[(q,b)][field] - idx[(q,a)][field])
                             for q in qs if all(idx[q,z].get(key) is not None for z in (a,b) for key in (kind,field))]
                    if len(pairs) < 3 or len({z[0] for z in pairs})<2 or len({z[1] for z in pairs})<2: return None
                    try: return round(float(spearmanr([p[0] for p in pairs], [p[1] for p in pairs]).correlation), 4)
                    except Exception: return None
                judge_qs=[q for q in qs if all(idx[q,z].get(m) is not None for z in (a,b)
                                                for m in ("n_entail","n_contra","llama_n_entail","llama_n_contra"))]
                complete=[q for q in qs if all(idx[q,z].get(m) is not None for z in (a,b) for m in ("n_entail","n_contra"))]
                down_support_up = sum(1 for q in complete if idx[(q,b)]["CIDEr"] < idx[(q,a)]["CIDEr"] and (idx[(q,b)].get("n_entail",0) or 0) > (idx[(q,a)].get("n_entail",0) or 0))
                up_conflict_up = sum(1 for q in complete if idx[(q,b)]["CIDEr"] > idx[(q,a)]["CIDEr"] and (idx[(q,b)].get("n_contra",0) or 0) > (idx[(q,a)].get("n_contra",0) or 0))
                paired[f"{model}|{ref_mode}|{a}->{b}"] = {
                    "n": len(qs), "n_complete_facts":len(complete),"n_complete_dual_judge":len(judge_qs), "spearman_cider_vs_support": corr_field("CIDEr","n_entail"),
                    "spearman_cider_vs_conflict": corr_field("CIDEr","n_contra"),
                    "spearman_cider_vs_core": corr_field("CIDEr","r_core"),
                    "spearman_cider_vs_length": corr_field("CIDEr","caption_words"),
                    "cider_down_support_up_n": down_support_up,
                    "cider_down_support_up_rate": round(down_support_up/len(complete), 4) if complete else None,
                    "cider_up_conflict_up_n": up_conflict_up,
                    "cider_up_conflict_up_rate": round(up_conflict_up/len(complete), 4) if complete else None,
                    "judge_direction_consistency_support": round(sum(1 for q in judge_qs if np.sign(((idx[(q,b)].get("n_entail",0) or 0)-(idx[(q,a)].get("n_entail",0) or 0))) == np.sign(((idx[(q,b)].get("llama_n_entail",0) or 0)-(idx[(q,a)].get("llama_n_entail",0) or 0))))/len(judge_qs),4) if judge_qs else None,
                }
    # write per-video + summary + receipt
    with (OUT/"E2_METRIC_BRIDGE_PER_VIDEO.jsonl").open("w", encoding="utf-8") as f:
        for r in all_rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (OUT/"E2_METRIC_BRIDGE_SUMMARY.csv").open("w", newline="", encoding="utf-8") as f:
        keys = list(dict.fromkeys(k for row in summaries for k in row))
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(summaries)
    (OUT/"E2_METRIC_BRIDGE_PAIRED.json").write_text(json.dumps(paired, indent=2), encoding="utf-8")
    # figure
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8,max(3.5,len(paired)*.17)))
        labels = list(paired); vals = [np.nan if paired[k]["cider_down_support_up_rate"] is None else paired[k]["cider_down_support_up_rate"] for k in labels]
        ax.barh(range(len(labels)), vals); ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("P(CIDEr down & support facts up)"); ax.set_title("E2-Metric-Bridge paired changes (post-hoc)")
        for fmt in ("pdf","svg","png"): fig.savefig(OUT/f"fig_metric_bridge.{fmt}", bbox_inches="tight", dpi=200)
        plt.close(fig)
    except Exception as e:
        (OUT/"fig_metric_bridge.PENDING.txt").write_text(f"PENDING: {e}\n")
    # lengths / validity
    full_words = [r["full_words"] for r in short_rows if r["full"]]; short_words = [r["short_words"] for r in short_rows if r["short"]]
    ref_summary = {"n_candidates": len(candidates), "n_valid_short": len(valid_short),
                   "full_words_mean": round(float(np.mean(full_words)),2) if full_words else None,
                   "full_words_median": float(np.median(full_words)) if full_words else None,
                   "short_words_mean": round(float(np.mean(short_words)),2) if short_words else None,
                   "short_words_median": float(np.median(short_words)) if short_words else None,
                   "short_validity_rate": round(len(valid_short)/len(candidates),4) if candidates else None,
                   "invalid_short_examples": [r["query_id"] for r in short_rows if not r["short_is_human_substring"]][:10]}
    receipt = {"schema": "uarcap_q2.e2_metric_bridge.v1", "status": "POST_HOC_EXPLORATORY",
               "utc": utcnow(), "n_candidates": len(candidates), "n_predictions": len(all_rows),
               "models": list(preds), "reference_modes": ["R-Full","R-Short"], "reference_summary": ref_summary,
               "cider_idf": "single fixed reference set per mode and arm; IDF computed from the same 600 (or valid-short) references for all arms",
               "limitations": ["CaReBench is single-reference; MSR-VTT/MSVD are multi-reference", "absolute CIDEr scores across datasets are not comparable",
                                "traditional metrics are real; fact-level labels are machine-assisted and uncalibrated",
                                "post-hoc analysis added after initial results; protocol primary endpoints unchanged"],
               "spice_status": summaries[0].get("SPICE_status") if summaries else "NO_DATA",
               "artifacts": {}}
    for name in ("E2_METRIC_BRIDGE_PER_VIDEO.jsonl","E2_METRIC_BRIDGE_SUMMARY.csv","E2_METRIC_BRIDGE_PAIRED.json"):
        p = OUT/name; receipt["artifacts"][name] = {"bytes": p.stat().st_size, "sha256": sha_file(p)}
    (OUT/"E2_METRIC_BRIDGE_RECEIPT.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    md = ["# E2-Metric-Bridge (post-hoc exploratory)", "",
          f"- Generated: {utcnow()}", f"- Candidates: {len(candidates)}; predictions scored: {len(all_rows)}",
          f"- R-Short valid human substring rate: {ref_summary['short_validity_rate']} ({len(valid_short)}/{len(candidates)})",
          f"- Word length R-Full mean/median: {ref_summary['full_words_mean']} / {ref_summary['full_words_median']}; R-Short: {ref_summary['short_words_mean']} / {ref_summary['short_words_median']}",
          f"- SPICE: {receipt['spice_status']}", "",
          "Traditional metrics use one fixed reference cohort per mode across all arms. CaReBench is single-reference; absolute scores are NOT comparable to multi-reference MSR-VTT/MSVD.", "",
          "## Corpus summary", "", "| model | ref | arm | n | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for s in summaries: md.append(f"| {s['model']} | {s['ref_mode']} | {s['arm']} | {s['n_videos']} | {s.get('BLEU_1'):.4f} | {s.get('BLEU_4'):.4f} | {s.get('METEOR'):.4f} | {s.get('ROUGE_L'):.4f} | {s.get('CIDEr'):.4f} | {s.get('SPICE')} |")
    md += ["", "## Paired change relations (video-level, Spearman)", "", "| model | ref | transition | n | CIDEr~support | CIDEr~conflict | CIDEr~core | CIDEr~length | CIDEr↓ & support↑ | CIDEr↑ & conflict↑ | judge direction agreement |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for k, v in paired.items(): md.append(f"| {k.replace('|',' | ')} | {v['n']} | {v['spearman_cider_vs_support']} | {v['spearman_cider_vs_conflict']} | {v['spearman_cider_vs_core']} | {v['spearman_cider_vs_length']} | {v['cider_down_support_up_rate']} | {v['cider_up_conflict_up_rate']} | {v['judge_direction_consistency_support']} |")
    md += ["", "These are post-hoc descriptive associations; semantic rewrites, length and judge errors are plausible explanations. No causal claim; fact-level conclusions remain subject to human calibration.", ""]
    (OUT/"E2_METRIC_BRIDGE_REPORT.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"status":"POST_HOC_EXPLORATORY","n_predictions":len(all_rows),"spice":receipt["spice_status"],"out":str(OUT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
