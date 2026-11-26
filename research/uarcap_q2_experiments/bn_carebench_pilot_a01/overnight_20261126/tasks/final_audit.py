#!/usr/bin/env python3
"""Final overnight audit: formal generation integrity, isolation, descriptive judge-bias
analysis, and a minimal human-review list. CPU-only; reads artifacts, writes summaries."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import re

BASE = Path(os.environ.get("OVERNIGHT_BASE", "/home/bn/uarcap_q2_runtime/overnight/20261126"))
O = Path("/home/bn/uarcap_q2_runtime/outputs")
FORMAL = O / "formal_candidate_unvalidated"
AUD = O / "final_audit"
AUD.mkdir(parents=True, exist_ok=True)


def utcnow(): return datetime.now(timezone.utc).isoformat()
def rj(p, d=None):
    try: return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception: return d if d is not None else {}
def rjl(p):
    try: return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]
    except Exception: return []
def norm(s):
    s = (s or "").lower(); s = re.sub(r"[^a-z0-9'\s]", " ", s)
    return " ".join(t for t in s.split() if t not in {"a","an","the"})
def sha_file(p):
    h = sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8*1024*1024), b""): h.update(c)
    return h.hexdigest()


def formal_integrity():
    cands = rjl(FORMAL/"formal_candidates.jsonl"); retr = {r["query_id"]: r for r in rjl(FORMAL/"formal_retrieval.jsonl")}
    frame_idx = {r["query_id"]: r for r in rjl(FORMAL/"formal_frame_indices.jsonl")}
    manifest = rj(FORMAL/"FORMAL_CAREBENCH_ANALYSIS_MANIFEST.json"); mrec = rj(FORMAL/"FORMAL_MANIFEST_RECEIPT.json")
    pilot = rj(O/"pilot_a1/pilot_a1_carebench_v1/pilot_a_excluded_carebench_vids.json")
    checks = {}
    def ck(name, ok, **d): checks[name] = {"ok": bool(ok), **d}
    qids = [c["query_id"] for c in cands]
    ck("candidate_count_600", len(cands) == 600 and len(set(qids)) == 600, n=len(cands))
    pilot_parents = set(pilot.get("excluded_parent_video_ids", []))
    sel_parents = [c.get("parent_video_id") for c in cands]
    ck("pilot_parents_excluded", not (set(sel_parents) & pilot_parents), overlap=sorted(set(sel_parents)&pilot_parents)[:5])
    val = set(manifest.get("e2_validation_set", {}).get("validation_query_ids", []))
    main = set(manifest.get("formal_main_excluding_validation_ids", []))
    ck("validation_disjoint_main", len(val) == 60 and not (val & main), n_val=len(val), overlap=len(val&main))
    ck("parent_isolation", bool(manifest.get("parent_isolation_verified")), value=manifest.get("parent_isolation_verified"))
    n_near = manifest.get("n_auto_near_dup_excluded", None)
    ck("near_duplicate_scan_executed", n_near is not None, n_auto_near_dup_excluded=n_near,
       n_eligible_pool=manifest.get("n_eligible_pool"), parent_dedup=manifest.get("parent_video_dedup_audit"))
    # decode
    dec_err = [q for q, r in frame_idx.items() if r.get("error") or len(r.get("frames", [])) != 16]
    ck("all_candidates_decoded_16_frames", len(frame_idx) == 600 and not dec_err, n_frames=len(frame_idx), errors=dec_err[:5])
    # features top7
    feat = Path("/home/bn/videoufo_research/features/videollama_faithful_v2")
    missing = []
    for q, r in retr.items():
        for c in (r.get("reranked_d") or [])[:7]:
            rep = (c.get("clip_ids") or [None])[0]
            if rep and not (feat/f"{rep}.pt").is_file(): missing.append(rep)
    ck("top7_features_present", len(missing) == 0, n_missing=len(set(missing)))
    # retrieval
    ret_bad = [q for q, r in retr.items() if len(r.get("original_top20", [])) < 7 or len(r.get("reranked_d", [])) < 7]
    ck("retrieval_top20_rerank7", len(retr) == 600 and not ret_bad, n=len(retr), bad=ret_bad[:5])

    model_reports = {}
    for model in ("videollama","qwen"):
        preds = rjl(FORMAL/model/"predictions.jsonl"); traces = rjl(FORMAL/model/"input_traces.jsonl")
        tmap = {(t.get("query_id"), t.get("arm")): t for t in traces}
        keys = [(p.get("query_id"), p.get("arm")) for p in preds]
        expected = {(q, a) for q in qids for a in ("ARM_A0","ARM_A1","ARM_A7")}
        ck(f"{model}_1800_unique", len(preds) == 1800 and set(keys) == expected and len(set(keys)) == 1800,
           n=len(preds), n_traces=len(traces))
        trace_bad = []
        a0_bad = []; prefix_bad = []; frame_bad = []; ckpt = set(); mids = set()
        by = defaultdict(dict)
        for p in preds:
            q, a = p.get("query_id"), p.get("arm")
            t = tmap.get((q, a)); ti = p.get("input_trace") or {}
            if not t or ti != t: trace_bad.append((q,a))
            if t:
                ckpt.add(t.get("checkpoint_sha256")); mids.add(t.get("model_id"))
                by[q][a] = t
                if len(t.get("frame_indices") or []) != 16: frame_bad.append((q,a))
                if a == "ARM_A0" and (t.get("text_evidence") or t.get("visual_evidence_ids") or t.get("visual_evidence_scores")):
                    a0_bad.append((q,a))
        for q, arms in by.items():
            if len(arms) != 3: continue
            t1, t7 = arms.get("ARM_A1",{}), arms.get("ARM_A7",{})
            if t1.get("text_evidence", [None])[:1] != t7.get("text_evidence", [])[:1]:
                prefix_bad.append(q+":text")
            if t7.get("visual_evidence_ids") and t1.get("visual_evidence_ids") and t7["visual_evidence_ids"][:1] != t1["visual_evidence_ids"]:
                prefix_bad.append(q+":visual")
            hs = {arms[a].get("visual_input_sha256") for a in ("ARM_A0","ARM_A1","ARM_A7")}
            if len(hs) != 1: frame_bad.append(q+":hash_drift")
        ck(f"{model}_trace_linkage", not trace_bad, n_bad=len(trace_bad), sample=trace_bad[:5])
        ck(f"{model}_a0_zero_evidence", not a0_bad, n_bad=len(a0_bad), sample=a0_bad[:5])
        ck(f"{model}_a1_prefix_a7", not prefix_bad, n_bad=len(prefix_bad), sample=prefix_bad[:5])
        ck(f"{model}_frames_16_and_arm_hash", not frame_bad, n_bad=len(frame_bad), sample=frame_bad[:5])
        ck(f"{model}_checkpoint_constant", len(ckpt) == 1 and len(mids) == 1, checkpoints=sorted(ckpt), model_ids=sorted(mids))
        model_reports[model] = {"n_predictions": len(preds), "checkpoint_sha256": sorted(ckpt), "model_ids": sorted(mids)}

    # artifact hashes
    arts = {}
    for rel in ("formal_candidates.jsonl","formal_retrieval.jsonl","formal_core_lock.json","FORMAL_CAREBENCH_ANALYSIS_MANIFEST.json",
                "videollama/predictions.jsonl","videollama/input_traces.jsonl","qwen/predictions.jsonl","qwen/input_traces.jsonl",
                "videollama_eval/a2_ref_fact_eval.jsonl","qwen_eval/a2_ref_fact_eval.jsonl"):
        p = FORMAL/rel
        arts[rel] = {"bytes": p.stat().st_size if p.is_file() else None, "sha256": sha_file(p) if p.is_file() else None}
    payload = {"schema": "uarcap_q2.formal_integrity_audit.v1", "generated_utc": utcnow(), "checks": checks,
               "all_passed": all(v["ok"] for v in checks.values()), "models": model_reports,
               "manifest_receipt": mrec, "artifacts": arts}
    (AUD/"FORMAL_INTEGRITY_AUDIT.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = ["# Formal Candidate Generation — Integrity Audit", "", f"- All checks passed: **{payload['all_passed']}**", "",
             "| Check | Result | Details |", "| --- | --- | --- |"]
    for k, v in checks.items(): lines.append(f"| `{k}` | {'PASS' if v['ok'] else 'FAIL'} | " + "; ".join(f"{a}={b}" for a,b in v.items() if a!='ok') + " |")
    lines += ["", "## Isolation note", "CaReBench uses anonymized FineAction parent IDs; YouTube-source independence cannot be proven beyond parent-level dedup/isolation. 883 eligible = 913 parent-deduped minus 30 pilot parents after the near-duplicate scan; see manifest audit fields.", ""]
    (AUD/"FORMAL_INTEGRITY_AUDIT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return payload


def judge_bias(model_dir, cross_dir):
    rows = rjl(model_dir/"a2_ref_fact_eval.jsonl"); cross = {(r["query_id"], r["norm_key"]): r for r in rjl(cross_dir/"A2_CROSS_JUDGE_LABELS.jsonl")}
    stats = {}
    for arm in ("ARM_A0","ARM_A1","ARM_A7"):
        ar = [r for r in rows if r["arm"] == arm and r.get("parse_status") == "OK"]
        gem = Counter(); llama = Counter(); n_claim = 0; dis = 0
        for r in ar:
            for c in r.get("claims", []):
                gem[c["label"]] += 1; n_claim += 1
                o = cross.get((r["query_id"], norm(c["claim"])))
                if o:
                    llama[o["llama_label"]] += 1
                    if o["llama_label"] != c["label"]: dis += 1
        n_l = sum(llama.values()) or 1
        stats[arm] = {"claims": n_claim, "gemma_label_counts": dict(gem), "llama_label_counts": dict(llama),
                      "disagreement_rate": round(dis/n_l,4), "n_cross": sum(llama.values()),
                      "delta_llama_minus_gemma": {l: round((llama[l]/n_l) - (gem[l]/max(1,n_claim)),4) for l in ("entailed","contradicted","unmentioned")}}
    return stats


def descriptive():
    out = {}
    specs = {
      "formal_videollama": (FORMAL/"videollama_eval", FORMAL/"videollama_machine_cross"),
      "formal_qwen": (FORMAL/"qwen_eval", FORMAL/"qwen_machine_cross"),
      "a3_qwen38": (O/"pilot_a3_exploratory/carebench_qwen38_eval", O/"pilot_a3_machine_cross_judge"),
      "a3_qwen35": (O/"pilot_a3_exploratory/carebench_qwen35_eval", O/"pilot_a3_machine_cross_judge_qwen35"),
    }
    for name, (md, cd) in specs.items():
        rows = rjl(md/"a2_ref_fact_eval.jsonl")
        per = {}
        for arm in ("ARM_A0","ARM_A1","ARM_A7"):
            ar = [r for r in rows if r["arm"] == arm and r.get("parse_status") == "OK"]
            n = sum(r["n_claims"] for r in ar)
            per[arm] = {"captions": sum(1 for r in rows if r["arm"]==arm), "claims": n,
                        "entailed": sum(r["n_entail_ref"] for r in ar), "contradicted": sum(r["n_contra_ref"] for r in ar),
                        "unmentioned": sum(r["n_unment_ref"] for r in ar),
                        "support_rate": round(sum(r["n_entail_ref"] for r in ar)/n,4) if n else None,
                        "conflict_rate": round(sum(r["n_contra_ref"] for r in ar)/n,4) if n else None,
                        "unmentioned_rate": round(sum(r["n_unment_ref"] for r in ar)/n,4) if n else None,
                        "core_coverage": round(sum(r["r_core_ref"] for r in ar if r.get("r_core_ref") is not None)/max(1,sum(1 for r in ar if r.get("r_core_ref") is not None)),4)}
        out[name] = {"per_arm": per, "judge_bias": judge_bias(md, cd) if (cd/"A2_CROSS_JUDGE_LABELS.jsonl").is_file() else {}}
    (AUD/"DESCRIPTIVE_RESULTS.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    lines = ["# Descriptive Automatic Evaluation (UNCALIBRATED)", ""]
    for name, v in out.items():
        lines += [f"## {name}", "", "| Arm | Claims | Entailed | Contradicted | Unmentioned | Support | Conflict | Unmentioned | Core | Gemma-Llama disagreement |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for arm, s in v["per_arm"].items():
            b = v["judge_bias"].get(arm, {})
            lines.append(f"| {arm} | {s['claims']} | {s['entailed']} | {s['contradicted']} | {s['unmentioned']} | {s['support_rate']} | {s['conflict_rate']} | {s['unmentioned_rate']} | {s['core_coverage']} | {b.get('disagreement_rate')} |")
        lines += ["", "Judge differential (Llama minus Gemma rate) by budget:", ""]
        for arm, b in v["judge_bias"].items():
            lines.append(f"- {arm}: {b.get('delta_llama_minus_gemma')}")
        lines.append("")
    lines += ["**No significance tests; descriptive only.**", ""]
    (AUD/"DESCRIPTIVE_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    return out


def human_review():
    items = []
    # A2 visual/machine review
    a2 = rj(O/"pilot_a2_machine/A2_HUMAN_REVIEW_REQUIRED.json");
    for x in a2.get("items", [])[:50]:
        items.append({"priority": "A2_machine_visual", "query_id": x.get("query_id"), "claim": x.get("claim"),
                      "evidence": x.get("model_opinions"), "frames": x.get("sampled_frames", [])[:3], "action": "choose final label"})
    # formal high-risk disagreements (entailed vs contradicted)
    for model, evdir, cross in (("formal_videollama", FORMAL/"videollama_eval", FORMAL/"videollama_machine_cross"),
                                 ("formal_qwen", FORMAL/"qwen_eval", FORMAL/"qwen_machine_cross")):
        rows = rjl(evdir/"a2_ref_fact_eval.jsonl"); cm = {(r["query_id"], r["norm_key"]): r for r in rjl(cross/"A2_CROSS_JUDGE_LABELS.jsonl")}
        seen = set(); n = 0
        for r in rows:
            for c in r.get("claims", []):
                if c["label"] not in ("entailed","contradicted"): continue
                o = cm.get((r["query_id"], norm(c["claim"])))
                if not o or o["llama_label"] not in ("entailed","contradicted") or o["llama_label"] == c["label"]: continue
                key = (r["query_id"], norm(c["claim"]))
                if key in seen: continue
                seen.add(key); n += 1
                if n > 25: break
                items.append({"priority": f"{model}_definite_disagreement", "query_id": r["query_id"], "claim": c["claim"],
                              "gemma": c["label"], "llama": o["llama_label"], "action": "choose final label"})
            if n > 25: break
    # random agreed audit sample
    rng = random.Random(20261126); rows = rjl((FORMAL/"qwen_eval")/"a2_ref_fact_eval.jsonl"); cm = {(r["query_id"], r["norm_key"]): r for r in rjl((FORMAL/"qwen_machine_cross")/"A2_CROSS_JUDGE_LABELS.jsonl")}
    pool = []
    for r in rows:
        for c in r.get("claims", []):
            o = cm.get((r["query_id"], norm(c["claim"])))
            if o and o["llama_label"] == c["label"]: pool.append((r["query_id"], c["claim"], c["label"]))
    for q, cl, l in rng.sample(pool, min(20, len(pool))):
        items.append({"priority": "random_agreed_audit", "query_id": q, "claim": cl, "gemma": l, "llama": l, "action": "verify algorithmically or visually"})
    e3 = rj(O/"pilot_e3_exploratory/E3_HUMAN_REVIEW_CANDIDATES.json", {"items": []})
    items += [{"priority": "E3_candidate", "query_id": x.get("query_id"), "true_action": x.get("true_action"),
               "conflict_action": x.get("conflict_action"), "action": "approve/reject candidate pair"} for x in e3.get("items", [])[:30]]
    payload = {"schema": "uarcap_q2.final_human_review.v1", "generated_utc": utcnow(), "n_items": len(items), "items": items}
    (AUD/"USER_REVIEW_TODO_FINAL.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = ["# Minimal Human Review TODO", "", f"- Items: {len(items)}", ""]
    for it in items[:80]:
        lines.append(f"- [{it['priority']}] `{it['query_id']}` — {it.get('claim') or it.get('true_action')} ({it.get('action')})")
    (AUD/"USER_REVIEW_TODO_FINAL.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    fi = formal_integrity(); de = descriptive(); hr = human_review()
    print(json.dumps({"formal_all_passed": fi["all_passed"], "n_formal_checks": len(fi["checks"]),
                      "n_review": hr["n_items"], "models": list(de)}, indent=2))
