"""Verify mapped source overlap and screen exact/near visual identity using existing frame signatures."""

import argparse
from pathlib import Path
import re
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from e3_repair import youtube_sources, YT
from uarcap_q2.common.io_utils import atomic_write_json,read_json,read_jsonl,sha256_file


def source_audit(clips,pilot,formal,historical):
    vufo=youtube_sources(clips)
    pil=read_json(pilot)
    fo=read_json(formal)["approved"]
    historical_set={x.strip() for x in historical.read_text(encoding="utf-8").splitlines() if x.strip()} if historical else set()
    p={x for x in pil["excluded_source_ids"]}
    f={x["source_id"] for x in fo}
    if not all(YT.fullmatch(s) for s in f|p|historical_set) or len(f)!=len(fo):
        raise ValueError("Unknown or repeated VidHalluc source")
    exclusions=[{"query_id":r["query_id"],"source_id":r["source_id"],
                 "overlap":[n for n,ids in (("VideoUFO",vufo),("Pilot",p),("Historical",historical_set)) if r["source_id"] in ids]}
                for r in fo if r["source_id"] in vufo|p|historical_set]
    return {"schema":"uarcap_q2.source_risk.v1","video_ufo_sources":len(vufo),"pilot_sources":len(p),
            "formal_sources":len(f),"pilot_videoufo_overlap":len(p&vufo),"formal_exclusions":exclusions,
            "historical_overlap_status":"CHECKED_MAPPED_IDS" if historical_set else "UNVERIFIED_NO_HISTORICAL_MAPPING",
            "carebench_source_join":"NOT_JOINABLE","input_sha256":{x.name:sha256_file(x)
                for x in (clips,pilot,formal,historical) if x}}


def risk(query, retrieved):
    if not query or not retrieved or any(s.get("status") == "UNVERIFIED" for s in (query, retrieved)):
        return {"status":"UNVERIFIED","reason":"Missing visual signatures"}
    q_sha=query.get("video_sha256")
    if q_sha and q_sha==retrieved.get("video_sha256"):
        return {"status":"CONFIRMED_DUPLICATE","reason":"Identical video SHA256"}
    q,t=query.get("ordered_dhash"),retrieved.get("ordered_dhash")
    if not q or not t or len(q)!=len(t) or len(q)<8 or any(not re.fullmatch(r"[a-f0-9]{16}",x or "") for x in q+t):
        return {"status":"UNVERIFIED","reason":"Need aligned ordered dHash arrays"}
    def alignment(left, right, shift, limit):
        pairs = [(v, right[i+shift]) for i,v in enumerate(left) if 0<=i+shift<len(right)]
        return sum(bin(int(a,16)^int(b,16)).count("1")<=limit for a,b in pairs)/len(pairs)
    close=alignment(q,t,0,5)
    dq,dt=(x.get("duration_seconds") for x in (query,retrieved))
    duration_ratio=min(dq,dt)/max(dq,dt) if dq and dt else None
    if duration_ratio is None:
        return {"status":"UNVERIFIED","frame_close_fraction":close,"reason":"Need clip durations"}
    phash_matches = []
    for left_key, right_key in (("ordered_phash","ordered_phash"), ("ordered_center_phash","ordered_phash"),
                                ("ordered_phash","ordered_center_phash")):
        left, right = query.get(left_key, []), retrieved.get(right_key, [])
        if len(left) == len(right) == len(q) and all(re.fullmatch(r"[a-f0-9]{16}", x) for x in left+right):
            phash_matches.extend((alignment(left,right,shift,8), shift, left_key, right_key) for shift in range(-2,3))
    best = max(phash_matches, default=None)
    high=duration_ratio>=.8 and (close>=.75 or (best is not None and best[0]>=.85))
    return {"status":"HIGH_RISK_NEAR_DUPLICATE" if high else "NO_OBVIOUS_SIGNATURE_RISK",
            "frame_close_fraction":close,"duration_ratio":duration_ratio,"best_phash_alignment":best}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    ss=p.add_subparsers(dest="command",required=True)
    s=ss.add_parser("sources")
    for k in ("clips","pilot","formal","out"):
        s.add_argument("--"+k,type=Path,required=True)
    s.add_argument("--historical",type=Path)
    v=ss.add_parser("visual")
    for k in ("query-signatures","clip-signatures","retrieval","out"):
        v.add_argument("--"+k,type=Path,required=True)
    a=p.parse_args()
    if a.command=="sources":
        payload=source_audit(a.clips,a.pilot,a.formal,a.historical)
    else:
        query={r["query_id"]:r for r in read_jsonl(a.query_signatures)}
        clip={r["clip_id"]:r for r in read_jsonl(a.clip_signatures)}
        rows=[]
        for retrieval in read_jsonl(a.retrieval):
            q= retrieval["query_id"]
            for rank, item in enumerate(retrieval["reranked_d"][:7],1):
                cid=item["clip_ids"][0]
                rows.append({"query_id":q,"rank":rank,"clip_id":cid,**risk(query.get(q),clip.get(cid))})
        payload={"schema":"uarcap_q2.visual_risk.v1","rows":rows,
                 "input_sha256":{k:sha256_file(getattr(a,k)) for k in ("query_signatures","clip_signatures","retrieval")},
                 "interpretation":"dHash and pHash screen with small temporal offsets and center-crop comparisons"}
    atomic_write_json(a.out,payload)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
