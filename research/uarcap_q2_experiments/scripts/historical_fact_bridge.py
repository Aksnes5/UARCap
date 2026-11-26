"""Stage a source-independent historical MSR-VTT subset from the pinned A output ZIPs."""

import argparse
from hashlib import sha256
from pathlib import Path
import re
import sys
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from uarcap_q2.common.io_utils import atomic_write_json,atomic_write_jsonl,primitive_hash,read_json,sha256_file

METADATA_SHA="3540d91f9fd5f9a56d75b00c6576f30814fd19395d027791a6d438e9dc297014"
COHORT_SHA="fce00da1d10e6e80fa7c0ee9932042e4ab608946f2039887c830a778d96acae2"
ARCHIVE_SHA={"A0":"708c72502c64cc35ffeadd369055f417be880d2a3467046074a8804e274aa431",
             "A1":"04d5b317e166cacebeb941c950165f3b7b3c5293a8a915e0f887575e8d70bb49",
             "A7":"33bc167b258209e54f09498a34de20a695d5f79bbe1f2dd399d1b4849d457456"}
YT=re.compile(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})(?![A-Za-z0-9_-])")


def source_map(metadata):
    if sha256_file(metadata)!=METADATA_SHA:
        raise ValueError("Wrong public MSR-VTT metadata version")
    rows=read_json(metadata)["videos"]
    return {v["video_id"]:m.group(1) for v in rows if (m:=YT.search(v.get("url") or ""))}


def selected(ids,sources,n=60):
    used=set()
    picked=[]
    for q in sorted(ids,key=lambda q:primitive_hash([20261126,q])):
        sid=sources.get(q)
        if sid and sid not in used:
            used.add(sid)
            picked.append(q)
        if len(picked)==n:
            break
    if len(picked)!=n:
        raise ValueError("Insufficient distinct historical sources")
    return picked


def main():
    import json
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata",type=Path,required=True)
    p.add_argument("--test-ids",type=Path)
    p.add_argument("--raw-zips",type=Path)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--sources-only",action="store_true")
    a=p.parse_args()
    by_src=source_map(a.metadata)
    if a.sources_only:
        from uarcap_q2.common.io_utils import atomic_write_text
        atomic_write_text(a.out,"\n".join(sorted(set(by_src.values())))+"\n")
        return 0
    if sha256_file(a.test_ids)!=COHORT_SHA:
        raise ValueError("Wrong pilot-excluded MSR-VTT cohort")
    ids=[Path(r).stem for r in a.test_ids.read_text(encoding="utf-8-sig").splitlines()]
    cohort=selected(ids,by_src)
    references={}
    prediction=[]
    for arm,expected in ARCHIVE_SHA.items():
        path=a.raw_zips/(arm+".zip")
        if sha256_file(path)!=expected:
            raise ValueError("Archived prediction bundle SHA mismatch")
        with zipfile.ZipFile(path) as z:
            names=[n for n in z.namelist() if n.endswith("_generated_captions.json")]
            if len(names)!=1:
                raise ValueError("Unexpected historical prediction member")
            rows=json.loads(z.read(names[0]))
        keyed={Path(row["image_name"]).stem:row for row in rows}
        if len(keyed)!=len(rows):
            raise ValueError("Repeated archive query")
        for q in cohort:
            r=keyed[q]
            refs=r["captions"]
            if len(refs)!=20 or references.setdefault(q,refs)!=refs:
                raise ValueError("Reference list changed across arms")
            prediction.append({"query_id":q,"arm":"ARM_"+arm,"model_id":"videollama_evcap_frozen_v08",
                               "caption":r["prediction"],"ok":True,"source_id":by_src[q],"dataset_id":"MSR-VTT",
                               "execution_mode":"FROZEN_PREDICTION_REUSE"})
    # No CaReBench core-event denominator is defined on this historical benchmark.
    reference_rows=[{"query_id":q,"reference_caption":"\n".join(f"Reference {i}: {text}" for i,text in enumerate(references[q],1)),
                     "core_event_candidate":None} for q in cohort]
    value={"schema":"uarcap_q2.historical_bridge_manifest.v1","dataset_id":"MSR-VTT",
           "query_ids":cohort, "n_sources":len(cohort),"archive_sha256":ARCHIVE_SHA,
           "metadata_sha256":METADATA_SHA,"cohort_sha256":COHORT_SHA,
           "status":"EXPLORATORY_REFERENCE_CONFLICT_ONLY"}
    lock={"schema":"uarcap_q2.historical_reference_lock.v1","entries":reference_rows,
          "lock_sha256":primitive_hash(reference_rows)}
    atomic_write_json(a.out/"HISTORICAL_BRIDGE_MANIFEST.json",value)
    atomic_write_json(a.out/"historical_core_lock.json",lock)
    atomic_write_jsonl(a.out/"predictions.jsonl",prediction)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
