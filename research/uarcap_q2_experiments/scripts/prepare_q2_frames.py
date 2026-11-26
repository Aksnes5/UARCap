"""Decode the already frozen Qwen sensitivity cohort on the CPU."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from uarcap_q2.common.io_utils import atomic_write_json,read_json,read_jsonl,sha256_file
from vlm_statement_verify import dense_frames


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ("plan","candidates","frame-root","out"):
        p.add_argument("--"+arg,type=Path,required=True)
    p.add_argument("--num-frames",type=int,choices=(32,64),required=True)
    a=p.parse_args()
    plan=read_json(a.plan)
    cohort=plan["query_ids"]
    candidates={r["query_id"]:r for r in read_jsonl(a.candidates)}
    if len(cohort)!=150 or len(set(cohort))!=150 or not set(cohort)<=candidates.keys() or plan["arm"]!="ARM_A7":
        raise ValueError("Frozen visual sensitivity cohort must be 150 distinct A7 queries")
    results={}
    for q in cohort:
        video=candidates[q]["video_path"]
        frames=dense_frames(video,a.frame_root/q,a.num_frames)
        results[q]={"video_path":video,**frames}
    atomic_write_json(a.out,{"schema":"uarcap_q2.qwen_sampling_decode.v1","num_frames":a.num_frames,
                             "plan_sha256":sha256_file(a.plan),"queries":results})
    return 0


if __name__=="__main__":
    raise SystemExit(main())
