"""Windows CPU contracts. Fixture labels do not represent scientific evidence."""

from copy import deepcopy
import json
from pathlib import Path
import sys

from PIL import Image
import pytest

from uarcap_q2.annotation.human_endpoints import verify
from uarcap_q2.common.io_utils import primitive_hash,read_json,read_jsonl,sha256_bytes
from uarcap_q2.final_analysis.core import cross_model_budget_differential,holm,metric_by_qid_arm
from uarcap_q2.final_analysis.validation_identity import audit, require_cohort, verify_caption_pair
from uarcap_q2.interventions.e3_output import ARMS, can_reuse,collect_generation
from uarcap_q2.interventions.neutral_repair import VISUAL,freeze,parse_expressed,parse_visual,proposals,statement_job
from uarcap_q2.retrieval.arm_builder import (build_evcap_modality_arm,build_natural_retrieval_arm,
                                              verify_natural_prefix_consistency)
from uarcap_q2.retrieval.reranker import RetrievalCandidate

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"scripts"),str(ROOT/"deployment")]
from e3_repair import youtube_sources
from fact_provenance import parse_relations
from q2_laptop_queue import build,merge,validate
from q2_task_entry import accept
from source_and_visual_risk import risk
from vlm_statement_verify import messages

ARCH=ROOT/"bn_carebench_pilot_a01/overnight_20261126_artifacts"
SAMPLES=ROOT/"handoff_20261126/samples"


def test_frozen_full_cohort_identity():
    f=ARCH/"formal"
    result=audit(f/"FORMAL_CAREBENCH_ANALYSIS_MANIFEST.json",
                 [f/"videollama/predictions.jsonl",f/"qwen/predictions.jsonl"],
                 [f/"eval_old/a2_ref_fact_eval.jsonl",f/"eval_qwen/a2_ref_fact_eval.jsonl"],SAMPLES/"VALIDATION60_RECEIPT.json")
    assert result["sets"]["validation"]["queries"]==60
    assert all(v["validation"]==180 for v in result["artifacts"].values())
    assert set(result["intersections"].values())=={0}


def test_caption_join_rejects_wrong_experimental_arm():
    pred=read_jsonl(SAMPLES/"formal_predictions_sample.jsonl")[0]
    evaluated={"model_id":pred["model_id"],"query_id":pred["query_id"],"arm":"ARM_A5",
               "caption":pred["caption"],"caption_sha256":sha256_bytes(pred["caption"].encode()),"parse_status":"OK"}
    with pytest.raises(ValueError,match="mismatch"):
        verify_caption_pair(pred,evaluated)


def test_parent_and_query_leakage():
    data=read_json(ARCH/"formal/FORMAL_CAREBENCH_ANALYSIS_MANIFEST.json")
    main,val=data["formal_main_excluding_validation_ids"],data["e2_validation_set"]["validation_query_ids"]
    wrong=deepcopy(data)
    wrong["formal_main_excluding_validation_ids"].append(val[0])
    with pytest.raises(ValueError,match="overlap"):
        require_cohort(wrong)
    data["units"][main[0]]["parent_video_id"]=data["units"][val[0]]["parent_video_id"]
    with pytest.raises(ValueError,match="Parent repeats"):
        require_cohort(data)


def test_human_template_rejected():
    d=read_json(ARCH/"formal/FORMAL_CAREBENCH_ANALYSIS_MANIFEST.json")
    with pytest.raises(ValueError,match="genuine"):
        verify(SAMPLES/"HUMAN_ENDPOINT_IMPORT_TEMPLATE.json",d)


def test_prefix_contract_on_actual_retrieval():
    row=read_jsonl(ARCH/"formal/formal_retrieval.jsonl")[0]
    cands=[RetrievalCandidate(**x) for x in row["reranked_d"]]
    packets={f"ARM_A{k}":build_natural_retrieval_arm(row["query_id"],f"ARM_A{k}",cands) for k in (0,1,3,5,7)}
    verify_natural_prefix_consistency(packets)
    assert packets["ARM_A5"].text_topn==5 and packets["ARM_A5"].visual_topn==5
    assert packets["ARM_A0"].evcap_cap_entry==[[],[],[]]
    ablated=build_evcap_modality_arm(row["query_id"],"T1V7",cands)
    assert ablated.text_evidence==packets["ARM_A1"].text_evidence
    assert ablated.visual_scores==packets["ARM_A7"].visual_scores
    assert len(ablated.evcap_cap_entry[0])==7
    assert not build_evcap_modality_arm(row["query_id"],"T7V0",cands).visual_clip_ids


def test_candidate_determinism_and_real_length_gating():
    orig=read_json(ARCH/"vidhalluc/E3_FORMAL_APPROVED.json")
    cand=orig["approved"][0]
    assert not cand.get("neutral_text")
    tokenizers=(lambda s:len(s.split()),lambda s:len(s.split())+1)
    opts=proposals(cand,tokenizers)
    assert opts and opts==proposals(cand,tokenizers)
    assert not proposals(cand,(lambda s:len(s)*10,lambda s:len(s)*10))


@pytest.mark.parametrize("raw",['{"status":"AFFIRMED"}','{"status":"ABSENT","expressed":false,"evidence":""}',
                              '{"status":"NEGATED","expressed":1,"evidence":"not present"}',
                              '{"status":"VAGUE","expressed":0,"evidence":""}'])
def test_e3_parse_failure_never_zero(raw):
    with pytest.raises(ValueError):
        parse_expressed(raw)


def test_e3_unverifiable_explicit_missing_and_negative():
    assert parse_expressed('{"status":"UNVERIFIABLE","expressed":null,"evidence":""}')["expressed"] is None
    assert parse_expressed('{"status":"NEGATED","expressed":0,"evidence":"not running"}')["expressed"]==0


def test_single_statement_prompt_hides_role():
    real=read_json(SAMPLES/"E3_BRIDGE_VERIFICATION_JOBS_SAMPLE.json")["jobs"][0]
    prompt=messages(real["statement"],[Image.new("RGB",(4,4),"white")])
    user=prompt[0]["content"][-1]["text"]
    assert real["statement"] in user and real["role"] not in user and real["job_id"] not in user
    with pytest.raises(ValueError,match="Invalid evidence"):
        parse_visual('{"verdict":"entailed","confidence":"high","evidence_frames":[16],"rationale":"item"}',16)


def screened_fixture(bridge):
    candidate={"query_id":"test","true_action":"sorting papers",
               "conflict_action":"shredding papers","video_path":"fixture"}
    spec=["sorting papers","shredding papers","sitting upright"]
    jobs=[statement_job("test",s,"fixture","fixture","f"*64) for s in spec]
    if bridge:
        candidate.update(f_plus=spec[0],f_minus=spec[1])
    old={"candidates" if bridge else "approved":[candidate]}
    text={"natural":True,"same_form":True,"pair_single_target":True,"implies_target":False,
          "negates_target":False,"presupposes_target":False,"rationale":"CPU fixture"}
    plan={"original_sha256":primitive_hash(old),"config_sha256":"a"*64,
          "candidates":[{"candidate":candidate,"source_verified":True,"independence_group":"source",
                         "targets":jobs[:2],"proposals":[{"job":jobs[2],"text_screen_pass":True,
                          "text_screen":{"schema":"uarcap_q2.e3_neutral_text_screen.v2","parse_status":"OK","raw":json.dumps(text)}}]}]}
    verdicts=[{"schema":VISUAL,"job_id":j["job_id"],"job_sha256":primitive_hash(j),"parse_status":"OK",
               "frames_sha256":"f"*64,"dense_frames_sha256":"d"*64,"n_frames":16,"n_dense_frames":64,
               "video_truth":v,"visible_in_16_frames":"yes","confidence":"high"}
              for j,v in zip(jobs,("entailed","contradicted","entailed"),strict=True)]
    return old,plan,verdicts


@pytest.mark.parametrize("bridge,floor",[(False,150),(True,30)])
def test_e3_freeze_floors_and_true_neutral(bridge,floor):
    old,plan,rows=screened_fixture(bridge)
    result=freeze(old,plan,rows,bridge=bridge)
    assert result["n_approved"]==1 and result["minimum"]==floor and not result["eligible"]
    rows[2]["video_truth"]="contradicted"
    assert freeze(old,plan,rows,bridge=bridge)["n_approved"]==0
    assert freeze(old,plan,[],bridge=bridge)["n_approved"]==0


def test_legacy_neutral_and_incomplete_pairs_rejected():
    assert not can_reuse({"arm":"E3_NEU"}, {}, {}, "")
    candidate={"query_id":"test"}
    with pytest.raises(ValueError,match="Incomplete"):
        collect_generation([], [candidate], "model", False)


def test_semantic_guard_and_conservation():
    v={"equivalent_pairs":[[0,0]],"replacement_pairs":[[1,1]],"uncertain_pairs":[],
       "evidence_labels":[["unrelated"],["contradicted"],["entailed"]]}
    before=["A man chops wood","A red car"]
    after=["A man is chopping wood","A blue car","A dog barks"]
    r=parse_relations(json.dumps(v),before,after,1)
    assert r["added_indices"]==[2] and not r["removed_indices"]
    after[0]="A man does not chop wood"
    with pytest.raises(ValueError,match="polarity"):
        parse_relations(json.dumps(v),before,after,1)


def test_paired_difference_and_missing_unchanged_by_order():
    def rows(a,b):
        return [{"model_id":"m","query_id":"q","arm":f"ARM_A{n}","parse_status":"OK",
                 "n_contra_ref":v,"n_claims":4} for n,v in ((0,a),(1,None),(7,b))]
    a,b=rows(2,3),rows(1,4)
    f=cross_model_budget_differential(a,b,"n_contra_ref")
    assert f==cross_model_budget_differential(list(reversed(a)),list(reversed(b)),"n_contra_ref")
    assert f["ARM_A0->ARM_A7"]["observed_mean"]==2
    assert metric_by_qid_arm(a,"n_contra_ref").get("ARM_A1")==None
    assert holm({"a":.01,"b":.02,"c":.5})["a"]["p_holm"]==.03


def test_official_sources_and_perceptual_risk(tmp_path):
    p=tmp_path/"clips.tsv"
    p.write_text("clip_id\tsource_id\ttopic\nabcdefghijk.2\tabcdefghijk\twood\n")
    assert youtube_sources(p)=={"abcdefghijk"}
    assert risk({}, {})["status"]=="UNVERIFIED"
    assert risk({"video_sha256":"x"},{"video_sha256":"x"})["status"]=="CONFIRMED_DUPLICATE"
    sig={"ordered_dhash":["ffffffffffffffff"]*8,"duration_seconds":10}
    assert risk(sig,sig)["status"]=="HIGH_RISK_NEAR_DUPLICATE"


def test_queue_matches_live_schema_and_smoke_hash():
    q=build()
    validate(q["tasks"])
    old=read_json(ROOT/"bn_carebench_pilot_a01/overnight_20261126/task_queue.json")
    combined=merge(old["tasks"],q["tasks"])
    assert combined[:len(old["tasks"])]==old["tasks"]
    assert combined==merge(combined,q["tasks"])
    with pytest.raises(ValueError):
        validate([q["tasks"][0],q["tasks"][0]])
    with pytest.raises(ValueError):
        validate([{**q["tasks"][0],"depends_on":["unknown"]}])
    by={r["id"]:r for r in q["tasks"]}
    for m in ("videollama","qwen"):
        a,b=by[f"q2l_v1_budget_{m}_smoke"],by[f"q2l_v1_budget_{m}"]
        assert a["argv"][-1]==b["argv"][-1]


def test_success_requires_complete_outputs(tmp_path):
    p=tmp_path/"result.jsonl"
    p.write_text('{"query_id":"q","arm":"ARM_A1"}\n',encoding="utf8")
    with pytest.raises(ValueError,match="count"):
        accept({"kind":"rows","count":2,"keys":["query_id","arm"]},p)
