"""Windows CPU fixtures and replay of the committed small research artifacts."""

from hashlib import sha256
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"scripts"),str(ROOT/"deployment")]
from e3_reserve_pool import MINIMUM, POLICY, combine, make_pool, require_approval
from evaluator_stress import fixture_cases, score, summarize
from generate_visual_signatures import cache_signature, index_locations, top7_ids, video_file
from q2_controlled_sensitivity import analyze, check_length_revision, plan
from q2_human_portal import bind, build as build_portal, import_reviews
from q2_laptop_queue import build, retire_pending_v1, validate
from q2_release_report import collect
from q2_task_entry import accept
from q2_visual_risk_sensitivity import summarize as risk_summary
from source_and_visual_risk import risk
from uarcap_q2.common.io_utils import atomic_write_json, atomic_write_jsonl, primitive_hash, read_json, sha256_file
from uarcap_q2.generators.modern_vlm_adapter import hash_pil_frames
from uarcap_q2.generators.qwen3_vllm_adapter import Qwen3VLLMConfig, Qwen3VLLMMultimodalAdapter
from uarcap_q2.interventions.neutral_repair import V2
from uarcap_q2.retrieval.arm_builder import build_natural_retrieval_arm

ARCH=ROOT/"bn_carebench_pilot_a01/overnight_20261126_artifacts"


def test_reserve_uses_all_original_formal_and_pilot_sources(tmp_path):
    formal=[{"source_id":f"{i:011d}","query_id":f"formal_{i}"} for i in range(100,280)]
    ach=tmp_path/"ach.json"
    ids=["12345678901","22345678901","32345678901","42345678901","00000000100","00000000000","52345678901"]
    choices={"Choices":{"A":"Opens the red door","B":"Closes the red door"},"Correct Answer":"A","Question":"What happens?"}
    atomic_write_json(ach, {"group":{f"{sid}_clip_0":choices for sid in ids}|{"52345678901_clip_1":choices}})
    pilot=atomic_write_json(tmp_path/"pilot.json",{"excluded_source_ids":[f"{i:011d}" for i in range(30)]})
    formal_path=atomic_write_json(tmp_path/"formal.json",{"candidates":formal})
    clips=tmp_path/"clips.tsv"
    clips.write_text("clip_id\tsource_id\n12345678901.0\t12345678901\n",encoding="utf-8")
    debug=tmp_path/"debug.txt"
    debug.write_text("22345678901\n",encoding="utf-8")
    history=tmp_path/"history.txt"
    history.write_text("32345678901\n",encoding="utf-8")
    result=make_pool(ach,tmp_path,clips,pilot,history,formal_path,debug)
    assert {r["source_id"] for r in result["candidates"]}=={"42345678901","52345678901"}
    assert {x for row in result["excluded"] for x in row["reasons"]}=={
        "PILOT_SOURCE","FORMAL_SOURCE","DUPLICATE_SOURCE","VIDEOUFO_SOURCE","HISTORICAL_SOURCE","DEBUG_SOURCE"}
    assert len(result["input_sha256"])==6


def test_reserve_does_not_extend_unapproved_or_adjust_floor(tmp_path):
    original=[{"query_id":f"old{i}","source_id":f"old_s_{i}","human_verification":"PENDING"} for i in range(MINIMUM-1)]
    base={"schema":V2,"config_sha256":"x","approved":original,"rejected":[],"n_approved":len(original)}
    first=combine(base,None,None,0)
    assert first["status"]=="E3_INSUFFICIENT_VALID_CANDIDATES" and first["eligible"] is False
    entry={"query_id":"new","source_id":"new_source"}
    pool={"policy":POLICY,"candidates":[entry]}
    pool_path=atomic_write_json(tmp_path/"pool.json",pool)
    auth=atomic_write_json(tmp_path/"auth.json",{"approved":True,"approved_by":"test-fixture-only","protocol_revision":"fixture",
        "policy":POLICY,"pool_sha256":sha256_file(pool_path),"max_additional_candidates":1})
    cap=require_approval(auth,pool,pool_path)
    reserve={"schema":V2,"config_sha256":"x","approved":[{**entry,"original_candidate_sha256":primitive_hash(entry),
             "human_verification":"PENDING"}],"rejected":[]}
    assert combine(base,reserve,pool,cap)["eligible"] is True
    with pytest.raises(ValueError):
        combine(base,{**reserve,"config_sha256":"changed"},pool,cap)


def test_archived_design_real_qwen_lengths(tmp_path):
    result=plan(ARCH/"supplementary/LENGTH_CONTROL_PLAN.json",ARCH/"supplementary/VISUAL_SAMPLING_PLAN.json",
                ARCH/"formal/qwen/predictions.jsonl")
    assert result["word_range"]==[13,15] and result["length_baseline_n"]==360
    assert len(result["qwen_factorial_query_ids"])==40 and len(result["visual_nested_query_ids"])==60
    assert result["qwen_factorial_query_ids"]==result["query_ids"][:40]
    st=atomic_write_json(tmp_path/"runtime.json",{"tasks":{"q2l_v1_length_qwen":{"status":"PENDING","attempts":0}}})
    assert check_length_revision(result,st,tmp_path)["revision"]=="DETAILED_TO_UNIFORM_LENGTH"
    (tmp_path/"failures.jsonl").touch()
    with pytest.raises(ValueError,match="runtime artifacts"):
        check_length_revision(result,st,tmp_path)


def test_analysis_uses_difference_in_differences_and_dense_frame_parity():
    trace={"visual_input_sha256":"z","frame_indices":[0],"checkpoint_sha256":"m","decoding_config":{"seed":42}}
    row=lambda value: {"words":12,"n_entail_ref":0,"n_unment_ref":0,"n_contra_ref":value,
                       "r_core_ref":None,"llama_n_contra":None,"trace":trace}
    design={"visual_nested_query_ids":["q"]}
    base={("q","ARM_A0"):row(1),("q","ARM_A7"):row(5)}
    a0={("q","ARM_A0"):row(2)}
    a7={("q","ARM_A7"):row(3)}
    summary=analyze("visual32",design,base,[a0,a7])
    assert summary["interaction"]["n_contra_ref"]["mean"]==-3
    assert summary["interaction"]["llama_n_contra"]["status"]=="UNAVAILABLE"
    a7["q","ARM_A7"]["trace"]={**trace,"visual_input_sha256":"bad"}
    with pytest.raises(ValueError,match="input changed"):
        analyze("visual32",design,base,[a0,a7])


def test_stress_real_production_prompt_parser_then_joint_summary(tmp_path):
    cases=fixture_cases()
    assert len(cases)==196 and len({r["category"] for r in cases})==14
    nli={c["claim"]:c["expected"] for c in cases if "expected" in c}
    extraction={c["caption"]:c["canonical_atoms"] for c in cases if "caption" in c}
    class CPUTestFixture:
        def complete_batch(self,prompts):
            results=[]
            for system,user in prompts:
                if user.startswith("[TASK:EXTRACT"):
                    results.append(json.dumps({"claims":extraction[user.split("CAPTION: ")[1]]}))
                else:
                    claim=json.loads(user.split("CLAIMS_JSON: ")[1])[0]
                    if "brass" in claim:
                        results.append("cannot parse")
                    else:
                        results.append(json.dumps({"claims":[{"claim":claim,"label":nli[claim],"evidence":"test-fixture"}]}))
            return results
    a,b=(tmp_path/f"{m}.jsonl" for m in ("gemma","llama"))
    score(CPUTestFixture(),a,"test_fixture")
    score(CPUTestFixture(),a,"test_fixture")
    score(CPUTestFixture(),b,"test_fixture")
    panel=summarize({"gemma":a,"llama":b})
    assert panel["models"]["gemma"]["observed"]==196
    assert panel["models"]["gemma"]["parse_failure_count"]==len(panel["shared_errors"])==14
    assert panel["models"]["gemma"]["atomic_split_checks"]["canonical_exact"]==14


def make_cpu_video(path):
    import av
    import numpy as np
    rng=np.random.default_rng(7)
    with av.open(str(path),mode="w") as container:
        stream=container.add_stream("mpeg4",rate=12)
        stream.width=48
        stream.height=48
        stream.pix_fmt="yuv420p"
        for i in range(24):
            image=av.VideoFrame.from_ndarray(rng.integers(0,255,(48,48,3),dtype=np.uint8),format="rgb24")
            for packet in stream.encode(image):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_signature_archive_asset_and_cache(tmp_path,monkeypatch):
    video=tmp_path/"abcdefghijk.2.mp4"
    make_cpu_video(video)
    archive=tmp_path/"clips.zip"
    with ZipFile(archive,"w") as z:
        z.write(video,"shard/"+video.name)
    cache=tmp_path/"cache"
    a=cache_signature(video,None,cache)
    b=cache_signature(archive,"shard/"+video.name,cache)
    assert a["video_sha256"]==b["video_sha256"] and len(a["timestamps_seconds"])==16
    assert len(a["ordered_phash"])==16 and risk(a,b)["status"]=="CONFIRMED_DUPLICATE"
    from uarcap_q2.datasets import video_decoder
    monkeypatch.setattr(video_decoder,"decode_and_sample_video",lambda *a,**kw: pytest.fail("cache not reused"))
    assert cache_signature(video,None,cache)==a
    with pytest.raises(ValueError,match="Unsafe"):
        with video_file(archive,"../escape.mp4"):
            pass
    clips=tmp_path/"clips.tsv"
    clips.write_text("clip_id\tsource_id\nabcdefghijk.2\tabcdefghijk\n")
    retr=atomic_write_jsonl(tmp_path/"retrieval.jsonl",[{"query_id":"q","reranked_d":[{"clip_ids":["abcdefghijk.2"]}]*7}])
    inventory=atomic_write_json(tmp_path/"inventory.json",{"assets":[{"path":str(archive)}]})
    locations=tmp_path/"locations.jsonl"
    index_locations(clips,retr,inventory,locations)
    assert read_json(locations.with_suffix(".receipt.json"))["n_resolved"]==1


def test_near_duplicate_shift_and_unverified():
    from random import Random
    rng=Random(7)
    sig=[f"{rng.getrandbits(64):016x}" for _ in range(17)]
    dh=[f"{rng.getrandbits(64):016x}" for _ in range(16)]
    rows=[{"video_sha256":str(i),"ordered_dhash":dh if i else sig[:16],"ordered_phash":phash,
           "duration_seconds":10} for i,phash in enumerate((sig[:16],sig[1:]))]
    assert risk(*rows)["status"]=="HIGH_RISK_NEAR_DUPLICATE"
    assert risk(rows[0],{"status":"UNVERIFIED"})["status"]=="UNVERIFIED"


def test_visual_sensitivity_drops_only_prespecified_cases():
    units={q:{"parent_video_id":q} for q in ("q1","q2","q3","q4")}
    tags=["CONFIRMED_DUPLICATE","HIGH_RISK_NEAR_DUPLICATE","UNVERIFIED","NO_OBVIOUS_SIGNATURE_RISK"]
    risks={"rows":[{"query_id":q,"rank":i,"status":s if i==1 else tags[-1]}
                   for q,s in zip(units,tags) for i in range(1,8)]}
    result=risk_summary(risks, units, {"effect":{"q1":5,"q2":3,"q3":-1,"q4":2}})
    assert result["results"]["effect"]["exclude_confirmed_and_high"]["n_queries"]==2
    assert result["results"]["effect"]["exclude_confirmed_and_high"]["n_remaining_with_unverified_pair"]==1


def test_qwen_frame_roles_and_real_processor_proof_guard():
    from PIL import Image
    frames=[Image.new("RGB",(8,8),color=(i,0,0)) for i in range(16)]
    external=[Image.new("RGB",(8,8),color=(i,1,0)) for i in range(8)]
    packet=build_natural_retrieval_arm("test","ARM_A0",[])
    adapter=Qwen3VLLMMultimodalAdapter(Qwen3VLLMConfig(multimodal_input_key="image",external_frames=8))
    no=SimpleNamespace(packet=packet,frame_images=frames,external_frame_images=[])
    yes=SimpleNamespace(packet=packet,frame_images=frames,external_frame_images=external)
    no_msgs,_,no_hash=adapter.factorial_messages(no)
    yes_msgs,_,yes_hash=adapter.factorial_messages(yes)
    assert no_msgs[1]["content"][:18]==yes_msgs[1]["content"][:18] and no_hash!=yes_hash
    assert sum(x["type"]=="image_url" for x in yes_msgs[1]["content"])==24
    engine=SimpleNamespace(get_tokenizer=lambda: SimpleNamespace(convert_tokens_to_ids=lambda k:7 if k=="<|vision_start|>" else 8))
    assert adapter.verify_factorial_processor(engine,SimpleNamespace(prompt_token_ids=[7,8,8]*24),24)["processor_image_tokens"]==48
    with pytest.raises(ValueError,match="do not prove"):
        adapter.verify_factorial_processor(engine,SimpleNamespace(prompt_token_ids=[7,8]*16),24)


def test_human_portal_scope_and_independent_roundtrip(tmp_path):
    from PIL import Image
    from e3_repair import load_frames
    video = tmp_path/"video.mp4"
    video.write_bytes(b"fixture bytes, not an observation")
    frame_paths=[]
    for i in range(16):
        p=tmp_path/f"{i}.png"
        Image.new("RGB",(8,8),color=(i,0,0)).save(p)
        frame_paths.append(p)
    base={"module":"E3_CANDIDATE","query_id":"q","video_sha256":sha256_file(video),"target_id":"f_plus",
          "claim":"A opens the door.","artifact_key":"q"}
    whole={**base,"scope":"FULL_VIDEO","video_path":str(video)}
    frames={**base,"scope":"GENERATOR_FRAMES","frame_paths":[str(p) for p in frame_paths],
            "generator_frames_sha256":hash_pil_frames(load_frames(frame_paths))}
    assert bind(whole)[0]!=bind(frames)[0]
    assert "video_path" not in bind(frames)[1]
    caption={**base,"module":"E3_OUTPUT","scope":"CAPTION_EXPRESSION","caption":"A opens the door.",
             "caption_sha256":sha256(b"A opens the door.").hexdigest(),"model_opinion":"do not copy"}
    requests=atomic_write_json(tmp_path/"req.json",{"schema":"uarcap_q2.human_requests.v2","rows":[whole,frames,caption,caption]})
    build_portal([requests],["first","second"],b"test-fixture-secret-bytes".ljust(32,b"0"),tmp_path)
    key=tmp_path/"PRIVATE_PORTAL_KEY.json"
    values=[]
    for a in ("first","second"):
        bundle=read_json(tmp_path/f"blind_{a}.json")
        assert len(bundle["tasks"])==3 and all("model_opinion" not in t for t in bundle["tasks"])
        path=atomic_write_json(tmp_path/f"{a}.json",{"annotator_id":a,"records":[
            {"task_type":t["task_type"],"task_key":t["blind_task_id"],"completed":True,
             "scope_label":t["labels"][0],"scope_rationale":"Recorded CPU test fixture"} for t in bundle["tasks"]]})
        att=atomic_write_json(tmp_path/f"{a}.att.json",{"annotator_id":a,"review_export_sha256":sha256_file(path),
             "independent_blind_review_attested":True})
        values.append((path,att,{t["blind_task_id"] for t in bundle["tasks"]}))
    assert not (values[0][2]&values[1][2])
    out=tmp_path/"labels.json"
    import_reviews(key,[x[0] for x in values],[x[1] for x in values],None,out)
    assert read_json(out)["status"]=="SCOPED_REVIEW_COMPLETE"
    assert read_json(out)["candidate_visual_review"]["E3_CANDIDATE"]["q"]=="PENDING_INCOMPLETE"
    with pytest.raises(ValueError,match="independent"):
        import_reviews(key,[values[0][0]]*2,[values[0][1]]*2,None,out)


def test_v2_queue_partial_paths_and_install_preserves_non_v1_assets():
    queue=build("v2")
    tasks=queue["tasks"]
    validate(tasks)
    by={t["id"].removeprefix("q2l_v2_"):t for t in tasks}
    assert len(tasks)>len(build()["tasks"])==55
    assert "q2l_v2_visual64_a0_qwen_gemma" not in by["report_visual32_gemma"]["depends_on"]
    assert len(by["budget_report_qwen"]["depends_on"])==1
    assert all("/deploy/laptop-20261126-v2/" in t["argv"][2] for t in tasks)
    assert by["e3_reserve_continue"]["id"] not in by["budget_qwen"]["depends_on"]
    old=build()["tasks"]
    assert retire_pending_v1(old,{"tasks":{}})==([],old)
    with pytest.raises(ValueError):
        retire_pending_v1(old,{"tasks":{old[0]["id"]:{"status":"RUNNING","attempts":1}}})


def test_release_inventory_independent_when_other_task_is_partial(tmp_path):
    accepted=atomic_write_json(tmp_path/"accepted.json",{"schema":"complete","n_candidates":150})
    partial=atomic_write_json(tmp_path/"partial.json",{"schema":"partial","n_approved":147,"eligible":False})
    task=lambda name,p,s,**fields:{"id":name,"output":str(p),"science_gate":"fixture",
                                 "acceptance":{"kind":"json","schema":s,**fields}}
    queue=atomic_write_json(tmp_path/"queue.json",{"version":"fixture","tasks":[task("partial",partial,"partial",eligible=True),
                           task("good",accepted,"complete")]})
    out=tmp_path/"final"
    result=collect(queue,None,out)
    assert [r["status"] for r in result["tasks"]]==["PARTIAL_OR_REJECTED","ARTIFACT_ACCEPTED"]
    assert read_json(out/"ARCHIVE_MANIFEST.json")["files"][str(accepted)]["sha256"]==sha256_file(accepted)
    assert (out/"MANUSCRIPT_EVIDENCE.md").is_file()


def test_actual_top7_archive_denominator_and_missing_acceptance(tmp_path):
    from uarcap_q2.common.io_utils import read_jsonl
    grid=read_jsonl(ARCH/"formal/formal_retrieval.jsonl")
    assert len(grid)==600 and len(top7_ids(grid))<=4200
    p=atomic_write_jsonl(tmp_path/"rows.jsonl", [{"query_id":"q","arm":"ARM_A0"}])
    with pytest.raises(ValueError):
        accept({"kind":"rows","count":2,"keys":["query_id","arm"]},p)


def test_scoped_http_reviewers_cannot_use_other_session_and_media_seek(tmp_path):
    from threading import Thread
    from urllib.error import HTTPError
    from urllib.parse import urlencode
    from urllib.request import Request, urlopen
    from uarcap_q2.annotation.web_app import create_annotation_server
    video=tmp_path/"fixture.mp4"
    video.write_bytes(b"0123456789")
    token="a"*48
    key=atomic_write_json(tmp_path/"reviewer.json",{"annotator_id":"fixture","tasks":[{"video_path":str(video)}]})
    server,_=create_annotation_server(key,tmp_path,access_token=token)
    worker=Thread(target=server.serve_forever,daemon=True)
    worker.start()
    url=f"http://127.0.0.1:{server.server_port}"
    try:
        for cookie in ("","b"*48):
            with pytest.raises(HTTPError) as err:
                urlopen(Request(url+"/api/state",headers={"Cookie":"uarcap_review="+cookie}),timeout=2)
            assert err.value.code==403
        path=url+"/media?"+urlencode({"path":str(video)})
        with urlopen(Request(path,headers={"Cookie":"uarcap_review="+token,"Range":"bytes=2-4"}),timeout=2) as response:
            assert response.status==206 and response.read()==b"234"
    finally:
        server.shutdown()
        server.server_close()
