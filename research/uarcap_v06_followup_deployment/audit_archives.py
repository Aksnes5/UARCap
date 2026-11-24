"""Read pinned evidence from Git and repeat frozen input preparation on Windows."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys


HERE = Path(__file__).resolve().parent
DESIGN = HERE.parent / 'uarcap_v06_followup_experiments'
sys.path.insert(0, str(DESIGN))
from launch_pinned import ARMS, argv_for, actual_vision
from prepare_inputs import primitive_hash, run

BASE = '52b5ed3a69e4b84a8e16ed747f141eefb5b1e2ec'
INPUTS = {'top20':'external/external_test_top20.json', 'cap':'external/external_test_d7_cap.json',
          'sims':'external/external_test_d7_sims.json','full_ids':'manifests/msrvtt_test2990.tsv',
          'test_ids':'manifests/msrvtt_untouched2690.tsv'}
PRED_BASENAME = 'eval_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review_generated_captions.json'
CHECKPOINT = 'train_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review.pt'
RUNTIME_DEPENDENCIES = ('common/dist_utils.py', 'dataset/msrvtt/vl_dataloader.py',
    'dataset/msrvtt/data_utils/tsv_file.py', 'dataset/msrvtt/data_utils/miscellaneous.py',
    'models/Qformer.py', 'models/blip2.py', 'models/base_model.py', 'models/eva_vit.py', 'models/modeling_llama.py')


def sha(data):
    return sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def verify(condition, explanation):
    if not condition:
        raise ValueError(explanation)


def inspect(repo, scratch):
    command = ['git','-C',str(repo)]
    def blob(path):
        return subprocess.check_output([*command,'cat-file','blob',f'{BASE}:{path}'])
    def receipt(data):
        return {'bytes':len(data),'sha256':sha(data)}

    scratch.mkdir(parents=True, exist_ok=False)
    frozen = read(DESIGN/'CPU_PREFLIGHT_SUMMARY.json')
    binding = read(DESIGN/'FROZEN_SOURCE_BINDINGS.json')
    inputs = {k:blob(path) for k,path in INPUTS.items()}
    verify({k:sha(data) for k,data in inputs.items()}==frozen['source_sha256'], 'Frozen source hash differs')
    for k,data in inputs.items():
        (scratch/Path(INPUTS[k]).name).write_bytes(data)
    full = inputs['full_ids'].decode().splitlines()
    eval_ids = inputs['test_ids'].decode().splitlines()
    verify(len(set(full))==2990 and len(set(eval_ids))==2690 and set(eval_ids)<=set(full), 'Invalid frozen queries')
    pilot = set(blob('dev300_queries.txt').decode().splitlines())
    verify([q for q in full if Path(q).stem not in pilot]==eval_ids, 'Pilot-filter order differs')
    top,cap,sims = (json.loads(inputs[k]) for k in ('top20','cap','sims'))
    rep = run(top,cap,sims,full,eval_ids,scratch/'prepared')
    verify(rep=={k:v for k,v in frozen.items() if k!='source_sha256'}, 'Follow-up input replay differs')
    for path,h in binding['code_sha256'].items():
        verify(sha(blob(f'evcap_external/{path}'))==h, f'Code SHA changed: {path}')
    yaml = blob('evcap_external/test2990/test2990_16frames.yaml')
    verify(sha(yaml)==binding['archived_yaml_sha256'], 'Archived YAML SHA changed')
    (scratch/'archived.yaml').write_bytes(yaml)

    assets = {path:receipt(inputs[name]) for name,path in INPUTS.items()}
    texts = {}
    old_outputs = {}
    for k in (1,3,7):
        archived = {}
        for split, cohort in (('test2990',full),('test2690',eval_ids)):
            path = f'evcap_external/outputs/eval_{split}_D{k}/{PRED_BASENAME}'
            raw = blob(path)
            rows = json.loads(raw)
            verify([r['image_name'] for r in rows]==cohort, f'Archive cohort/order changed: {path}')
            verify(all(isinstance(r['captions'],list) and len(r['captions'])==20 and r['prediction'].strip() for r in rows),
                   f'Invalid archived prediction or reference list: {path}')
            archived[split] = rows
            assets[path] = receipt(raw)
        eval_set=set(eval_ids)
        verify([r for r in archived['test2990'] if r['image_name'] in eval_set]==archived['test2690'],
               f'Filtering D{k} fails to reproduce released test2690 rows')
        refs = [r['captions'] for r in archived['test2990']]
        texts[k] = primitive_hash(refs)
        old_outputs[str(k)] = {'full':len(full),'pilot_excluded':len(eval_ids),'references':sum(map(len,refs)),
                              'full_ref_lists_canonical_sha256':texts[k],
                              'exact_filtered_rows':True,'historical_runtime_trace_present_in_archive':False}
    verify(len(set(texts.values()))==1, 'Reference captions differ across archived baselines')
    dump(scratch/'reference_witness.json',refs)

    built = {p.stem:read(p) for p in (scratch/'prepared').glob('*.json')}
    feature_counts = {}
    for arm,(t,v,cap_key,score_key) in ARMS.items():
        current_cap = cap if cap_key=='frozen_D_cap' else built[cap_key]
        current_sims = sims if score_key=='frozen_D_sims' else built[score_key]
        all_rep_ids = set()
        for q in full:
            vision = actual_vision(current_cap[q],current_sims[q],v)
            all_rep_ids.update(vision['representatives'])
            if arm.startswith('B'):
                verify(vision==actual_vision(cap[q],sims[q],3)==built['B_visual_contract'][q], 'B visual map changed')
                verify(current_cap[q][0][:3]==cap[q][0][:3] and all(not x for x in current_cap[q][2][3:]), 'B head/tail map changed')
            elif arm=='A0':
                verify(current_cap[q]==[[],[],[]] and current_sims[q]==[[],[]], 'A0 nonempty slot')
            else:
                verify(len(cap[q][0])>=t, 'Archived D prefix too short')
        (scratch/f'expected_feature_ids_{arm}.txt').write_text('\n'.join(sorted(all_rep_ids))+'\n', encoding='utf-8')
        feature_counts[arm]={'unique_cache_files':len(all_rep_ids),'cache_loads_per_full_arm':v*len(full),
                             'sorted_feature_ids_sha256':primitive_hash(sorted(all_rep_ids))}
    test_script = blob('scripts/run_test.sh').decode()
    verify(bool(re.search(r'--bs 1 --topn \$K --random_seed 42 --max_num_frames 16 --num_video_query_token 32',test_script)),
           'Archived test runner flags do not match the expected invocation')
    ckpt_audit = blob('reports/audit_checkpoint_usage.md').decode()
    ckpt_sha = re.search(r'sha256 `([a-f0-9]{64})`',ckpt_audit)[1]
    date_doc = {'basis_commit':BASE,'source_read':'READABLE_LOCAL_GIT_BLOBS','git_assets':assets,
        'baseline_predictions':old_outputs,'archive_run_script':receipt(blob('scripts/run_test.sh')),
        'archive_evaluator':binding['code_sha256'],'archive_yaml':receipt(yaml),
        'runtime_dependency_sha256':{name:sha(blob(f'evcap_external/{name}')) for name in RUNTIME_DEPENDENCIES},
        'archived_checkpoint':{'basename':CHECKPOINT,'reported_sha256':ckpt_sha,'reported_bytes':147337150,
                               'hash_authenticated_to_each_historical_process':False},
        'cpu_preflight':rep,'required_cache_by_arm':feature_counts,
        'smoke_prefix':{'queries_per_arm':8,'first_eight_eligible_B':sum(q in set(built['B_eligible_queries']) for q in full[:8])},
        'historical_log_retention':'No execution-bound receipt with historical query frame/feature/prompt/weight hashes was found in the frozen Git archive'}
    dump(scratch/'archive_evidence.json',date_doc)
    return date_doc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--scratch',type=Path,required=True)
    args=p.parse_args()
    result=inspect(args.repo,args.scratch)
    print(json.dumps({k:result[k] for k in ('baseline_predictions','required_cache_by_arm','smoke_prefix')},indent=2))


if __name__=='__main__':
    main()
