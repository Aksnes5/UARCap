"""Default to a local dry run. Future execution requires an approved receipt and a GPU lock."""

import argparse
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import importlib
from itertools import islice
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
from time import perf_counter

from audit_archives import CHECKPOINT, DESIGN, dump, read, sha, verify
from runtime_trace import attach, frozen


HERE=Path(__file__).resolve().parent
CP_HASH='a8b5af79d55a2c7e684771475d645be6373208bdaae5a4211bee556581eb8002'
SCOPE='uarcap-v06-deploy-1'
SMOKE_LIMIT=8
EXECUTION_KEYS=('device','gpu_uuid','test_yaml_sha256','frame_receipts','checkpoint_sha256','llm_model',
                'llm_manifest_sha256','python_environment_sha256','video_feature_dir','offline_cache_dirs','canonical_checkpoint')


def options(root, d, arm, mode):
    verify(re.fullmatch(r'[A-Za-z0-9_-]+',d['run_id']) and 'eval' not in d['run_id'],
           'Run identifier would change the frozen evaluator checkpoint path')
    verify(d['checkpoint_name']==CHECKPOINT,'Wrong checkpoint basename')
    attempt=d.get('attempt',1)
    verify(type(attempt) is int and attempt>=1,'Attempt must be a positive integer')
    logical_id=f'{mode}_{d["run_id"]}'
    logical=dict(d,run_id=f'{logical_id}_attempt{attempt}',logical_run_id=logical_id)
    kv=frozen.argv_for(arm,logical)
    pairs=dict(zip(kv[::2],kv[1::2]))
    expected=f'outputs/eval_v06_{logical["run_id"]}_{arm}'
    verify(pairs['--out_path']==expected,'Unexpected output routing')
    out=root/expected
    return logical, kv, out, root/expected.replace('eval','train')/CHECKPOINT


def verify_verified_report(path, family, mode, frozen_ids, run_id, execution_inputs):
    report=read(Path(path))
    verify(all((report.get('scope')==SCOPE, report.get('family')==family, report.get('mode')==mode,
                report.get('full_ids_sha256')==frozen_ids, report.get('trace_parity') is True,
                report.get('run_id')==run_id, report.get('execution_inputs')==execution_inputs)),
           'Required prior trace validation is missing or mismatches this runtime')
    verify(report.get('deployment_code_sha256')==implementation_hashes(), 'Prior gate used different instrumentation')
    return report['runtime']


def implementation_hashes():
    files=(HERE/'deploy_guard.py',HERE/'runtime_trace.py',DESIGN/'launch_pinned.py')
    return {p.name:sha(p.read_bytes().replace(b'\r\n',b'\n')) for p in files}


def file_receipts(root, d, arm, frozen_def, cache_ids):
    verify(d['checkpoint_sha256']==CP_HASH,'Wrong historical checkpoint hash')
    yaml=Path(d['test_yaml'])
    roles={}
    for s in yaml.read_text(encoding='utf-8').splitlines():
        if s.strip() and not s.lstrip().startswith('#'):
            k,v=s.split(':',1)
            roles[k.strip()]=Path(v.strip().strip('"\''))
    receipts={r['role']:r for r in d['frame_receipts']}
    verify(set(roles)=={'img','label','caption'} and set(receipts)==set(roles), 'Unexpected YAML role set')
    for role,path in roles.items():
        row=receipts[role]
        verify(path.is_absolute() and path.resolve()==Path(row['path']).resolve(),f'YAML and receipt differ: {role}')
        verify(row.get('digest_status')=='VERIFIED', f'No trusted asset digest: {role}')
        offset=path.with_suffix('.lineidx')
        verify(frozen.digest(offset)==row['lineidx_sha256'],f'Offset has changed: {role}')
        if role!='img':
            verify(frozen.digest(path)==row['sha256'],f'Annotation bytes have changed: {role}')
        else:
            verify(path.stat().st_mtime_ns==row['mtime_ns'],'Image TSV changed since read-only verification')
    verify(all(re.fullmatch(r'[A-Za-z0-9_.-]+',key) for key in cache_ids),'Bad clip cache identifiers')
    for name in cache_ids:
        p=Path(d['video_feature_dir'])/(name+'.pt')
        verify(p.is_file() and os.access(p,os.R_OK), 'Expected feature unavailable: '+name)
    for key in ('llm_manifest','python_environment'):
        path=Path(d[key])
        verify(path.is_absolute() and frozen.digest(path)==d[key+'_sha256'],f'Approved {key} receipt has changed')
    verify(Path(d['llm_model']).is_dir(), 'Approved Vicuna model directory is missing')
    verify('7b' in d['llm_model'], 'Pinned constructor needs the 7b path to select its Video-LLaMA weights')
    for key in ('TORCH_HOME','HUGGINGFACE_HUB_CACHE'):
        path=Path(d['offline_cache_dirs'][key])
        verify(path.is_absolute() and path.is_dir(), f'Missing existing {key}')
    return frozen.validate_inputs(root,d,arm,frozen_def)


def verify_isolation(root, d, out, checkpoint):
    original=Path(d['original_evcap_root']).resolve(strict=True)
    verify(original!=root and original not in root.parents and root not in original.parents,
           'Use a dedicated checkout outside the historical inference tree')
    verify((root/'outputs').resolve()==root/'outputs' and not out.exists() and not out.is_symlink(),
           'Output path is not isolated or already exists')
    canonical=Path(d['canonical_checkpoint']).resolve(strict=True)
    verify(checkpoint.resolve(strict=True)==canonical, 'Checkpoint slot must resolve to the approved existing file')


@contextmanager
def physical_gpu_lock(d):
    if not sys.platform.startswith('linux'):
        raise RuntimeError('Use the authorized existing Linux runtime for real execution')
    import fcntl
    lock=Path(d['shared_gpu_lock'])
    verify(lock.is_absolute() and lock.is_file(),'Existing project GPU lock is required')
    verify(re.fullmatch(r'GPU-[0-9a-fA-F-]{36}', d['gpu_uuid']) and d['device']=='cuda:0','GPU assignment is invalid')
    with lock.open('rb') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        drv=subprocess.check_output(['nvidia-smi','-i',d['gpu_uuid'],'--query-gpu=driver_version','--format=csv,noheader'], text=True).strip()
        active=subprocess.check_output(['nvidia-smi','-i',d['gpu_uuid'],'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
        verify(not active.strip(),'GPU has an active compute client or a status that could not be confirmed')
        old=os.environ.get('CUDA_VISIBLE_DEVICES')
        verify(old in (None,d['gpu_uuid']),'Existing CUDA visibility does not match the approved GPU UUID')
        os.environ['CUDA_VISIBLE_DEVICES']=d['gpu_uuid']
        yield drv


def execute(root, d, arm, mode, argv, out, driver, design, execution_inputs, prior_runtimes):
    started=perf_counter()
    out.mkdir(parents=True,exist_ok=False)
    os.chdir(root)
    sys.path.insert(0,str(root))
    for name,path in d['offline_cache_dirs'].items():
        verify(name in ('TORCH_HOME','HUGGINGFACE_HUB_CACHE'),'Unexpected cache variable')
        os.environ[name]=path
    os.environ.update(TRANSFORMERS_OFFLINE='1',HF_HUB_OFFLINE='1',USE_TF='0',TOKENIZERS_PARALLELISM='false')
    with (out/'input_trace.jsonl').open('x',encoding='utf-8') as trace, (out/'launcher.log').open('x',encoding='utf-8') as log:
        def emit(row):
            trace.write(json.dumps(row,ensure_ascii=False)+'\n'); trace.flush()
        with redirect_stdout(log), redirect_stderr(log):
            torch=importlib.import_module('torch')
            from importlib import metadata
            runtime={'driver':driver,'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,
                     'gpu_uuid':d['gpu_uuid'],'transformers':metadata.version('transformers')}
            verify(all(r==runtime for r in prior_runtimes), 'Current runtime differs from the verified prerequisite')
            hub=importlib.import_module('timm.models.hub')
            cached=hub.download_cached_file
            def cached_only(url,*args,**kwargs):
                filename=Path(torch.hub.urlparse(url).path).name
                verify((Path(hub.get_cache_dir())/filename).is_file(),f'Existing vision cache required: {filename}')
                return cached(url,*args,**kwargs)
            hub.download_cached_file=cached_only
            model=importlib.import_module(frozen.MODULE)
            common={k:d[k] for k in EXECUTION_KEYS}
            common['run_id']=d['logical_run_id']
            _,_,cap,sims=frozen.ARMS[arm]
            dump(out/'run_receipt.json',{'scope':SCOPE,'arm':arm,'mode':mode,'common':common,'runtime':runtime,
                 'code_sha256':design['code_sha256'],'engine_args':frozen.BASE_ARGS,'execution_inputs':execution_inputs,
                 'deployment_code_sha256':implementation_hashes(),'attempt':d.get('attempt',1),
                 'arm_artifact_sha256':{key:frozen.digest(d['artifacts'][key]) for key in (cap,sims)}})
            contract=read(Path(d['artifacts']['B_visual_contract'])) if arm.startswith('B') else {}
            attach(model.EVCap,torch,arm,contract,emit)
            if mode=='smoke':
                loader=importlib.import_module('dataset.msrvtt.vl_dataloader')
                old=loader.make_data_loader
                def bounded_loader(*args,**kwargs):
                    it, selected=old(*args,**kwargs)
                    return islice(it,SMOKE_LIMIT),selected
                loader.make_data_loader=bounded_loader
            sys.argv=[str(root/frozen.EVALUATOR),*argv]
            runpy.run_path(str(root/frozen.EVALUATOR),run_name='__main__')
            outputs=list(out.glob('*generated_captions.json'))
            verify(len(outputs)==1,'Evaluator did not produce exactly one prediction object')
            payload=read(outputs[0]); want=SMOKE_LIMIT if mode=='smoke' else 2990
            verify(len(payload)==want,'Incorrect generation count')
            dump(out/'runtime_telemetry.json',{'wall_seconds':perf_counter()-started,'trace_bytes':(out/'input_trace.jsonl').stat().st_size,
                 'prediction_bytes':outputs[0].stat().st_size,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                 'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
            dump(out/'arm_completed.json',{'scope':SCOPE,'mode':mode,'arm':arm,'rows':want,
                 'predictions':outputs[0].name,'prediction_sha256':frozen.digest(outputs[0]),
                 'trace_sha256':frozen.digest(out/'input_trace.jsonl'),'receipt_sha256':frozen.digest(out/'run_receipt.json')})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('evcap-root','deployment','archive-scratch'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--arm',choices=frozen.ARMS,required=True)
    p.add_argument('--mode',choices=('dry-run','smoke','full'),default='dry-run')
    p.add_argument('--gpu-approval-id')
    a=p.parse_args()
    root=a.evcap_root.resolve(); d=read(a.deployment)
    # Refuse before any reads of model weights, CUDA imports, GPU queries, locks or filesystem writes.
    if a.mode!='dry-run':
        verify(a.gpu_approval_id and d.get('approval_id')==a.gpu_approval_id and d.get('server_data_ready') is True,
               'Explicit approved execution and verified server inventory are required')
    mode='full' if a.mode=='dry-run' else a.mode
    run,argv,out,ckpt=options(root,d,a.arm,mode)
    if a.mode=='dry-run':
        print(json.dumps({'mode':'dry-run','loads_model':False,'arm':a.arm,'argv':argv,'out':str(out),'checkpoint_slot':str(ckpt)},indent=2))
        return
    verify_isolation(root,run,out,ckpt)
    design=read(DESIGN/'FROZEN_SOURCE_BINDINGS.json'); design['preflight']=read(DESIGN/'CPU_PREFLIGHT_SUMMARY.json')
    archive=read(HERE/'ARCHIVE_EVIDENCE.json')
    for name, expected in archive['runtime_dependency_sha256'].items():
        verify(frozen.digest(root/name)==expected, 'Archived runtime dependency hash mismatch: '+name)
    ids_hash=design['preflight']['source_sha256']['full_ids']
    execution_inputs=frozen.primitive_hash({k:d[k] for k in EXECUTION_KEYS})
    prior=[]
    if mode=='full':
        prior.append(verify_verified_report(d['smoke_report'][a.arm[0]],a.arm[0],'smoke',ids_hash,'smoke_'+d['run_id'],execution_inputs))
    if a.arm.startswith('B'):
        prior.append(verify_verified_report(d['A_full_report'],'A','full',ids_hash,'full_'+d['run_id'],execution_inputs))
    feature_ids=(a.archive_scratch/f'expected_feature_ids_{a.arm}.txt').read_text(encoding='utf-8').splitlines()
    verify(frozen.primitive_hash(feature_ids)==archive['required_cache_by_arm'][a.arm]['sorted_feature_ids_sha256'], 'Feature inventory changed')
    verify(file_receipts(root,run,a.arm,design,feature_ids)==out,'Invalid output assignment')
    with physical_gpu_lock(d) as driver:
        execute(root,run,a.arm,mode,argv,out,driver,design,execution_inputs,prior)


if __name__=='__main__':
    main()
