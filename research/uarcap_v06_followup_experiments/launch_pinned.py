"""Emit or launch an authorized follow-up arm with instrumentation in a separate checkout."""

import argparse
from contextlib import contextmanager
from hashlib import sha256
import importlib
import json
import math
import os
from pathlib import Path
import re
import runpy
import sys

from prepare_inputs import primitive_hash


EVALUATOR = 'eval_evcap_fix_video_llama_xclip_msrvtt_fusion_gate_prompt5.py'
MODULE = 'models.evcap_fix_video_llama_xclip_msrvtt_smallcap_quality_fusion_gate_prompt5'
PROMPT = 'prompts/prompt_evcap_msrvtt_detail_smallcap_prompt5.txt'
MODEL_FILE = MODULE.replace('.', '/') + '.py'
BASE_ARGS = {'bs':1,'random_seed':42,'max_num_frames':16,'num_video_query_token':32,'beam_width':5,
             'freeze_video_qformer':True,'vit_precision':'fp16','freeze_llm':True,'freeze_vit':True,
             'freeze_qformer':True,'load_video_feature':True,'share_query_tokens':True,'retriever':True,
             'deterministic':True,'use_fusion':True,'no_retrieval':False}
ARMS = {'A0':(0,0,'A_k0_cap','A_k0_sims'), 'A1':(1,1,'frozen_D_cap','frozen_D_sims'),
        'A7':(7,7,'frozen_D_cap','frozen_D_sims'), 'B3':(3,3,'B_text3_cap','B_shared_visual_sims'),
        'B7HQ':(7,3,'B_text7_hq_cap','B_shared_visual_sims'), 'B7LQ':(7,3,'B_text7_lq_cap','B_shared_visual_sims')}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_bytes())


def digest(path):
    with Path(path).open('rb') as f:
        h=sha256()
        while data:=f.read(1024*1024):
            h.update(data)
    return h.hexdigest()


def argv_for(arm, d):
    require(re.fullmatch(r'[A-Za-z0-9_-]+', d['run_id']), 'Unsafe run identifier')
    require(re.fullmatch(r'[A-Za-z0-9_.-]+\.pt', d['checkpoint_name']), 'Invalid checkpoint basename')
    t,v,cap,sims=ARMS[arm]
    raw={'device':d['device'],'test_yaml':d['test_yaml'],'out_path':f'outputs/eval_v06_{d["run_id"]}_{arm}',
         'ckpt_name':d['checkpoint_name'],'prompt_path':PROMPT,'llm_model':d['llm_model'],
         'ext_path':d['artifacts'][cap], 'sim_scores_path':d['artifacts'][sims], 'video_feature_dir':d['video_feature_dir'],
         'topn':max(t,v),'text_topn':t,'visual_topn':v,**BASE_ARGS}
    if arm.startswith('B'):
        raw['topn']=7
    return [v for k,val in raw.items() for v in ('--'+k,str(val))]


def actual_vision(cap, sims, k):
    txts, _, groups=cap
    require(len(set(sims[0]))==len(sims[0]), 'Non-unique similarity keys')
    weight_by_caption=dict(zip(*sims,strict=True))
    require(len(txts)>=k and len(groups)>=k, 'Wrong visual prefix length')
    require(all(groups[i] and txts[i] in weight_by_caption and math.isfinite(weight_by_caption[txts[i]]) for i in range(k)),
            'Missing clip representative or score')
    return {'representatives':[groups[i][0] for i in range(k)],
            'similarities':[weight_by_caption[txts[i]] for i in range(k)],
            'lookup_text_hashes':[sha256(txts[i].encode()).hexdigest() for i in range(k)]}


def attach_tracing(cls, torch, arm, contract, emit):
    """Hook exactly the frozen evaluator's encode/fuse boundary. Intended for batch size one."""
    encode, fuse, wrap=cls.encode_img,cls.fusion_train,cls.prompt_wrap
    k=ARMS[arm][1]
    active={}
    pending={}

    def fingerprint(t):
        raw=t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
        return {'shape':list(t.shape),'dtype':str(t.dtype),'sha256':sha256(raw).hexdigest()}

    @contextmanager
    def trace_loads(records):
        old=torch.load
        def load_feature(path,*args,**kwargs):
            x=old(path,*args,**kwargs)
            records.append((Path(path).name.removesuffix('.pt'),fingerprint(x)))
            return x
        torch.load=load_feature
        try:
            yield
        finally:
            torch.load=old

    def audited_fuse(self, hidden, attention, ids, train_data):
        require(len(ids)==1 and ids[0]==active.get('query'), 'Missing encoder trace context')
        q=ids[0]
        vision=actual_vision(self.cap[q],self.sims_scores[q],k)
        if arm.startswith('B'):
            require(vision==contract[q], f'Visual contract changed for {q}')
        if arm=='A0':
            require(self.cap[q]==[[],[],[]] and self.sims_scores[q]==[[],[]], 'A0 evidence must be empty')
        records=[]
        with trace_loads(records):
            fused=fuse(self,hidden,attention,ids,train_data)
        require([r[0] for r in records]==vision['representatives'], f'Missing/changed cached feature for {q}')
        require(all(r[1]['shape']==[32,768] for r in records), f'Unexpected cached feature shape for {q}')
        if arm=='A0':
            require(bool(torch.count_nonzero(fused).item()==0), 'A0 aggregate must be zero')
        active.update(vision=vision, feature_tensors=[r[1] for r in records], aggregate=fingerprint(fused))
        return fused

    def audited_encode(self,image,ids,train_data,*args,**kwargs):
        require(not active and not pending and len(ids)==1, 'Nested or multi-query encoder call')
        q=ids[0]
        active.update(query=q, frames=fingerprint(image), training=bool(self.training))
        require(self.use_fusion and self.deterministic and self.load_video_feature and self.visual_topn==k and
                self.text_topn==ARMS[arm][0] and not self.no_retrieval, 'Frozen architecture flags were changed')
        snapshots=[]
        def on_attention(_module, inputs):
            snapshots.append([fingerprint(v) for v in inputs[:3]])
        hook=self.attn.register_forward_pre_hook(on_attention)
        try:
            outputs=encode(self,image,ids,train_data,*args,**kwargs)
            require(len(snapshots)==1 and 'aggregate' in active, 'Missing or repeated attention input')
            require(snapshots[0][1]==snapshots[0][2]==active['aggregate'], 'K/V differs from the measured aggregate')
            pending.update(active, qformer_query=snapshots[0][0], gated_projection=fingerprint(outputs[0]),
                           query_projection=fingerprint(outputs[2]))
            return outputs
        finally:
            active.clear()
            hook.remove()
    def audited_wrap(self, img, atts, prompts, ids, query_proj):
        require(len(ids)==1 and ids[0]==pending.get('query') and len(prompts)==1,
                'Unexpected prompt wrapper conditions')
        result=wrap(self,img,atts,prompts,ids,query_proj)
        used=self.cap[ids[0]][0][:self.text_topn]
        emit(dict(pending, prompt_template_sha256=sha256(prompts[0].encode()).hexdigest(),
                  text_count=len(used), consumed_text_sha256=primitive_hash(used), prompt_tokens=int(result[1].sum().item())))
        pending.clear()
        return result

    cls.encode_img,cls.fusion_train,cls.prompt_wrap=audited_encode,audited_fuse,audited_wrap


def validate_inputs(root, d, arm, frozen):
    for filename,expected in frozen['code_sha256'].items():
        require(digest(root/filename)==expected, f'Pinned source mismatch: {filename}')
    require(digest(d['test_yaml'])==d['test_yaml_sha256'], 'Dataset YAML differs from approved receipt')
    require({s['role'] for s in d['frame_receipts']}=={'img','label','caption'}, 'Frame, label and caption receipts required')
    for spec in d['frame_receipts']:
        path=Path(spec['path'])
        require(path.is_absolute() and path.is_file() and path.stat().st_size==spec['size'] and
                re.fullmatch(r'[a-f0-9]{64}',spec['sha256']), 'Frame/annotation receipt unavailable')
    require(all(Path(d[k]).is_absolute() for k in ('test_yaml','llm_model','video_feature_dir')),
            'Supply absolute deployment input paths')
    require(re.fullmatch(r'[a-f0-9]{64}',d['llm_manifest_sha256']) and
            re.fullmatch(r'[a-f0-9]{64}',d['python_environment_sha256']), 'Missing runtime model/environment receipts')
    t,v,cap,sims=ARMS[arm]
    for name in (cap,sims):
        p=d['artifacts'][name]
        require(Path(p).is_absolute(), 'Supply absolute artifact paths')
        if name.startswith('frozen_D'):
            key='cap' if name.endswith('cap') else 'sims'
            require(digest(p)==frozen['preflight']['source_sha256'][key], f'Changed archived {key}')
        else:
            require(primitive_hash(load(p))==frozen['preflight']['inputs_fingerprints'][name], f'Prepared {name} changed')
    if arm.startswith('B'):
        require(Path(d['artifacts']['B_visual_contract']).is_absolute(), 'Supply absolute visual contract path')
        require(primitive_hash(load(d['artifacts']['B_visual_contract']))==frozen['preflight']['inputs_fingerprints']['B_visual_contract'],
                'Changed visual contract')
    out=root/f'outputs/eval_v06_{d["run_id"]}_{arm}'
    ckpt=root/f'outputs/train_v06_{d["run_id"]}_{arm}'/d['checkpoint_name']
    require(not out.exists(), 'Use a new run ID to preserve previous outputs')
    require(digest(ckpt)==d['checkpoint_sha256'], 'Checkpoint differs from approved receipt')
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evcap-root',type=Path,required=True)
    p.add_argument('--deployment',type=Path,required=True)
    p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--gpu-approval-id',help='Explicit operator reference to user approval. Omit for plan output only.')
    args=p.parse_args()
    root=args.evcap_root.resolve()
    deployment=load(args.deployment)
    invocation=argv_for(args.arm,deployment)
    if not args.gpu_approval_id:
        print(json.dumps({'approval':'NO','arm':args.arm,'argv':invocation},indent=2))
        return
    require(deployment.get('approval_id')==args.gpu_approval_id and deployment.get('server_data_ready') is True,
            'Operator receipt and server data check required')
    here=Path(__file__).resolve().parent
    frozen=load(here/'FROZEN_SOURCE_BINDINGS.json')
    frozen['preflight']=load(here/'CPU_PREFLIGHT_SUMMARY.json')
    out=validate_inputs(root,deployment,args.arm,frozen)
    out.mkdir(parents=True,exist_ok=False)
    os.chdir(root)
    sys.path.insert(0,str(root))
    torch=importlib.import_module('torch')
    module=importlib.import_module(MODULE)
    contract=load(deployment['artifacts']['B_visual_contract']) if args.arm.startswith('B') else {}
    common={k:deployment[k] for k in ('run_id','device','test_yaml_sha256','frame_receipts','checkpoint_sha256',
                                      'llm_manifest_sha256','python_environment_sha256','video_feature_dir')}
    runtime={'torch_version':torch.__version__,'cuda_version':torch.version.cuda}
    (out/'run_receipt.json').write_text(json.dumps({'arm':args.arm,'common':common,'runtime':runtime,
          'code_sha256':frozen['code_sha256'],'engine_args':BASE_ARGS},indent=2)+'\n',encoding='utf-8')
    with (out/'input_trace.jsonl').open('x',encoding='utf-8') as log:
        def emit(row):
            log.write(json.dumps(row,ensure_ascii=False)+'\n'); log.flush()
        attach_tracing(module.EVCap,torch,args.arm,contract,emit)
        sys.argv=[str(root/EVALUATOR),*invocation]
        runpy.run_path(str(root/EVALUATOR),run_name='__main__')


if __name__=='__main__':
    main()
