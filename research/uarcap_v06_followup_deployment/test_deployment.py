"""Use Git-pinned method bodies and CPU tensors without loading models or generating captions."""

import argparse
import ast
from copy import deepcopy
from contextlib import nullcontext
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from audit_archives import BASE, CHECKPOINT, read
from deploy_guard import options
from runtime_trace import attach, frozen, vision_py39
from validate_bundle import check_rows, ZERO_AGGREGATE


REPO = Path(__file__).resolve().parents[2]
SCRATCH = Path(os.environ['UARCAP_ARCHIVE_SCRATCH'])


def raw(name):
    return subprocess.check_output(['git','-C',str(REPO),'cat-file','blob',f'{BASE}:evcap_external/{name}']).decode()


def bind(nodes, scope):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<frozen-ast>', 'exec'), scope)


def rearrange(x, pattern, b):
    if pattern == '(b f) hw c-> b f hw c':
        return x.reshape(b, -1, x.shape[-2], x.shape[-1])
    return x.flatten(1,2)


class Bert(torch.nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, query_embeds, encoder_hidden_states, **kwargs):
        values = encoder_hidden_states.mean(dim=(1, 2), keepdim=True)
        return SimpleNamespace(last_hidden_state=self.dropout(query_embeds + values))


class Tokens:
    pad_token_id = 0

    def __init__(self):
        self.raw = []

    def __call__(self, text, **kwargs):
        self.raw.append(text)
        tokens = torch.tensor([[ord(c) % 20 + 1 for c in text]], dtype=torch.long)
        return SimpleNamespace(to=lambda _device: SimpleNamespace(input_ids=tokens))


class CpuPreflight(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.full = (SCRATCH/'msrvtt_test2990.tsv').read_text().splitlines()
        cls.eligible = set(read(SCRATCH/'prepared'/'B_eligible_queries.json'))
        cls.design = read(SCRATCH/'archive_evidence.json')
        cls.raw_prompt = '###Human: {} ###Assistant: '.format(raw(frozen.PROMPT))
        cls.dcap = read(SCRATCH/'external_test_d7_cap.json')
        cls.dsim = read(SCRATCH/'external_test_d7_sims.json')
        cls.built = {p.stem:read(p) for p in (SCRATCH/'prepared').glob('*.json')}

    def config(self):
        return dict(run_id='cpu_review', device='cuda:0', test_yaml='test.yaml', checkpoint_name=CHECKPOINT,
                    llm_model='offline-llm', video_feature_dir='existing-cache',
                    artifacts={key:f'{key}.json' for vals in frozen.ARMS.values() for key in vals[2:]})

    def frozen_model(self):
        tree = ast.parse(raw(frozen.MODEL_FILE))
        evcap = next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='EVCap')
        methods = ['encode_img','fusion_train','prompt_wrap']
        scope = dict(torch=torch,os=os,random=random,rearrange=rearrange)
        bind([x for x in evcap.body if isinstance(x,ast.FunctionDef) and x.name in methods],scope)
        return type('CpuFrozenMethods',(),{k:scope[k] for k in methods})

    def test_actual_evaluator_cli_and_model_zero_fallback(self):
        tree = ast.parse(raw(frozen.EVALUATOR))
        start = tree.body[-1]
        utils = ast.parse(raw('dataset/msrvtt/data_utils/miscellaneous.py'))
        definitions = [n for n in utils.body if isinstance(n,ast.FunctionDef) and n.name=='str_to_bool']
        parser_scope = {'argparse':argparse}
        bind(definitions,parser_scope)
        bind([n for n in start.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='parser'
              or isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='parser.add_argument'],
             parser_scope)
        call = next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func)=='EVCap')
        parsed = next(node for node in ast.parse(raw(frozen.MODEL_FILE)).body
                      if isinstance(node,ast.ClassDef) and node.name=='EVCap')
        init = next(n for n in parsed.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
        assigns = {ast.unparse(n.targets[0]):n.value for n in init.body if isinstance(n,ast.Assign)}
        for arm, (t,v,_,_) in frozen.ARMS.items():
            _, argv, _, _ = options(REPO,self.config(),arm,'full')
            args = parser_scope['parser'].parse_args(argv)
            scope = {'args':args,'topn':args.topn}
            for name,k in (('text_topn',t),('visual_topn',v)):
                kw = next(x.value for x in call.keywords if x.arg==name)
                scope[name] = eval(compile(ast.Expression(kw),'<frozen>', 'eval'),scope)
                actual = eval(compile(ast.Expression(assigns['self.'+name]),'<frozen>', 'eval'),scope)
                self.assertEqual(actual,k)
            self.assertTrue(args.use_fusion and args.retriever and not args.no_retrieval)
            self.assertEqual((args.beam_width,args.random_seed,args.max_num_frames), (5,42,16))

    def test_attempt_keeps_logical_cohort_but_uses_new_checkpoint_slot(self):
        first,_,a,ca = options(REPO,self.config(),'A0','full')
        second,_,b,cb = options(REPO,dict(self.config(),attempt=2),'A0','full')
        self.assertEqual(first['logical_run_id'],second['logical_run_id'])
        self.assertNotEqual(a,b)
        self.assertNotEqual(ca,cb)
        self.assertNotEqual(a,options(REPO,self.config(),'A0','smoke')[2])

    def test_guard_refuses_unauthorized_gpu_before_asset_access(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            p=base/'fixture.json'
            p.write_text(json.dumps({'server_data_ready':False}),encoding='utf-8')
            res=subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('deploy_guard.py')),
                '--evcap-root',str(base), '--deployment',str(p),'--archive-scratch',str(base/'absent'),
                '--arm','A0','--mode','smoke'],capture_output=True,text=True)
            self.assertNotEqual(res.returncode,0)
            self.assertIn('Explicit approved execution and verified server inventory are required',res.stderr)
            self.assertEqual(list(base.iterdir()),[p])

    def run_family(self, arms, order):
        traces, caps, sims = {}, {}, {}
        for arm in arms:
            torch.manual_seed(2026)
            t,v,cap_key,sim_key = frozen.ARMS[arm]
            caps[arm] = self.dcap if cap_key=='frozen_D_cap' else self.built[cap_key]
            sims[arm] = self.dsim if sim_key=='frozen_D_sims' else self.built[sim_key]
            cls = self.frozen_model()
            attach(cls,torch,arm,self.built['B_visual_contract'] if arm.startswith('B') else {}, traces.setdefault(arm,[]).append)
            model = cls()
            model.__dict__.update(dict(training=True, low_resource=False, frames=2, retriever=True, num_video_query_token=32,
               text_topn=t, visual_topn=v, deterministic=True, load_video_feature=True, video_feature_dir='feature',
               use_fusion=True, no_retrieval=False, cap=caps[arm], sims_scores=sims[arm], maybe_autocast=nullcontext,
               visual_encoder=lambda x:x, ln_vision=torch.nn.Identity(), query_tokens=torch.full((1,32,768),.25),
               video_query_tokens=torch.full((1,32,768),.5), Qformer=SimpleNamespace(bert=Bert()),
               video_Qformer=SimpleNamespace(bert=Bert(0.1)), video_frame_position_embedding=torch.nn.Embedding(2,768),
               attn=torch.nn.MultiheadAttention(768,8,batch_first=True), llama_proj=torch.nn.Linear(768,8),
               fusion_llama_proj=torch.nn.Linear(768,8), llama_tokenizer=Tokens()))
            for name,activation in (('update_gate',torch.nn.Sigmoid),('reset_gate',torch.nn.Sigmoid),('new_transform',torch.nn.Tanh)):
                setattr(model,name,torch.nn.Sequential(torch.nn.Linear(1536,768),activation()))
            model.llama_model=SimpleNamespace(training=False,model=SimpleNamespace(embed_tokens=torch.nn.Embedding(22,8)))
            for q in order:
                expected = vision_py39(caps[arm][q],sims[arm][q],v)['representatives']
                def fake_load(path, **kwargs):
                    i=expected.index(Path(path).stem)
                    return torch.full((32,768),float(i+1))
                with patch.object(torch,'load',side_effect=fake_load) as loader, torch.no_grad():
                    gated, mask, query = model.encode_img(torch.ones(2,3,8),[q],None)
                    out, attn_mask = model.prompt_wrap(gated,mask,[self.raw_prompt],[q],query)
                    self.assertEqual(loader.call_count,v)
                    self.assertEqual(self.raw_prompt.count('<VideoHere>'),2)
                    self.assertEqual(attn_mask.dtype,torch.int32)
                    self.assertTrue(bool(torch.isfinite(out).all()))
                if arm=='A0':
                    before, rest=self.raw_prompt.split('<VideoHere>',1)
                    middle, after=rest.split('<VideoHere>',1)
                    self.assertEqual(model.llama_tokenizer.raw[-3:],[before,middle,after.format('\n')])
                    self.assertEqual(traces[arm][-1]['aggregate'],ZERO_AGGREGATE)
                    self.assertGreater(int(torch.count_nonzero(gated)),0)
        return traces,caps,sims

    def test_a_real_methods_cpu_zero_k_and_prompt(self):
        order = self.full[:1]
        traces,caps,sims = self.run_family(('A0','A1','A7'),order)
        check_rows(traces,order,'A',caps,sims,self.eligible)
        damaged = deepcopy(traces)
        damaged['A0'][0]['aggregate']['sha256'] = '0'*64
        with self.assertRaises(ValueError):
            check_rows(damaged,order,'A',caps,sims,self.eligible)

    def test_b_real_methods_cpu_fixed_visual_and_negative_controls(self):
        order = [next(q for q in self.full if q in self.eligible),next(q for q in self.full if q not in self.eligible)]
        traces,caps,sims = self.run_family(('B3','B7HQ','B7LQ'),order)
        check_rows(traces,order,'B',caps,sims,self.eligible)
        self.assertEqual([x['text_count'] for x in traces['B7HQ']],[7,3])
        for key in ('vision','softmax_weights','video_qformer_inputs','reset_gate_input','feature_tensors',
                    'encoder_rng','consumed_text_sha256'):
            damaged = deepcopy(traces)
            damaged['B7HQ'][0][key] = None
            with self.subTest(key=key),self.assertRaises(ValueError):
                check_rows(damaged,order,'B',caps,sims,self.eligible)

    def test_candidate_feature_contract_exact_across_all_arms(self):
        self.assertEqual(self.design['required_cache_by_arm']['A7']['unique_cache_files'],15213)
        self.assertEqual(self.design['required_cache_by_arm']['B3'],self.design['required_cache_by_arm']['B7HQ'])
        self.assertEqual(self.design['required_cache_by_arm']['B3'],self.design['required_cache_by_arm']['B7LQ'])
