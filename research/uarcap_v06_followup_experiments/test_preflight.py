"""CPU-only checks. Tiny NumPy test doubles exercise the tracing contract without torch."""

from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from evaluate_followup import check_trace
from launch_pinned import actual_vision, argv_for, attach_tracing
from prepare_inputs import Item, anchor, choose, pack, visual_contract


class Tensor:
    def __init__(self, values):
        self.x=np.asarray(values,dtype=np.float32)
        self.shape=self.x.shape
        self.dtype=self.x.dtype

    def detach(self): return self
    def contiguous(self): return self
    def cpu(self): return self
    def view(self, kind): return TensorBytes(self.x.view(kind))
    def sum(self): return self.x.sum()


class TensorBytes:
    def __init__(self, x): self.x=x
    def numpy(self): return self.x


def items():
    return [Item(f'a unique event {i}',(f'clip{i}.0',),.5-i*.002) for i in range(20)]


def pins(*qids):
    return [{'query':q,'frames':'same','training':False,'vision':{'representatives':['clip0.0']},
             'feature_tensors':['same'],'aggregate':'same','qformer_query':'same','gated_projection':'same',
             'query_projection':'same','prompt_template_sha256':'same'} for q in qids]


class Preflight(unittest.TestCase):
    def test_judges_candidates_without_references(self):
        candidates=items()
        result,category=choose(candidates,(0,1,2))
        self.assertEqual(category,'ELIGIBLE')
        h,l=result
        self.assertGreaterEqual(h.relevance-l.relevance,.005)
        self.assertEqual(len(set(h.indices) & set(l.indices)),0)
        self.assertEqual(choose(candidates,(0,1,2))[0],result)

    def test_no_relevance_gap_is_ineligible(self):
        candidates=[Item(x.text,x.clips,.5) for x in items()]
        self.assertEqual(choose(candidates,(0,1,2))[1],'NO_BALANCED_SEPARATED_TAIL_PAIR')

    def test_head_requires_actual_similarity(self):
        candidates=items(); cap=pack(candidates[:3]); sims=cap[:2]
        self.assertEqual(anchor(candidates,cap,sims),(0,1,2))
        with self.assertRaisesRegex(ValueError,'differs'):
            anchor(candidates,cap,[sims[0],[.5,.5,.5]])

    def test_tails_have_no_visual_mapping(self):
        candidates=items(); a=pack(candidates[:3],candidates[3:7]); b=pack(candidates[:3],candidates[7:11]); sims=pack(candidates[:3])[:2]
        self.assertTrue(all(group==[] for group in a[2][3:]))
        self.assertTrue(set(a[0][3:]).isdisjoint(b[0][3:]))
        self.assertEqual(actual_vision(a,sims,3),actual_vision(b,sims,3))
        self.assertEqual(actual_vision([[],[],[]],[[],[]],0)['representatives'],[])

    def test_native_zero_arguments(self):
        d={'run_id':'test','checkpoint_name':'a.pt','device':'cuda:0','test_yaml':'test.yml',
           'llm_model':'model','video_feature_dir':'features','artifacts':{'A_k0_cap':'empty','A_k0_sims':'empty_sims'}}
        args=iter(argv_for('A0',d)); parsed=dict(zip(args,args,strict=True))
        self.assertEqual([parsed[k] for k in ('--topn','--text_topn','--visual_topn')],['0','0','0'])
        self.assertEqual(parsed['--use_fusion'],'True')
        self.assertEqual(parsed['--no_retrieval'],'False')

    def test_verified_matrix_rejects_fusion_drift(self):
        groups={name:pins('q1','q2') for name in ('B3','B7HQ','B7LQ')}
        check_trace(groups,['q1','q2'],'B')
        groups['B7HQ'][1]['aggregate']='perturbed'
        with self.assertRaisesRegex(ValueError,'trace parity'):
            check_trace(groups,['q1','q2'],'B')

    def test_verified_matrix_rejects_missing_query(self):
        with self.assertRaisesRegex(ValueError,'out-of-order'):
            check_trace({name:pins('q2','q1') for name in ('B3','B7HQ','B7LQ')},['q1','q2'],'B')

    def test_actual_loader_and_attention_hook(self):
        examples=items(); cap=pack(examples[:3],examples[3:7]); sim=pack(examples[:3])[:2]
        token=np.ones((32,768),dtype=np.float32)
        torch=SimpleNamespace(uint8=np.uint8,load=lambda f:Tensor(token*int(Path(f).stem[4:].split('.')[0])))

        class Attention:
            def __init__(self): self.fn=None
            def register_forward_pre_hook(self,fn):
                self.fn=fn
                return SimpleNamespace(remove=lambda:setattr(self,'fn',None))

        class Fake:
            use_fusion=deterministic=load_video_feature=True
            no_retrieval=training=False
            text_topn,visual_topn=7,3
            def __init__(self):
                self.cap={'q':cap}; self.sims_scores={'q':sim}; self.attn=Attention()
            def fusion_train(self,hidden,att,qs,data):
                x=[torch.load(f'{group[0]}.pt').x for group in self.cap['q'][2][:self.visual_topn]]
                return Tensor(np.mean(x,axis=0)[None,...])
            def encode_img(self,image,qs,data):
                h=Tensor(token[None,...]); f=self.fusion_train(h,None,qs,data); self.attn.fn(self.attn,(h,f,f))
                return f,None,h
            def prompt_wrap(self,img,atts,ps,ids,qp):
                return img,Tensor(np.ones((1,72)))

        out=[]
        attach_tracing(Fake,torch,'B7HQ',{'q':visual_contract(examples[:3])},out.append)
        model=Fake(); body=Tensor(np.ones((16,3,2,2))); a,mask,b=model.encode_img(body,['q'],None)
        model.prompt_wrap(a,mask,['frozen'],['q'],b)
        self.assertEqual(out[0]['vision']['representatives'],['clip0.0','clip1.0','clip2.0'])
        self.assertEqual(out[0]['aggregate']['sha256'],out[0]['gated_projection']['sha256'])
        self.assertEqual(out[0]['prompt_tokens'],72)
        self.assertEqual(out[0]['text_count'],7)


if __name__=='__main__':
    unittest.main()
