"""Extend the frozen hook with observed Q-Former inputs, gates, weights and masks."""

from hashlib import sha256
import math
from pathlib import Path
import sys


DESIGN = Path(__file__).resolve().parents[1] / 'uarcap_v06_followup_experiments'
sys.path.insert(0, str(DESIGN))
import launch_pinned as frozen

EXTENDED_PINS = ('image_qformer_inputs','video_qformer_inputs','attn_key','attn_value',
                 'update_gate_input','reset_gate_input','new_transform_input','softmax_weights','visual_mask',
                 'module_training','encoder_rng')
A_EXTENDED_PINS = ('image_qformer_inputs','video_qformer_inputs','visual_mask','module_training','encoder_rng')


def vision_py39(cap, sims, k):
    """Same frozen lookup with an explicit length check for legacy Python 3.9."""
    captions, _, clips = cap
    keys, values = sims
    frozen.require(len(keys)==len(values)==len(set(keys)), 'Non-unique or unpaired similarity keys')
    index = dict(zip(keys,values))
    frozen.require(len(captions)>=k and len(clips)>=k, 'Wrong visual prefix length')
    frozen.require(all(clips[i] and captions[i] in index and math.isfinite(index[captions[i]]) for i in range(k)),
                   'Missing clip representative or score')
    return {'representatives':[clips[i][0] for i in range(k)], 'similarities':[index[captions[i]] for i in range(k)],
            'lookup_text_hashes':[sha256(c.encode()).hexdigest() for c in captions[:k]]}


def digest_tensor(t, torch):
    cpu=t.detach().contiguous().cpu()
    if torch.is_floating_point(cpu):
        frozen.require(bool(torch.isfinite(cpu).all().item()), 'Non-finite traced tensor')
    return {'shape':list(cpu.shape),'dtype':str(cpu.dtype),'sha256':sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest()}


def attach(cls, torch, arm, contract, emit):
    """Install on the pinned class after runtime approval. Every sample is traced."""
    digest=lambda t:digest_tensor(t,torch)
    observed={}

    def receipt(row):
        missing=set(EXTENDED_PINS)-set(observed)
        frozen.require(not missing, 'Missing runtime observations: '+', '.join(sorted(missing)))
        frozen.require(observed['attn_key']==observed['attn_value']==row['aggregate'], 'Attention K/V differs from aggregate')
        k=frozen.ARMS[arm][1]
        expected_shape=[1,32,768]
        frozen.require(row['aggregate']['shape']==expected_shape and row['qformer_query']['shape']==expected_shape,
                       'Unexpected query/aggregate shape')
        frozen.require((observed['softmax_weights'] is None)==(k==0), 'Wrong number of visual softmax operations')
        if k:
            frozen.require(len(observed['softmax_weights'])==k and
                           abs(sum(observed['softmax_weights'])-1)<1e-5, 'Unexpected actual visual softmax')
        emit(dict(row, schema_version='uarcap-v06-deploy-trace-1', **observed))
        observed.clear()

    original_wrap=cls.prompt_wrap

    def audited_prompt(self,img,atts,prompts,ids,qproj):
        result=original_wrap(self,img,atts,prompts,ids,qproj)
        emb,mask=result
        frozen.require(tuple(mask.shape)==tuple(emb.shape[:2]) and mask.dtype==torch.int32 and
                       bool((mask==1).all().item()), 'Invalid prompt attention mask for batch size one')
        # Read all bytes once. The content hash is kept out of cross-arm pins because the text changes.
        put('wrapped_prompt',digest(emb)); put('prompt_mask',digest(mask))
        return result

    cls.prompt_wrap=audited_prompt
    frozen.actual_vision=vision_py39
    frozen.attach_tracing(cls,torch,arm,contract,receipt)
    encode, fused = cls.encode_img, cls.fusion_train

    def put(k,v):
        frozen.require(k not in observed, f'Duplicate runtime observation: {k}')
        observed[k]=v

    def audited_fusion(self, q, m, ids, train_data):
        original=torch.softmax
        seen=[]
        def capture(logits,*args,**kwargs):
            w=original(logits,*args,**kwargs)
            seen.append(w.detach().cpu().tolist())
            return w
        torch.softmax=capture
        try:
            result=fused(self,q,m,ids,train_data)
            frozen.require(len(seen)==(1 if frozen.ARMS[arm][1] else 0), 'Unexpected softmax capture count')
            put('softmax_weights',seen[0] if seen else None)
            return result
        finally:
            torch.softmax=original

    def audited_encoder(self, image, ids, train_data, *args, **kwargs):
        # The inherited evaluator leaves the parent mode intact. Preserve it and validate RNG alignment
        # because freeze_video_qformer only disables gradients and may leave dropout active.
        put('module_training',{'image_qformer':bool(self.Qformer.bert.training),
                               'video_qformer':bool(self.video_Qformer.bert.training),
                               'attention':bool(self.attn.training),'llm':bool(self.llama_model.training)})
        put('encoder_rng',{'cpu':digest(torch.get_rng_state()),
                           'cuda':digest(torch.cuda.get_rng_state(image.device)) if image.is_cuda else None})
        def qformer(label):
            def hook(_module, _args, kw):
                keys=('query_embeds','encoder_hidden_states','encoder_attention_mask')
                put(label,{k:digest(kw[k]) for k in keys})
            return hook

        def attn(_module, supplied):
            key=digest(supplied[1]); put('attn_key',key)
            put('attn_value',key if supplied[1] is supplied[2] else digest(supplied[2]))

        def gate(label):
            return lambda _module, supplied:put(label,digest(supplied[0]))

        handles=[]
        try:
            handles.append(self.Qformer.bert.register_forward_pre_hook(qformer('image_qformer_inputs'),with_kwargs=True))
            handles.append(self.video_Qformer.bert.register_forward_pre_hook(qformer('video_qformer_inputs'),with_kwargs=True))
            handles.append(self.attn.register_forward_pre_hook(attn))
            handles.append(self.update_gate.register_forward_pre_hook(gate('update_gate_input')))
            handles.append(self.reset_gate.register_forward_pre_hook(gate('reset_gate_input')))
            handles.append(self.new_transform.register_forward_pre_hook(gate('new_transform_input')))
            x=encode(self,image,ids,train_data,*args,**kwargs)
            frozen.require(tuple(x[0].shape[:2])==tuple(x[2].shape[:2])==(1,32) and tuple(x[1].shape)==(1,32),
                           'Visual projection shape or mask changed')
            put('visual_mask',digest(x[1]))
            return x
        finally:
            for h in handles:
                h.remove()

    cls.encode_img,cls.fusion_train=audited_encoder,audited_fusion
