#!/usr/bin/env python3
"""Predicate/verb X-CLIP text embeddings over the UNION of all retrieval caption sets.

Covers PRE_FIX dev300, FAITHFUL_V2 dev300, and val497 candidates so H3 lookups by
caption string never miss. Outputs external/uar_action_union_*.{npy,json}.
"""
import json
import os
import sys

import numpy as np
import torch
import nltk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_uar_action import phrases_for, _Log  # noqa: E402

XCLIP = "/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/EvcapProject/X-CLIP"
CKPT = os.path.join(XCLIP, "modelpth/k600_16_16.pth")
ROOT = "/home/bn/videoufo_research"
SOURCES = ["external/external_sims_full.json", "external/external_v2_sims.json",
           "external/external_val_sims.json"]


def main():
    caps = set()
    for rel in SOURCES:
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            print("skip missing", rel, flush=True); continue
        S = json.load(open(p))
        caps |= {c for v in S.values() for c in v[0]}
        print(rel, len(S), flush=True)
    caps = sorted(caps)
    pred, verb = [], []
    for c in caps:
        p, v = phrases_for(c)
        pred.append(p)
        verb.append(v)
    print("unique captions", len(caps), "pred_nonempty", sum(bool(x) for x in pred), flush=True)
    sys.path.insert(0, XCLIP)
    os.chdir(XCLIP)
    import clip
    from models import xclip_all as xclip
    sd = torch.load(CKPT, map_location="cpu")["model"]
    model = xclip.build_model(sd, T=16, droppath=0.0, use_checkpoint=False, logger=_Log(),
                              prompts_alpha=0.1, prompts_layers=2, use_cache=True,
                              mit_layers=1, text_batch_size=0).cuda().eval()

    def encode(phr, fallback):
        out = np.zeros((len(phr), 512), dtype=np.float16)
        bs = 512
        for s in range(0, len(phr), bs):
            batch = [("a person is " + x) if x else fallback for x in phr[s:s + bs]]
            toks = clip.tokenize(batch, context_length=77, cut=True).cuda()
            with torch.no_grad(), torch.cuda.amp.autocast():
                e = model.encode_text(toks)
            out[s:s + bs] = e.float().cpu().numpy().astype(np.float16)
        return out

    np.save(os.path.join(ROOT, "external/uar_action_union_emb_fp16.npy"), encode(pred, "a person is doing something"))
    np.save(os.path.join(ROOT, "external/uar_verbonly_union_emb_fp16.npy"), encode(verb, "a person is doing something"))
    with open(os.path.join(ROOT, "external/uar_action_union_meta.json"), "w") as f:
        json.dump({"captions": caps, "phrases": pred, "verb_only": verb}, f)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
