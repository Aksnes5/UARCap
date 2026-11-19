#!/usr/bin/env python3
"""Predicate/action phrase + verb-only ablation, encoded with X-CLIP text."""
import json
import os
import sys

import numpy as np
import torch
import nltk

XCLIP = "/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/EvcapProject/X-CLIP"
CKPT = os.path.join(XCLIP, "modelpth/k600_16_16.pth")
ROOT = "/home/bn/videoufo_research"
AUX = {"is", "are", "was", "were", "be", "been", "being", "has", "have", "had",
       "do", "does", "did", "can", "could", "will", "would", "should", "may",
       "might", "must", "s", "am"}
OBJ = {"DT", "JJ", "JJR", "JJS", "PRP$", "PRP", "NN", "NNS", "NNP", "NNPS", "CD"}
STOP = {"in", "on", "at", "to", "from", "with", "by", "for", "of", "into", "onto",
        "through", "across", "along", "around", "under", "near", "behind", "down",
        "up", "off", "over", "out", "during", "while", "as", "than", "about",
        "and", "or", "but", "that", "which", "who", "whom", "whose", "when", "where"}
SKIP = {"a", "an", "the", "this", "that", "these", "those", "his", "her", "their",
        "its", "my", "your", "our", "some", "any", "another", "each", "every"}


class _Log:
    def info(self, m):
        pass


def phrases_for(c):
    toks = [t.strip('.,!?;:"()') for t in c.split()]
    toks = [t for t in toks if t]
    tags = [t for _, t in nltk.pos_tag(toks)]
    vi = next((i for i, t in enumerate(tags) if t.startswith("VB") and toks[i].lower() not in AUX), None)
    if vi is None:
        return "", ""
    verb = toks[vi].lower()
    verb_only = verb
    out = [verb]
    j = vi + 1
    if j < len(toks) and tags[j] == "RP":
        out.append(toks[j].lower()); j += 1
    if j < len(toks) and tags[j] in ("DT", "PRP$"):
        out.append(toks[j].lower()); j += 1
    k = 0
    while j < len(toks) and k < 3:
        w = toks[j].lower()
        if tags[j] in ("IN", "CC", "TO", "WDT", "WP", "WRB") or w in STOP:
            break
        if tags[j] in ("DT", "PRP$", "PRP", "CD") or w in SKIP:
            j += 1
            continue
        if w.isalpha():
            out.append(w); k += 1
        j += 1
    return " ".join(out), verb_only


def main():
    S = json.load(open(os.path.join(ROOT, "external/external_sims_full.json")))
    caps = sorted({c for v in S.values() for c in v[0]})
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

    np.save(os.path.join(ROOT, "external/uar_action_emb_fp16.npy"), encode(pred, "a person is doing something"))
    np.save(os.path.join(ROOT, "external/uar_verbonly_emb_fp16.npy"), encode(verb, "a person is doing something"))
    with open(os.path.join(ROOT, "external/uar_action_meta.json"), "w") as f:
        json.dump({"captions": caps, "phrases": pred, "verb_only": verb}, f)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
