# Phase 2 setup — external retrieval into the EVCap generator

## 1. Faithful external corpus / dedup

The paper's MSR-VTT caption list was built as `list(dict.fromkeys(captions))`
on `Text/msrvtt_all_train_text.json` (X-CLIP `1.ipynb`, cell 16): exact string
dedup, order preserved, no lowercasing or punctuation removal.

VideoUFO faithful external corpus therefore uses `Brief Caption.strip()` as the
exact key:

- 1,091,712 clips -> **812,932 unique captions**.
- each caption keeps **all** VideoUFO clip IDs that carry it (no random drop);
- deterministic fusion pick = lexicographically smallest clip ID in the sorted
  list (documented, saved in the JSON `clip_ids` order);
- the raw (non-dedup) diagnostics from Phase 1 are kept.

## 2. Writable code copy

`/home/bn/videoufo_research/evcap_external/` (copy of the paper-final code;
original 8 TB tree untouched and mounted `ro`).

Patches (see `reports/patches/`):

1. `models/..._prompt5.py`
   - `sim_scores_path` constructor argument;
   - `video_feature_dir` + external clip-ID feature naming;
   - `deterministic` selection (first ID) replacing `random.choice`;
   - `use_fusion` switch;
   - no-fusion prompt projection (`llama_proj(video_hidden)`) so the generator
     prompt remains dimensionally valid;
   - `no_retrieval` prompt path (single video block, default prompt).
2. `eval_..._prompt5.py`: passes `--sim_scores_path`, `--video_feature_dir`,
   `--use_fusion`, `--deterministic`, `--no_retrieval`.
3. `evaluation/` copied writable (PTBTokenizer writes temp files).

The original MSR-VTT in-domain run reproduces the first dev queries exactly;
the residual differences are caused by deterministic vs `random.choice`
selection (15.8% of in-domain top-7 caption slots map to >1 train video;
62.3% of queries are affected).

## 3. Fixed dev300

`dev300_queries.txt` = the exact 300 IDs from Phase 1 (`linspace` over the
2,990 test videos), no resampling. Dataset subset:
`evcap_external/dev300/` (filtered pre-extracted 16-frame TSV + lineidx +
label/caption TSV + yaml).

## 4. External retrieval JSON

`external/external_sims_full.json` (`[top20 captions, scores]`),
`external/external_reranked_full.json` (`[top7 captions, qualities, clip_ids]`),
`external/external_top20_full.json` (diagnostics), built with exact
X-CLIP scoring (VideoSpecificPrompt) over the 812,932 unique captions.
Quality rerank here is ROUGE-L caption-vs-caption (the paper's noun-phrase
decomposition was not reproduced because spaCy/en_core_web_lg is absent).

## 5. External Video-LLaMA features

`features/external_top7/` — 1,989 unique first-pick clips from the top-7,
[32,768] FP32 each, computed with the EVCap visual pipeline (16 uniform frames,
short-side 224 nearest, center crop 224, ImageNet normalize). Resumable,
manifest at `features/external_top7_manifest.json`.
