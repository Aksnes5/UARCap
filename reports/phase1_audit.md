# Phase 1 audit — EVCap video paper + VideoUFO external retrieval

Date: 2026-11-19. Read-only audit; no pipeline code changed.

## 1. Repositories

| Role | Path |
|---|---|
| Paper-final EVCap code (used for Table 1 reproduction) | `/media/bn/37b77782-.../xyl/Display/lj/EVCapCode` |
| X-CLIP retrieval code | `/media/bn/37b77782-.../xyl/Display/EvcapProject/X-CLIP` |
| Reproduction harness / venv / Tables | `/media/bn/37b77782-.../xyl/Display/repro_evcap` |
| Checkpoint | `outputs/train_evcap_msrvtt_quality_text/train_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review.pt` (147,337,150 B) |

The paper-final code copy is on the 8 TB ext4 volume, currently mounted
`ro,relatime,norecovery`; all new artifacts are written to
`/home/bn/videoufo_research/`.

## 2. Data flow (MSR-VTT, paper method)

```
offline  X-CLIP (ViT-B/16, 16 frames, Kinetics-600 k600_16_16.pth)
         query video --encode_video--> v (512)
         corpus captions --encode_text--> T (512)
         score = v . (T + VideoSpecificPrompt(T, frame_tokens))
         top-20 captions
         -> msrvtt_train_allGT_avg_frames_xclip_retrieve_20_captions_no_duplicate_sim_scores.json

offline  quality rerank (ROUGE of each caption vs the other 19, sentence-as-inference)
         -> rerank_sentence_by_sentence_as_inference_rouge_quality_no_duplicate.json
         -> make_with_video_ids.py attaches source video ids
         -> ..._with_video_ids.json  (3-element: captions, scores?, video_id lists)

online   EVCap eval
         ext_path JSON -> self.cap
         hardcoded sim_scores JSON -> self.sims_scores
         topn=7 -> prompt_wrap
         top-7 Video-LLaMA 32x768 features -> fusion_train -> cross_attn -> cross-gating
         Vicuna-7B-v1.3 beam search -> caption -> cocoeval
```

**Key finding.** §3.3 of the paper says X-CLIP retrieves *captions* from a text
corpus.  The corpus videos are never embedded for retrieval.  The faithful
retrieval artifact for VideoUFO is therefore 1,091,712 caption text embeddings,
not 1,091,712 video embeddings.  Video features enter only in cross-gating, for
the top-K retrieved clips.

## 3. Item map

| # | Item | Location |
|---|---|---|
| 1 | corpus load | offline `X-CLIP/main_all_no_duplicate.py` (`Text/msrvtt_all_train_text_no_duplicated.json`); online `--ext_path` + hardcoded sim-scores JSON |
| 2 | X-CLIP video embedding | `X-CLIP/models/xclip_all.py::XCLIP.encode_video` (line 107), `forward` (142) |
| 3 | top-k | `X-CLIP/main_all_no_duplicate.py:147` `tot_similarity.topk(config.TOPK, -1)` |
| 4 | top-20 -> rerank -> top-7 | retrieve JSON -> `ext_data/pre/phrase/select/rerank_sentence_by_sentence_as_inference_quality.py` -> `make_with_video_ids.py`; eval `--topn 7`, `prompt_wrap` `[:self.topn]` |
| 5 | quality rerank | same rerank script; scores from `ext_data/pre/phrase/compute_word_weight/rouge_cider_phrase_use_sentence_as_inference.py` |
| 6 | cross-gating | `models/..._prompt5.py::encode_img` (update_gate/reset_gate/new_transform/fusion_llama_proj); `fusion_train` softmax(sim) weighted features + cross_attn |
| 7 | Video-LLaMA / Vicuna | `init_video_Qformer`, `video_Qformer` in `encode_img`; `LlamaForCausalLM` Vicuna-7B-v1.3; `search.py::beam_search` |
| 8 | MSR-VTT eval | `eval_..._prompt5.py`, `datasets/MSRVTT-v2/test_16frames.yaml`, `dataset/msrvtt/vl_dataloader.py`, `evaluation/cocoeval.py` |
| 9 | metrics | `evaluation/pycocoevalcap/eval.py`: Bleu 1-4, METEOR, ROUGE_L, CIDEr, SPICE |
| 10 | checkpoint reusable | yes; reproduced Table 1: 86.69 / 50.11 / 32.47 / 66.64 / 72.20 |

## 4. Minimal code changes for external retrieval

1. `models/..._prompt5.py:124` — make `sim_scores_path` a constructor argument.
2. `models/..._prompt5.py:330` — feature path is hardcoded
   `ext_data/video_query/{selected_video_id[26:-4]}.pt`; add a configurable
   directory + explicit VideoUFO clip-id -> feature-file mapping.
3. `models/..._prompt5.py:327` — `random.choice(corresponding_video_ids)` makes
   evaluation stochastic; choose deterministically (first/best) for experiments.
4. `eval_..._prompt5.py` — pass `--sim_scores_path`, `--video_feature_dir`,
   and a `--use_fusion` switch for the naive baseline.
5. New offline scripts (no model change): external retrieval JSON, quality
   rerank, Video-LLaMA feature cache.
6. Use the non-MSDV model pair (`..._prompt5.py`), not `..._prompt5_msvd.py`:
   the latter hardcodes `ext_data/msvd_video_query/...` and silently falls back
   to zero features for MSR-VTT ids.

## 5. Frozen manifests

`manifests/clips_{10k,50k,100k,250k,500k,full}.tsv`,
`manifest_stats.json`, `SHA256SUMS`.  Construction: source-hash order,
round-robin across sources within each topic, proportional topic allocation,
prefixes are strictly nested (verified set inclusion).  See README.

## 6. Artifacts built

- `embeddings/videoufo_brief_text_fp16.npy`: 1,091,712 x 512 fp16, 1.1 GB,
  228 s (X-CLIP `encode_text`, 16-frame ViT-B/16 checkpoint).
- `embeddings/msrvtt_test_video_fp16.npy`: 2,990 x 512 MSR-VTT test queries.
- `indexes/videoufo_brief_full_flat.faiss`: 2,132 MB, 2.5 s build.
- `indexes/videoufo_brief_full_ivf4096.faiss`: 2,149 MB, 145.9 s build;
  nprobe 16: 0.237 ms/query, recall@20 94.3%; nprobe 32: 0.429 ms, 96.6%.
