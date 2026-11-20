# MSVD existing-data audit (Display workspace)

Date: 2026-11-20. Read-only inspection; no download, no GPU impact.

## 1. Canonical path (from code/config)

The original EVCap code references MSVD **relative**:
`datasets/MSVD/train_16frames.yaml` / `test_16frames.yaml`
(`train_evcap_..._msvd.py`, `eval_..._prompt5_msvd.py`, `scripts/*msvd.sh`).
From the project root `lj/EVCapCode`, `datasets` is a symlink →
`/media/.../Display/lj/datasets`.

**Canonical dataset path:** `/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/lj/datasets/MSVD`

A second complete copy exists at
`/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/EvcapProject/datasets/MSVD`
(identical structure, separate inodes — not hardlinks; both 1.8 G videos).

## 2. Raw videos

| item | value |
|---|---|
| `videos/` | 1,970 `.avi` files, 1.8 GB |
| `.mp4` | 0 |
| split ↔ video consistency | 1200 train + 670 test + 100 val = 1,970 unique ids; every id has a video; no unused videos |

## 3. Annotations and splits

| split | caption/label rows | frames |
|---|---|---|
| train | 1,200 | 1,200 × 16 = 19,200 JPEG (1.6 GB) |
| val | 100 | 100 × 16 = 1,600 JPEG |
| test | 670 | 670 × 16 = 10,720 JPEG (835 MB) |

Extra files: `frame_tsv/{test,train}_16frames_img_size256.img.tsv` (16 frames per
video, base64 JPEG), `*.caption_coco_format.json`, linelist/lineidx,
`trainandtest.caption.tsv`. `val` is a separate 100-video split (disjoint from
train/test). Standard MSVD split sizes are intact.

## 4. Old MSVD experiment assets (verified present)

| asset | path | reusable |
|---|---|---|
| MSVD checkpoint (paper config, top3 prompt5) | `lj/EVCapCode/outputs/old/train_evcap_msrvtt_quality/train_evcap_top3_bs1_..._prompt5_msvd.pt` | **yes** |
| Reproduced test output (670) | `repro_evcap/outputs/old/eval_repro_msvd_test/` — CIDEr **1.583** (B1 91.5 / B4 69.7 / M 47.0 / R 84.0 / SPICE 8.8), matches paper 158.32 | yes, reference |
| In-domain retrieval JSONs | `ext_data/msvd_rerank_sentence_by_sentence_..._with_video_ids.json`, `msvd_train_allGT_avg_frames_xclip_retrieve_20_captions_...json`, `msvd_video_similarity_top20.json`, verb/noun JSONs | yes, but **in-domain only** |
| Video-LLaMA features for in-domain candidates | `ext_data/msvd_video_query/` 1,870 `.pt`, 183 MB | yes for in-domain; **not** for external retrieval |
| X-CLIP text corpus features | `EvcapProject/X-CLIP/Text/msvd/retrieve_all_no_duplicate/` per-video JSON | yes, in-domain text side |
| 16-frame preprocess | dataset frame dirs + img TSV | yes |
| Generated captions (many configs) | `outputs/eval_*msvd*` | reference only |

## 5. Answers to the required questions

1. **Real data path used by the original paper:** `lj/datasets/MSVD` (via
   `EVCapCode/datasets` symlink); duplicate at `EvcapProject/datasets/MSVD`.
2. **Raw videos still present:** yes — 1,970 `.avi`.
3. **≈1,970:** exactly 1,970.
4. **Annotations/splits intact:** yes — 1200/100/670, captions + labels + COCO
   format + 16-frame TSV/JPEG.
5. **Can it directly serve `MSVD query → FULL VideoUFO retrieval → frozen
   method`?** Partially. Videos, annotations, splits and the MSVD checkpoint are
   usable as-is. The retrieval and fusion sides are **not** ready:
   - FAITHFUL_V2 X-CLIP query features for the 670 MSVD test videos must be
     extracted with the current script (old features, if any, are not faithful-v2
     and not stored as a query matrix).
   - MSVD-query × FULL VideoUFO retrieval JSONs must be built (X-CLIP caption
     embeddings and the 1.09 M-clip corpus already exist).
   - Faithful Video-LLaMA features for the retrieved candidates must be
     extracted (only the in-domain top candidates exist today).
   - Frozen similarity-gap selector needs the new retrieval sims; its rule is
     parameter-free and already frozen.
6. **What must be regenerated:** the four items above; plus generation + metrics
   for MSVD test under external retrieval. All are compute-only, no new data.
7. **Any reason to re-download MSVD:** **no.** The complete dataset is present.

## 6. Compute estimate for the MSVD extension

- X-CLIP query features, 670 avi: ~2–3 min GPU.
- FULL VideoUFO retrieval for 670 queries: ~4–6 min GPU.
- Video-LLaMA faithful features for selected candidates (k=7 union): order
  few-thousand clips, ~10–20 min GPU.
- Generation: D1/D3/D5/D7 × 670 ≈ 4 × 10–20 min.
Total ≈ 1.5–2 h GPU, can run after the current test2690 grid.

No files were modified; nothing was downloaded.
