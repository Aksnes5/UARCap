# MASTER PAPER HANDOFF

Canonical fact source for paper writing. All CIDEr numbers are points (×100).
Do not use values from chat logs, PRE_FIX_DIAGNOSTIC, or invalid-artifact lists.

## A. Paper one-sentence story
The original EVCap paper performs retrieval-augmented video captioning with a
small in-domain caption corpus, X-CLIP retrieval and ROUGE-based quality
reranking before cross-gated fusion. We replace that corpus with a frozen
million-scale external corpus (VideoUFO) and evaluate the same generator across
corpus scales and context budgets. Larger corpora improve retrieval-side access
but downstream caption quality does not improve accordingly; instead, large
numbers of external textual contexts consistently degrade generation.
Diagnostics locate the failure in imperfect evidence selection plus set-level
textual integration overload, specific to external evidence in this setting and
not explained by visual context count, prompt position, or individual toxic
tail items.

## B. Relation to prior paper

### Inherited from the original EVCap paper
- X-CLIP video-to-caption retrieval (k600_16_16 checkpoint).
- Quality-aware ROUGE-L reranking of retrieved captions.
- Retrieved caption → corresponding retrieved-video mapping.
- Video-LLaMA video features and cross-gating fusion module.
- Frozen Vicuna-7B-v1.3 caption generator and EVCap checkpoint.
- 16-frame input protocol and decoding recipe.

### New in this work
- Million-scale external retrieval corpus built from VideoUFO, with frozen
  nested manifests 10K/50K/100K/250K/500K/FULL.
- Six-scale retrieval/generation study with both raw CF and D (quality rerank).
- Context-budget (k=1/3/5/7) study across scales and rerankers.
- Untouched-test evaluation protocol (pilot300 excluded; test2690 confirmatory).
- Cross-dataset validation on MSVD670.
- Adaptive similarity-gap context-budget policy (frozen on val497).
- Matched in-domain vs external budget control.
- Fixed-DF oracle candidate-selection diagnostics.
- Text×Visual 2×2, prompt-position controls, tail-quality dose response,
  good/hard/easy distractor diagnostics, evidence-level leave-one-out.
- Nested-scale candidate-survival/churn analysis and source-cluster bootstrap.
- New claim: external-evidence context overload and its decomposition into
  selection vs integration bottlenecks.

Do not present inherited modules (X-CLIP, reranking, fusion, generator) as new
contributions.

## C. Dataset and corpus facts

### VideoUFO
- Clips: 1,091,712; unique YouTube sources: 345,437; 93 parquet metadata files.
- Original download: 200 tar shards, 804.77 GiB videos + 42.67 GiB metadata.
- Unique brief captions: 812,014 (CSV count); deduplicated caption index used by
  retrieval: 812,932 named groups in the frozen full manifest pipeline.
- Frozen nested manifests (clip rows): 10K=10,000; 50K=50,000; 100K=100,000;
  250K=250,000; 500K=500,000; FULL=1,091,712. Strict nesting verified.
- Manifest SHA-256 (short): 10k 141e775b, 50k b89fe74a, 100k 1436e5ff,
  250k 77ad4376, 500k aac9401d, full f2bacf23.
- Replicate manifests: repB50k b0da62ca, repC50k e1d11acf, repD250k a7e7430e
  (seeds frozen in configs/subset_replication_seeds.json; 4.6–4.8% overlap with
  canonical 50K).

### MSR-VTT
- Official split: train 6,513 videos (130,260 raw captions), val 497,
  test 2,990.
- pilot300 (used during exploration) and untouched2690 (confirmatory);
  untouched manifest SHA fce00da1, zero overlap, union = test.
- Internal selection split val497 used for all method choices.

### MSVD
- 1,970 `.avi` videos; train 1,200 / val 100 / test 670; 16-frame caches present.
- Source-level contamination audit: 0 YouTube-source overlap with VideoUFO
  across all splits; caption exact overlap 0.10–0.17% (generic phrases).
- Original in-domain test reference: CIDEr 158.32.

## D. Exact retrieval pipeline (Method-section ready)
1. Query video: 16 frames, center-of-bin sampling `idx=floor((i+0.5)N/16)`.
2. X-CLIP ViT-B/16 (`k600_16_16.pth`) video encoding; short side 256, center
   crop 224, ImageNet mean/std, INTER_NEAREST.
3. Candidate corpus: VideoUFO brief captions (caption retrieval, not video
   retrieval), one embedding per unique brief caption (512-d).
4. Top-20 by cosine similarity between query video embedding and caption
   embeddings, with the X-CLIP prompt generator applied to caption features.
5. D reranking: score each candidate by mean pairwise ROUGE-L against the other
   top-20 candidates; order descending (ties by original rank).
6. CF = raw X-CLIP order, no reranking.
7. k ∈ {1,3,5,7}: number of top-ranked captions passed to the generator; D and
   CF keep their own ordering.
8. Each retained caption carries its source clip ID(s); deterministic first
   sorted clip is used for fusion features.
9. Retrieved-video feature: Video-LLaMA video Q-Former, 32×768, from 16 frames
   at `linspace(0,N-1,16)`, 224, ImageNet normalization.
10. Cross-gating fusion of video and retrieved-video features (original module).
11. Generator: frozen EVCap checkpoint + Vicuna-7B-v1.3.
12. Decoding: deterministic clip choice, beam width 5, seed 42, max 16 frames,
    same prompt template across conditions.

Explicitly: stage one is **query video → caption retrieval**; corpus videos are
never embedded for retrieval.

## E. Faithfulness / preprocessing (final)
- 16 frames; X-CLIP INTER_NEAREST; short side 256; center crop 224; ImageNet.
- Query features: `msrvtt_{val,test}_video_xclip_faithful_v2_*`.
- Candidate fusion features: `features/videollama_faithful_v2/`
  (`linspace(0,N-1,16)`, 224, ImageNet norm).
- Checkpoint sha256 `a8b5af79…`; Vicuna-7B-v1.3 local.
- Beam width 5, seed 42, deterministic clip choice, fusion enabled.
- MSVD uses its original top3 prompt5 checkpoint
  (sha256 `93e3665b…`), same decoding settings.

Superseded pipelines that must not enter results: INTER_LINEAR query resize,
center-of-bin Video-LLaMA fusion features, PRE_FIX_DIAGNOSTIC retrieval caches.

## F. Frozen final results (canonical; CIDEr points ×100)

### Six-scale D grid
| scale | k1 | k3 | k5 | k7 |
|---|---|---|---|---|
| 10K | 50.49 | 51.46 | 52.40 | 50.07 |
| 50K | 51.85 | 52.45 | 51.31 | 50.14 |
| 100K | 52.90 | 52.60 | 50.89 | 51.10 |
| 250K | 53.20 | 51.60 | 51.40 | 51.80 |
| 500K | 52.83 | 52.96 | 51.49 | 51.52 |
| FULL | 51.77 | 53.14 | 50.51 | 50.05 |

### Six-scale CF grid
| scale | k1 | k3 | k5 | k7 |
|---|---|---|---|---|
| 10K | 51.22 | 52.05 | 52.30 | 51.52 |
| 50K | 52.72 | 53.57 | 52.90 | 51.32 |
| 100K | 52.17 | 51.74 | 51.63 | 51.69 |
| 250K | 52.22 | 52.50 | 51.59 | 50.97 |
| 500K | 52.36 | 52.91 | 50.91 | 50.42 |
| FULL | 53.20 | 52.42 | 51.29 | 51.13 |

Global max 50K CF3 53.57; D-track max 250K D1 53.20. k=7 optimal 0/12;
k=1/3 10/12; k=5 only at 10K; combined k7 penalty 1.83 (D 2.06, CF 1.60).

### MSR-VTT untouched2690
B 73.09 > D1 71.21 > Adaptive 70.67 > D3 70.08 > D5 69.48 > D7 68.27.
D3−D7 +1.81 (CI [0.87,2.75], p=2e-4); D1−D3 +1.14 (p=0.005);
Adaptive−D3 +0.58 (p=0.046); Adaptive−D1 −0.56 (ns); D3−B −3.02 (p<1e-4).

### MSVD670
In-domain reference 158.32; D1 156.39; D3 151.45; D5 144.39; D7 141.59;
Adaptive 151.94. D3−D7 +9.86 (p=1e-4); Adaptive−D3 +0.49 (ns).

### Adaptive budgeting
OOF val497 +1.03 (CI [−0.30,2.52], ns); test2690 +0.58 vs D3 (p=0.046);
vs D1 −0.56 ns; MSVD ≈D3. Status MIXED.

### Text × Visual 2×2
T3V3 53.14; T7V3 50.31; T3V7 53.66; T7V7 50.05. Text main −3.24
[−5.68,−0.97] p=0.007; visual main +0.13 ns; interaction −0.78 ns.

### Matched in-domain control
D1/D3/D5/D7 = 54.16/55.14/55.00/55.43 (D3−D7 −0.27 ns) vs external
51.77/53.14/50.51/50.05 (D3−D7 +3.11, p=0.012): EXTERNAL_EVIDENCE_OVERLOAD.

### Fixed-DF Oracle
Oracle1 55.04 / Oracle3 55.18 / Oracle7 53.23 vs D1 51.77 / D3 53.14 / D7 50.05.
+3.27 / +2.05 / +3.13; Oracle7−Oracle3 −2.03 (p=0.033).

### Position / tail / Good-Hard-Easy
posrev 50.14; top3-last 49.96 vs D7 50.05 (both ns). MidTail7 50.45;
LowTail7 51.00 (ns). GoodTail7 52.71 (+2.64, p=0.006); HardWrong7 48.01
(−2.02, p=0.022); EasyWrong7 48.98 (−1.04, ns); Hard−Easy −0.97 (ns).

### LOO Stage 1 (150×7)
Rank effects 1 +1.46; 2 −0.72; 3 +1.22; 4 +0.71; 5 +1.25; 6 +1.23; 7 +0.74.
Head +0.66 vs tail +0.98; medians 0; expansion gate not met.

## G. Six-tier mechanism verdicts
1. Selection bottleneck — **SUPPORTED**
2. External textual integration overload — **STRONG**
3. Position/primacy explanation — **NOT SUPPORTED**
4. Visual-fusion overload — **NOT SUPPORTED** (current setting)
5. Single toxic tail / rank-dependent item — **NOT SUPPORTED**
6. Plausible distractor — **MIXED**

## H. Safe claims
Large budgets are consistently suboptimal (k=7 never optimal in 12 settings;
penalty 1.83). Coverage rises with k/scale while utility does not. The dominant
channel is textual context count; visual 3→7 has no detected independent effect.
Prompt position does not explain it. Matched in-domain retrieval is flat, while
external degrades. Oracle selection helps (+2.0–3.3) yet Oracle7 < Oracle3.
Good tails help; hard distractors hurt; hard-vs-easy inconclusive. No single
toxic tail item. The pattern replicates on MSVD.

## I. Prohibited claims
First context-overload/adaptive-k/video-RAG claims; monotone scale→optimal-k;
"reranking has no effect"; "visual evidence can never hurt"; definitive
HardWrong>EasyWrong; adaptive always beats best fixed k; SPICE as semantic
proof; lower rank inherently worse; single toxic tail; external replaces
in-domain; first-version UAR independent gain.

## J. Limitations
In-domain control confounds source and corpus size (6,513 vs 1.09 M). Adaptive
was fit on one 497-query split; test2690 confirmatory edge is marginal. Oracle
and distractor conditions use GT-CIDEr for candidate selection only. Source-
cluster bootstrap assumes independent sources (2,004 sources for 2,690 queries).

## K. Canonical paths
reports/FINAL_NUMBERS_FOR_PAPER.md; reports/FINAL_CLAIM_MATRIX.md;
reports/FINAL_RESULTS_LEDGER.csv; reports/paper_claim_matrix.md;
reports/final_one_page.md; reports/mechanism_summary.md;
reports/scale_budget.md; reports/in_domain_vs_external_budget_control.md;
reports/oracle_fixeddf_audit.md; reports/msvd_external_frozen_validation.md;
reports/msrtt_test2690_confirmatory.md; reports/evidence_level_distracting_effect.md.

## L. Invalid / superseded
See `reports/FINAL_INVALID_RESULT_REGISTRY.md`. Pre-FIX results, approximate
FAISS sanity numbers, pre-fixed-DF Oracle/Good/Hard/Easy, unscaled factorial
−6.48/+0.27, failed LOO/in-domain/replicate attempts, and legacy in-domain
features are NOT citable.

## M. Fact precedence (highest first)
1. `reports/MASTER_PAPER_HANDOFF.md`
2. `reports/paper_claim_matrix.md` / `reports/FINAL_CLAIM_MATRIX.md`
3. `reports/final_one_page.md`
4. `reports/mechanism_summary.md`
5. Final result CSV/JSON (`FINAL_RESULTS_LEDGER.csv`, metrics files)
6. Conversation/history logs (never a source of numbers)

## N. Frozen commit
Final frozen-results commit before this package: `701827f3b66d1c5e5acb96f243b604b87db97ca0`.
The handoff-package commit follows it on `main`; no frozen number changes.
