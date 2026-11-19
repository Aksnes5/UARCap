# FAITHFUL_V2 — preprocessing fix, pre/post retrieval overlap, CF3/D3/H3 rerun

Date: 2026-11-19. dev300, 300 queries, same EVCap checkpoint, deterministic decoding.

## 1. Confirmed preprocessing deviations and fixes

| Path | PRE_FIX_DIAGNOSTIC | FAITHFUL_V2 |
|---|---|---|
| MSR-VTT query X-CLIP resize | `cv2.INTER_LINEAR` | `cv2.INTER_NEAREST` (official X-CLIP) |
| VideoUFO Video-LLaMA fusion frames | center-of-bin (16 center frames) | `linspace(0, N-1, 16)` endpoints |
| Frames / size / norm / checkpoint | 16, 224, ImageNet, same ckpt | unchanged |

New artifacts: `embeddings/msrvtt_test_video_xclip_faithful_v2_*`,
`features/videollama_faithful_v2/` (1,781 clips, 0 decode failures),
`external/external_v2_{sims,top20,reranked}.json`.

## 2. Pre/post FULL retrieval overlap (dev300)

| Metric | PRE_FIX | FAITHFUL_V2 | Δ |
|---|---|---|---|
| top1 caption changed | — | 152/300 = 50.7% | — |
| top20 set overlap | — | 0.6222 | — |
| top3 set overlap | — | 0.5411 | — |
| top1 similarity | 0.2820 | 0.2782 | −0.0038 |
| top20 mean similarity | 0.2686 | 0.2645 | −0.0041 |
| oracle@20 (CIDEr) | 0.2189 | 0.2155 | −0.0034 |
| oracle@7 | 0.1548 | 0.1596 | +0.0048 |
| oracle@3 | 0.1142 | 0.1154 | +0.0012 |
| top1 caption CIDEr | 0.0762 | 0.0753 | −0.0009 |
| rerank top1 CIDEr | 0.0981 | 0.1005 | +0.0024 |

The fix reshuffles candidates substantially but leaves retrieval quality flat.

Action-phrase coverage had to be rebuilt over the union of PRE_FIX + v2 + val497
captions (14,641 strings) because v2 introduced ~9k unseen captions; otherwise
`KeyError` or silent zero action scores.

X-CLIP grounding features were completed for all v2-selected clips
(297 newly fetched); after that S_grounding changed the UAR selection
(1,780 → 1,752 clips).

## 3. Six metrics (×100)

| | B1 | B4 | M | R-L | CIDEr | SPICE |
|---|---|---|---|---|---|---|
| CF3v2 | 86.33 | 46.66 | 31.83 | 64.69 | 65.57 | 8.49 |
| D3v2 | 85.95 | 47.62 | 32.53 | 65.11 | **67.09** | **8.77** |
| H3v2 | 86.17 | 46.96 | 32.22 | 64.94 | 66.32 | 8.62 |
| H3 (PRE_FIX) | 87.10 | 48.29 | 32.28 | 65.09 | 67.63 | 8.77 |
| D7 (PRE_FIX) | 84.33 | 44.42 | 31.59 | 63.98 | 63.25 | 8.55 |
| CF7 (PRE_FIX) | 85.15 | 45.42 | 31.80 | 64.19 | 64.38 | 8.77 |

## 4. Paired per-query CIDEr differences (10k bootstrap, 10k sign-flip permutation)

| Comparison | Δ CIDEr | 95% CI | p | W/L |
|---|---|---|---|---|
| H3v2 − CF3v2 | +0.75 | [−1.0, +2.8] | 0.472 | 32/35 |
| H3v2 − D3v2 | −0.77 | [−2.7, +1.1] | 0.429 | 47/53 |
| H3v2 − H3pre | −1.31 | [−3.3, +0.6] | 0.196 | 31/44 |
| H3v2 − CF7 | +1.94 | [−0.4, +4.5] | 0.133 | 60/53 |
| H3v2 − D7 | +3.07 | [+0.3, +5.9] | **0.032** | 78/59 |
| D3v2 − CF3v2 | +1.52 | [−0.7, +4.1] | — | — |

## 5. Conclusion

Under the faithful preprocessing and the like-for-like k=3 comparison, the
PRE_FIX H3 advantage does **not** reproduce: H3v2 (66.32) is numerically below
D3v2 (67.09) and statistically indistinguishable from both D3v2 and CF3v2.
The PRE_FIX 67.63 was not solely a preprocessing artifact (H3v2 − H3pre = −1.31,
p=0.20), but the gain over reranking-only (D) is gone, and even D3v2 − CF3v2 is
not significant. Therefore the previous claim "H3 > D" is withdrawn; the k=3
regime is dominated by candidate-set quality, with no measurable UAR gain at
this sample size.

Do not run E/F/G component ablations for a "win" until a k>3 or larger dev set
shows a reproducible H−D gap.
