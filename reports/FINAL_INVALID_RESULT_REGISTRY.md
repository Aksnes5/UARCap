# FINAL INVALID / SUPERSEDED RESULT REGISTRY

Never cite these in the paper. Original files are preserved for provenance.

| Old result | Why invalid/superseded | Replacement |
|---|---|---|
| PRE_FIX H3 67.63 > D (dev300) | Query resize INTER_LINEAR, fusion features center-of-bin; no same-k D baseline | FAITHFUL_V2: CF3v2 65.57, D3v2 67.09, H3v2 66.32; H3−D3 −0.77 p=0.43 |
| PRE_FIX scale 10K 63.99 / 100K 64.98 / 500K 63.31 / FULL 63.25 | Same preprocessing deviations | val497 FAITHFUL_V2 six-scale D/CF grids |
| Approximate FAISS top-20 sanity runs | Approximate vs exact scores | Exact X-CLIP scoring used throughout final results |
| Oracle1/3/7 and Good/Hard/Easy built with global-batch Cider | Per-candidate Cider degenerates when scored alone (idf=0); batch-dependent | Fixed-DF scorer (reference corpus 9,940 GT sentences); results 55.04/55.18/53.23 and Good 52.71 / Hard 48.01 / Easy 48.98 |
| Factorial means −6.48 (text) / +0.27 (visual) | Unscaled sum contrasts, not averaged marginal effects | Averaged: text −3.24, visual +0.13, interaction −0.78 |
| "FULL D3 53.14 is the overall best fixed configuration" | Wrong global max | Global max 50K CF3 53.57; D-track max 250K D1 53.20 |
| "Best k decreases monotonically with corpus size" | Data non-monotone (5/3/1/1/3/3 D; 5/3/1/3/3/1 CF) | "Large budgets rarely optimal; exact optimum scale-dependent" |
| First LOO runs (fail before lineidx fix) | Dataset loader error; no outputs | LOO 150×7 rerun (1050 counterfactuals) |
| First in-domain feature/run attempts (cv2-before-torch segfault; dtype/shape errors) | No valid outputs | FAITHFUL_V2 features + D1/3/5/7 rerun |
| First replicate B/C/D retrieval attempts (manifest newline KeyError) | No outputs | Replicates rerun successfully |
| Legacy in-domain `ext_data/video_query` features for matched control | Not proven equivalent to FAITHFUL_V2 pipeline | Re-extracted FAITHFUL_V2 candidate features |
| Old chat-log numbers (67.63, 64.98, 72.2 etc. in different contexts) | Mixed pre-fix/scales; not canonical | FINAL_NUMBERS_FOR_PAPER.md only |
