# Phase 2 results — dev300 (partial: B and D complete)

Fixed 300 MSR-VTT test queries (`dev300_queries.txt`), same checkpoint,
frames, beam width 5, seed 42, deterministic selection.

| group | B1 | B2 | B3 | B4 | METEOR | ROUGE-L | CIDEr | SPICE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B in-domain | 84.32 | 72.90 | 60.08 | 48.10 | 31.63 | 64.23 | **66.51** | 8.10 |
| D external full | 84.33 | 70.81 | 57.13 | 44.42 | 31.59 | 63.98 | **63.25** | 8.55 |
| D-B | +0.01 | -2.09 | -2.95 | -3.68 | -0.04 | -0.25 | **-3.26** | +0.45 |

Original (randomized) reference on the same 300 queries: B1 85.30, B4 47.90,
METEOR 31.50, ROUGE 64.70, CIDEr 67.40, SPICE 8.30. Deterministic B is within
~1 point (protocol effect; 62.3% of in-domain queries have multi-video captions).

## Retrieval-only diagnostic (top-1 retrieved caption vs GT)

| corpus | CIDEr |
|---|---:|
| MSR-VTT train (in-domain) | 0.272 |
| VideoUFO FULL dedup | 0.130 |

External retrieval quality is roughly half of in-domain at the caption level.

## Per-query D vs B

- D wins 79, loses 87, ties 134; mean CIDEr 0.633 vs 0.665.
- Similarity/CIDEr correlation: external 0.162, in-domain 0.155 (weak both).
- Average caption length: B 7.21 words, D 8.68 words (external makes captions
  longer and more specific; SPICE up, n-gram metrics down).

## Qualitative

D better: video-game action, fashion show, bowling (external caption names the
specific action/scene that B missed).
D worse: motorcycle vs car, award ceremony vs singing, monsters vs Star Wars
discussion, dark hallway vs horror trailer (scene/style match, action wrong).
Cases in `reports/D_vs_B_cases.md` / `.json`.

A (no retrieval) and C (naive top-20) are still running; preliminary C outputs
are numbered lists ("1. ... 2. ..."), a format failure caused by the top-20
prompt. A with a neutral/generic prompt also degenerates to numbered lists;
the original Video-LLaMA baseline checkpoint is not available, so A is only an
approximation with the same EVCap checkpoint.

## Naive external / no-retrieval baselines (partial + caveat)

The EVCap checkpoint was trained with the fusion branch enabled. Running the
same checkpoint with `--use_fusion False` is out of distribution: the generator
degenerates into numbered list copying.

- C (top-20 naive, no fusion, 35/300 completed before stopping):
  CIDEr **0.058**, ROUGE-L **0.171** on the completed subset; **63%** of
  outputs start with "1." / enumerate multiple captions.
- A (generic prompt, no fusion, 6/300): CIDEr **0.007**; 67% numbered.
- These are not valid paper baselines with this checkpoint; the original
  Video-LLaMA baseline checkpoint is not present in the repo.

A corrected no-retrieval baseline (fusion branch active, empty retrieval) was
started after these observations; see the final report for its metrics.
