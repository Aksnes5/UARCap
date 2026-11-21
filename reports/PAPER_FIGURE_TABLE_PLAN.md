# Figure and table plan

## MAIN PAPER
- **Fig. 1** Framework: external corpus query → X-CLIP caption retrieval → D
  rerank → k budget → Video-LLaMA fusion → frozen generator.
- **Fig. 2** Scale × budget curves (2 panels: D and CF; x=k, lines=corpus scale).
- **Fig. 3** Coverage vs downstream utility (oracle@k and CIDEr vs k; candidate
  churn inset).
- **Fig. 4** Text×Visual 2×2 (four cells + averaged factorial effects).
- **Fig. 5** External vs matched in-domain budget curves.
- **Fig. 6** Normal vs Oracle budget curves (Oracle1/3/7 vs D1/3/7).
- **Table 1** MSR-VTT untouched2690 main results (B/D1/D3/D5/D7/Adaptive).
- **Table 2** MSVD cross-dataset results.
- **Table 3** Mechanism summary (2×2, position, in-domain, oracle, tail/distractor).

## SUPPLEMENT
- **Fig. S1** Good/Hard/Easy evidence diagnostic.
- **Fig. S2** LOO per-rank distracting effects (150 queries).
- **Fig. S3** Candidate survival/churn across nested scales.
- **Table S1** Six-scale D and CF grids (full).
- **Table S2** k=7 penalties and best-k counts.
- **Table S3** Source-cluster bootstrap vs query bootstrap.
- **Table S4** Invalid/superseded registry (methodological transparency).

## PROBABLY OMIT
- Individual per-query case studies from PRE_FIX diagnostics.
- Approximate-retrieval sanity plots.
- Any figure requiring raw 0–1 CIDEr; all axes in CIDEr points.
