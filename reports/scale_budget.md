# Scale × context-budget interaction (val497)

Date: 2026-11-20. Same val497 queries, preprocessing, checkpoint, prompt, beam,
seed; only the VideoUFO manifest scale and k change. CF/D use identical ranking
code; D = ROUGE-L quality rerank.

## Generation CIDEr (×100)

| corpus | D1 | D3 | D5 | D7 | best k |
|---|---|---|---|---|---|
| 10K | 50.49 | 51.46 | **52.40** | 50.07 | 5 |
| 100K | **52.90** | 52.60 | 50.89 | 51.10 | 1 |
| 500K | 52.83 | **52.96** | 51.49 | 51.52 | 3 |
| FULL | 51.77 | **53.14** | 50.51 | 50.05 | 3 |

Other metrics are in `reports/scale_budget.json`; per-group captions/metrics are
committed under `evcap_cap/external/outputs/eval_val497_<scale>_D<k>/`.

## Candidate coverage (oracle CIDEr of the D ordering)

| corpus | oracle@1 | @3 | @5 | @7 |
|---|---|---|---|---|
| 10K | 0.0763 | 0.1055 | 0.1245 | 0.1361 |
| 100K | 0.1029 | 0.1388 | 0.1577 | 0.1707 |
| 500K | 0.1201 | 0.1564 | 0.1731 | 0.1834 |
| FULL | 0.1176 | 0.1481 | 0.1720 | 0.1835 |

Coverage grows from 10K to 100K and then saturates (500K ≈ FULL).

## Interpretation (what the data supports)

- **Supported:** a small corpus (10K) benefits from a larger evidence budget
  (best k=5), while 500K and FULL peak at k=3 and degrade at k≥5. The direction
  "more corpus → smaller optimal budget" holds between the extremes.
- **Not supported:** a monotone scale→budget law. 100K's optimum is k=1, but its
  D1−D3 difference (0.29 CIDEr) is within noise; the four-scale pattern is best
  described as "optimum ≤3 once the corpus is ≥100K, 10K is the exception".
- Retrieval coverage itself saturates above 100K, so extra corpus beyond 500K
  adds little candidate potential while adding redundancy pressure.

## Caveats

- Single 497-query validation split; the k=1 vs k=3 gap at 100K is not
  individually significant.
- FULL D3 (53.14) is the overall best fixed configuration across scales.
- This grid is characterization only; the frozen adaptive selector was not
  re-tuned on it.
