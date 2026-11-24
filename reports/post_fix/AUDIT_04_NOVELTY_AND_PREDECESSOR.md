# AUDIT 04 — Novelty vs the published predecessor

Predecessor: Fang et al., "Retrieval Augmented video captioning with
quality-aware re-ranking and cross-gating fusion", Displays 93 (2026) 103420
(local PDF at `/media/.../Display/1.pdf`, not redistributed in the repo).

## What the predecessor already did
- X-CLIP video-to-caption retrieval; retrieve **k=20** captions from the
  **training set of the same dataset** (in-domain).
- Quality-aware re-ranking: quality defined via **noun-phrase pseudo-labels**
  from the other retrieved captions, scored with a metric such as CIDEr/ROUGE-L
  (Section 3.4, Eq. 1).
- Select **k̃=7** for MSR-VTT and **k̃=3** for MSVD after re-ranking.
- Cross-gating fusion of retrieved-video features; structured prompt.
- **Table 4 already ablates the number of retained captions k̃**: moderate k̃=7
  is best in-domain, k̃=1 insufficient, larger k̃ dilutes context.
- Report MSR-VTT CIDEr 72.2 and MSVD 158.1/158.32.

## What this work adds
- External VideoUFO corpus at 10K–FULL (1.09M clips), nested manifests.
- Six-scale × {CF,D} × {1,3,5,7} study on a fixed generator.
- Untouched test protocol (pilot300 excluded), MSVD external validation.
- Candidate churn / similarity–utility decoupling analysis.
- Oracle fixed-DF diagnostics; Text×Visual 2×2; position; tail/distractor; LOO.
- In-domain vs external matched budget comparison.

## Direct overlap / risk
- The k-budget question itself is **not new**: predecessor Table 4 already asked
  how many re-ranked captions to keep and found k̃=7 best in-domain.
- Current work's new empirical claim is the **reversal**: external retrieval
  prefers k=1 (POST_FIX), while in-domain prefers k=7 (73.76). This is a
  distribution/source effect, not a new re-ranker or architecture.
- **Implementation mismatch**: the current D uses mean pairwise ROUGE-L over full
  captions, not the predecessor's noun-phrase pseudo-label quality. It must be
  described as an approximation, not as the published module.

## Verdict
Novelty is **conditional**: the scale study alone risks being read as a larger
re-run of the predecessor's k ablation. A defensible contribution must be:
(1) the in-domain/external budget-response reversal at million scale;
(2) the Oracle evidence that selection headroom exists and k=7 remains worse
under oracle ordering. Without those, the paper is an incremental scaling study.
