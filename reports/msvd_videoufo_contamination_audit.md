# MSVD ↔ VideoUFO contamination audit

Date: 2026-11-20. Read-only; no GPU.

## Source IDs

MSVD filenames `<youtube_id>_<start>_<end>.avi` parsed for all 1,970 videos:
1,491 unique YouTube sources (test670: 522 sources).

VideoUFO clip IDs embed `<youtube_id>.<segment>`; 345,437 sources.

| scope | MSVD sources | same source in VideoUFO | same source + ≥1 s time overlap |
|---|---|---|---|
| all 1,970 | 1,491 | **0** | **0** |
| test 670 | 522 | **0** | **0** |

`MSVD_VIDEO_SOURCE_LEAKAGE = NONE`. No cleaned corpus or manifest change is
required; MSVD retrieval can use the existing FULL VideoUFO manifest.

## Caption text overlap (normalized exact)

| split | unique MSVD captions | exact matches in VideoUFO | ratio |
|---|---|---|---|
| train | 38,256 | 40 | 0.10% |
| val | 3,563 | 6 | 0.17% |
| test | 22,285 | 33 | 0.15% |

Matches are generic phrases ("a man is talking") with no source linkage; the
rate is uniform across splits, i.e. vocabulary overlap, not clip duplication.

## Post-retrieval checker (prepared)

`scripts/audit_candidate_features.py` already reports candidate counts; the
MSVD retrieval chain will additionally flag: same-source candidates, exact
caption matches, and near-duplicate captions (ROUGE-L ≥ 0.8) once top-20 exists.
No expensive 812K × 670 semantic all-pairs is planned.
