# Fixed-DF oracle scorer audit

## Definition
CIDEr compatibility (TF-IDF cosine, n-gram order 1–4, no length penalty) between
a candidate caption and the GT references of its query, with IDF fixed once from
the corpus of **9,940 GT reference sentences** (497 queries × 20 references;
5,924 unique tokens). Tokenization: lowercase alphanumeric surface tokens; this
differs from pycocoevalcap's PTB tokenizer and is recorded explicitly.

## Invariance / sanity
- Batch invariance: max abs difference 0.0 (`ORACLE_BATCH_INVARIANT = YES`).
- Score mean 0.0789, std 0.0720; zero-score fraction 0.0.
- 480 distinct top-3 oracle rankings across 497 queries; no all-equal query.

## Consequence
Fixed-DF rankings differ strongly from the previous global-batch Cider scorer:
top-3 set agreement only 44/497. Therefore all oracle-dependent artifacts built
with the old scorer are marked `INVALID_ORACLE_SCORER` and are being regenerated:
Oracle1/3/7, GoodTail7, HardWrong7, EasyWrong7.

MidTail7 / LowTail7 depend only on ROUGE quality ordering and are
`ORACLE-INDEPENDENT`; their existing outputs remain valid.
