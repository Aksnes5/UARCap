# CPU-only residual scientific audit

Formal home of the post-freeze, CPU-only residual audit of already-generated
UARCap outputs. All paths in this README are repo-relative.

## Scope

CPU-only post-freeze audit of already generated outputs: frozen generated
captions, frozen metrics, retrieval candidate pools, references, and the
generation dataloader code. The audit explains four residual questions and
records one blocked item:

| Issue | Status | Report |
|---|---|---|
| SPICE↑ vs CIDEr↓: length / style / copying | RESOLVED (descriptive) | `reports/cpu_audit/STYLE_LENGTH_COPYING_AUDIT.md` |
| Oracle pool depletion | PARTIAL | `reports/cpu_audit/ORACLE_POOL_DEPLETION_AUDIT.md` |
| val497 vs test2690 CIDEr gap | RESOLVED (EXPLAINED) | `reports/cpu_audit/VAL_TEST_CIDER_GAP_AUDIT.md` |
| Matched source×budget interaction | RESOLVED (secondary) | `reports/cpu_audit/SOURCE_BUDGET_INTERACTION_AUDIT.md` |
| Matched k=0 | BLOCKED_REQUIRES_GENERATION | summary, §5 |

Frozen reference commit: `e42c6953b29f9983141e814b17922cb2f7d99c5a`.
Branch base: `origin/main` at `c31dc82` (adds documentation only).

## No model reruns

This audit did **not**:

- run model generation;
- run training or fine-tuning;
- call CUDA;
- recompute X-CLIP embeddings or any query/clip feature;
- overwrite, move, or delete any frozen efficacy output or frozen statistic.

Only CPU re-evaluation of frozen text and metadata was performed. Frozen
`metrics.json` values are quoted as-is; audit recomputations are recorded
separately and never replace them.

## Key findings

### 1. Style / copying

On the two splits with intact visual input:

- MSR-VTT test2690 mean output length: **7.91 → 8.74 words** (k=1→7).
- MSVD670 mean output length: **6.77 → 7.61 words**.
- 3-gram copying from the supplied retrieved captions:
  **0.142 → 0.334** (MSR-VTT), **0.138 → 0.387** (MSVD).
- Exact verbatim copies of a supplied caption:
  **1.1% → 6.0%** (MSR-VTT), **0% → 3.9%** (MSVD).
- The same k-sweep shows SPICE↑ and CIDEr↓ (CIDEr 71.21→68.27 MSR-VTT;
  156.39→141.59 MSVD).

This is **descriptive only, not causal**. Within a condition, longer outputs
correlate with lower CIDEr and more evidence overlap correlates with higher
CIDEr; the between-k trend is a separate, ecological pattern.

### 2. val497 frame-input bug

The val497 caption generator did **not** see the 16-frame video:

- `evcap_external/val497/val16_video.yaml` supplies `img:` as a **video-path
  TSV** (`val_video_paths.tsv`), not a pre-extracted frame TSV.
- The eval branch of `VisionLanguageTSVDataset.__getitem__` sets
  `start, end = 0, 0`.
- `pyav_decode` turns 0/0 into a one-frame decode window; PyAV retains only
  1–2 frames.
- `temporal_sampling` then replicates the **first frame** to fill the 16-frame
  input (empirically 15 copies of frame 0 + 1 copy of frame 1).
- test2690, dev300 and MSVD use pre-extracted frame TSVs, whose loader branch
  ignores `start`/`end` and returns 16 distinct frames; they are unaffected.

Consequences:

- **val497 absolute CIDEr (≈50–55) must not be compared** with test2690,
  MSVD, or published scores.
- All val497 conditions share the same broken visual input, so **within-val
  paired contrasts remain internally interpretable**; their magnitude and
  generalization to a correctly framed val re-run must be qualified.
- The val497-versus-test2690 gap is a pipeline artifact, not evidence about
  external retrieval or split difficulty.

Full source-level trace: `reports/cpu_audit/VAL_FRAME_INPUT_BUG_CODE_TRACE.md`.
The code was **not** fixed and val497 was **not** re-run in this audit.

### 3. Source×budget interaction

Matched val497, n=497 queries, 361 YouTube sources:

- external D3−D7: **+3.08 CIDEr points** [0.70, 5.68], p = 0.0138.
- in-domain D3−D7: **−0.29** [−2.31, 1.69], p = 0.78.
- interaction (external − in-domain): **+3.37** [0.30, 6.69], p = 0.0365.
- source-cluster bootstrap 95% CI: **[0.27, 6.88]**.

This is a secondary analysis of frozen outputs. It does not modify or replace
the two original separate tests.

### 4. Oracle pool depletion

Using the frozen unigram fixed-reference TF-IDF scorer on the existing top-20
pool:

- ranks 1–3 compatibility mean **0.1380**; ranks 4–7 mean **0.1031**; a
  **25.3%** relative drop; the per-query difference is positive in
  **497/497** queries.
- no special 3→4 cliff: 3→4 drop **0.0092** vs 1→2 drop **0.0209** (the
  decline is smooth and monotone).
- depletion does **not** predict the per-query Oracle3−Oracle7 CIDEr loss:
  Spearman ≈ **−0.04**; high- versus low-depletion halves ≈ **1.71 vs 2.19**
  CIDEr points.

Interpretation: candidate depletion is a **real contributor**, but not a
sufficient primary explanation. Integration overload cannot be excluded.

### 5. k=0

- No matched current-pipeline k=0 output exists for val497, test2690 or MSVD.
- The only `topn=0` run is `eval_dev300_A` (CIDEr 64.18), which is
  `PRE_FIX_DIAGNOSTIC`, pilot300-only, uses `external_empty_cap.json` and
  `video_feature_dir=None`; it is **ineligible** as a matched k=0.
- Status: **BLOCKED_REQUIRES_GENERATION**. Do not substitute the Fang et al.
  70.8 number.

## File map

Reports:

- `reports/cpu_audit/STYLE_LENGTH_COPYING_AUDIT.md`
- `reports/cpu_audit/style_length_copying_summary.csv`
- `reports/cpu_audit/ORACLE_POOL_DEPLETION_AUDIT.md`
- `reports/cpu_audit/VAL_TEST_CIDER_GAP_AUDIT.md`
- `reports/cpu_audit/SOURCE_BUDGET_INTERACTION_AUDIT.md`
- `reports/cpu_audit/CPU_ONLY_RESIDUAL_AUDIT_SUMMARY.md`
- `reports/cpu_audit/VAL_FRAME_INPUT_BUG_CODE_TRACE.md`

Machine-readable evidence:

- `reports/cpu_audit/style_length_copying_full.json`
- `reports/cpu_audit/cpu_audit_valtest_forensics.json`
- `reports/cpu_audit/val_frame_input_bug_evidence.json`
- `reports/cpu_audit/oracle_pool_depletion.json`
- `reports/cpu_audit/oracle_rank_compatibility.csv`
- `reports/cpu_audit/source_budget_interaction.json`
- `reports/cpu_audit/source_budget_interaction_per_query.csv`

Sample frames (visual evidence for the static-first-frame bug):

- `reports/cpu_audit/frames/*.png`

Audit scripts (CPU-only; not imported by any frozen pipeline):

- `scripts/cpu_audit/perquery_cider.py`
- `scripts/cpu_audit/cider_fixed.py`
- `scripts/cpu_audit/style_audit.py`
- `scripts/cpu_audit/valtest_forensics.py`
- `scripts/cpu_audit/frame_input_evidence.py`
- `scripts/cpu_audit/oracle_depletion.py`
- `scripts/cpu_audit/source_budget_interaction.py`

## Reproduction

The scripts are preserved as run. They read frozen outputs from the research
workspace and write only new files. They require a CPU Python environment with
`numpy` and the repository's local `pycocoevalcap` copy. They contain the
server paths used for the audit (see "Sensitive data" below); adjust paths if
run elsewhere.

## Sensitive data

- No passwords, tokens, API keys, SSH private keys, or personal credentials are
  present in the added files.
- Server absolute paths appear only where they are part of the provenance
  evidence (YAML `img`/`label`/`caption` fields in
  `val_frame_input_bug_evidence.json`, and `ROOT`/dataset paths in the audit
  scripts). They are kept as the minimal reproducible provenance; this README
  uses repo-relative paths.
- No frozen scientific evidence was removed.

## Transfer note

Audit results are unchanged from the CPU-only run. Only path references inside
the committed Markdown files were adjusted to the `reports/cpu_audit/` location.
No number, CI, p-value, or interpretation was altered during transfer.
