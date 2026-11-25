# Manuscript Changelog: v0.7 → v0.8 Submission Edition

**Source draft:** `paper/UARCap_paper_v0.7.md` (commit `7e413fc9cef70ea1a9ca545ad5a1675e783768f9`)  
**Final manuscript:** `paper/UARCap_paper_v0.8.md` & `submission_v0.8/UARCap_Manuscript_v0.8.*`  
**Supplementary Materials:** `paper/UARCap_Supplementary_Materials.md` & `submission_v0.8/UARCap_Supplementary_Materials.*`  
**Immutable experimental data modified?** **NO.** All numbers derive directly from `reports/v07_final/` and `reports/local_independent_review/results/`.

---

## 1. Structural & Narrative Improvements

1. **Explicit RQ1 → RQ2 → RQ3 narrative spine:** Reorganized the Introduction, Results, Controlled Follow-up Experiments, and Discussion around three progressive scientific questions (*Candidate access → Matched net utility → Evidence extension and attribution*), eliminating the chronological/development-log feel of earlier drafts.
2. **Cleaner separation of main text and Supplementary Materials:** Moved detailed manifest audits, complete 7-contrast/6-contrast benchmark families, 26 repaired validation/LOO control rows, Experiment B matching balance, query-vs-cluster statistical tables, runtime SHA-256 verification summaries, raw-video decoder-repair provenance, and tokenization-reconciliation notes into a standalone, journal-ready `UARCap_Supplementary_Materials` document (Sections S1–S10, Tables S1–S7).
3. **Sharper positioning relative to Fang et al. (*Displays*, 2026):** Explicitly stated in Sections 1, 2.1, 3.2, and S2 what Fang et al. already established (architecture, cross-gating, in-domain budget sensitivity at MSR-VTT *k*=7 / MSVD *k*=3), what this paper independently adds (nested million-clip access–utility dissociation, matched zero-evidence net utility, verified visual-tensor text extensions, and clear attribution boundaries), and that our D ordering is a full-caption pairwise ROUGE-L consensus approximation to Fang et al.'s published noun-phrase pseudo-label formulation.
4. **Journal-standard Declarations section:** Added complete Author Contributions, Funding, Competing Interests, Ethics Approval, Data and Code Availability (with VideoUFO CC BY 4.0 and third-party video non-redistribution wording), and AI-Assisted Writing Disclosure with explicit `[TO BE COMPLETED BY AUTHORS]` placeholders.

## 2. Academic Prose & Readability Upgrades

1. **Eliminated staccato, telegraphic cadence:** Rewrote choppy 8–12 word sentence chains in v0.7 (notably in the Abstract, Introduction, Section 6, and Discussion) into fluid, natural international-journal academic prose with clear topic sentences and logical transitions.
2. **Removed internal engineering/audit terminology from main prose:** Replaced raw internal labels (`POST_FIX`, `saved engineering reproduction`, `archival record`, raw GitHub links inside the text body) with standard scientific phrasing while keeping every technical disclosure intact in Section 4.1 and Supplementary Materials S8–S10.
3. **Preserved every statistical boundary without overclaiming:** Retained full precision on all five prespecified contrasts (`A1−A0` surviving five-test Holm; `A7−A0` surviving A-family Holm only; `B7HQ−B3` and `B7LQ−B3` failing B-family Holm; `B7HQ−B7LQ` inconclusive; zero contrasts clearing the ±1.0 practical margin; cohort-specific IDF; pointwise vs. simultaneous CIs).

## 3. Figure & Packaging Upgrades

1. **Publication-grade figure redesign (`paper/v0.8_assets/build_figures_v08.py`):**
   - Adopted an Okabe–Ito-inspired colorblind-safe palette (`#1B4F72` navy, `#C04D00` vermilion, `#0E7C66` teal) combined with distinct marker fills (filled circle vs. hollow square) and line styles (solid vs. dashed/dotted) for grayscale/B&W print clarity.
   - Upgraded **Figure 1** into a crisp two-tier architectural & intervention diagram showing exact sample/source counts for Experiments A and B.
   - Upgraded **Figure 2** with distinct line/marker styles and explicit `Ref.` baseline indicator at 10K.
   - Upgraded **Figure 3** with inherited training-budget reference lines ($k=7$ MSR-VTT; $k=3$ MSVD), shaded MSVD extrapolation zone ($k>3$), Adaptive comparator reference line, and collision-free numeric annotations.
   - Upgraded **Figure 4** with cohort sizes/sources in panel titles and baseline dashed reference lines at A0 and B3.
   - Upgraded **Figure 5** with a horizontal group divider between Experiments A and B, exact `Effect [95% CI]` alongside both Holm *p*-value columns (bolding adjusted *p* < .05), and a clean non-overlapping legend below the plot.
2. **Complete multi-format submission package (`submission_v0.8/`):** Produced synchronized Markdown, formatted DOCX, Microsoft-Word-rendered PDF (visually checked page-by-page), compilable LaTeX (`main.tex` + `supplementary.tex` + `UARCap_References.bib` + PDF vector figures), exported CSV/Markdown tables, Elsevier-style Highlights, Cover Letter draft, Submission Checklist, and Claim–Evidence Matrix.
