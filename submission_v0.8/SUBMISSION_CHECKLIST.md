# Submission Checklist — UARCap v0.8

**Manuscript title:** *When Does External Retrieval Help Video Captioning? Evidence Budgets in a Million-Clip Memory*  
**Submission package:** `submission_v0.8/`  
**Target status:** Generic international journal submission package (priority target: SCI Q3; retains upgrade clarity for higher-tier journals)

---

## 1. Scientific & Statistical Quality Control (Verified)

- [x] All core numbers match immutable formal artifacts (`reports/v07_final/` and `reports/local_independent_review/results/`).
- [x] Experiment A primary contrast (`A1 − A0 = +1.40`, 95% source-cluster CI `[+0.45, +2.36]`, family Holm `p = .0066`, five-test Holm `p = .0165`) reported accurately as conditional net benefit.
- [x] `A7 − A0 = −1.54` (`[−2.94, −0.18]`, family Holm `p = .0271`, five-test Holm `p = .0813`) explicitly scoped to A-family adjustment only (never claimed to pass five-test Holm).
- [x] Experiment B (`B7HQ − B3 = −1.27`, family Holm `p = .0573`; `B7LQ − B3 = −1.13`, `p = .0612`; `B7HQ − B7LQ = −0.14`, `p = .7698`) accurately framed as directional/inconclusive without overclaiming confirmed general text overload or band equivalence.
- [x] Practical clarity criterion (`±1.0` CIDEr point) explicitly reported as **unmet** across all five prespecified contrasts.
- [x] `HQ`/`LQ` explicitly defined as relative X-CLIP similarity bands (mean gap `0.00835`), never human-verified semantic quality.
- [x] `A0` explicitly defined as the same retrieval-trained checkpoint with active fusion pathway and zero external tensors (not a retrained video-only baseline).
- [x] `A1`/`A7` exact 2,990-query output agreement with archived `D1`/`D7` explicitly framed as endpoint reproducibility, not an independent sample.
- [x] Raw-video validation decoder repair (`val497`/`val150`) transparently disclosed; zero invalid pre-fix validation claims restored.
- [x] Pilot-excluded MSR-VTT (`test2690`) and MSVD (`test670`) clearly described as previously analyzed benchmark material, not unseen external replication.
- [x] Cohort-specific reference IDF explicitly noted so Experiment A (`n = 2,690`) and Experiment B (`n = 2,618`) means are never compared directly across panels.
- [x] Distinction from Fang et al. (*Displays* 93, 2026, 103420) explicitly documented, including the full-caption ROUGE-L consensus approximation versus Fang et al.'s noun-phrase pseudo-label formulation.
- [x] All 31 citations verified against `UARCap_References.bib` (`REFERENCES_VERIFIED.bib`); zero fabricated references.
- [x] Pointwise 95% source-cluster CIs explicitly distinguished from simultaneous intervals.

## 2. Figures, Tables, and Deliverable Formats (Verified)

- [x] Five publication-grade figures (`fig1_framework` to `fig5_forest`) generated programmatically from immutable JSON/CSV sources (`figures/build_figures_v08.py`) in PDF, SVG, and 300-DPI PNG formats.
- [x] Colorblind-safe and grayscale-distinguishable styling verified visually across all five figures.
- [x] All six main tables (`Tables 1–6`) and seven supplementary tables (`Tables S1–S7`) exported to `tables/` and consistent across Markdown, DOCX, PDF, and LaTeX.
- [x] Manuscript delivered in Markdown (`UARCap_Manuscript_v0.8.md`), formatted Word (`UARCap_Manuscript_v0.8.docx`), PDF (`UARCap_Manuscript_v0.8.pdf`), and compilable LaTeX (`latex/main.tex`).
- [x] Supplementary Materials delivered in Markdown, DOCX, PDF, and LaTeX (`latex/supplementary.tex`).
- [x] Visual page-by-page PDF inspection completed for both Manuscript PDF and Supplementary Materials PDF (zero missing figures, clipped tables, or broken equations).

## 3. Author-Side Items Required Before Journal Portal Upload

- [ ] **Target journal selection:** Confirm specific journal (e.g., *Displays*, *Signal, Image and Video Processing*, *Multimedia Tools and Applications*, *Neurocomputing*, or *ACM TOMM*) and apply its final template/citation style (`elsarticle`, `sn-jnl`, `IEEEtran`, etc.).
- [ ] **Author list, order, affiliations, ORCIDs, and corresponding author email:** Replace `[TO BE COMPLETED BY AUTHORS]` placeholders in Manuscript, Supplementary Materials, LaTeX, and Cover Letter.
- [ ] **Funding & grant numbers:** Complete the Funding declaration.
- [ ] **Competing interests & shared-authorship disclosure:** Confirm competing interests and disclose shared authorship with Fang et al. (2026) if required by the selected journal's editorial policy.
- [ ] **AI-assisted writing disclosure:** Review and finalize the generative-AI disclosure statement per the target journal's author guidelines.
- [ ] **Repository archive URL / DOI:** Decide whether to keep the GitHub repository public or mint an archival Zenodo DOI (and confirm blind-review anonymization if submitting to a double-blind venue).
- [ ] **Cover letter customization:** Fill in `[Target Journal Name]`, `[Editor-in-Chief Name]`, and author signature in `UARCap_Cover_Letter_Draft.docx`.
