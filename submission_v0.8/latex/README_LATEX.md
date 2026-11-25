# LaTeX Compilation Guide (UARCap v0.8)

This folder contains self-contained, journal-neutral LaTeX source files for the main manuscript (`main.tex`), supplementary materials (`supplementary.tex`), verified BibTeX database (`UARCap_References.bib`, 31 references), and vector PDF figures (`figures/fig1_framework.pdf` through `figures/fig5_forest.pdf`).

## Standard Compilation (`pdflatex` + `bibtex`)

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

For Supplementary Materials:

```bash
pdflatex supplementary.tex
pdflatex supplementary.tex
```

## Notes for Journal Template Adaptation

- Author list, affiliations, corresponding email, and Declarations contain explicit `[TO BE COMPLETED BY AUTHORS]` placeholders.
- Replace `\documentclass[11pt,a4paper]{article}` and `\bibliographystyle{plainnat}` with the target journal class (e.g., `elsarticle`, `sn-jnl`, `acmart`, or `IEEEtran`) once the target venue is finalized.
