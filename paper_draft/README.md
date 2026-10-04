# Paper draft

Open `main.pdf` for the compiled manuscript. Edit `main.tex`, the modular files under `sections/` and `appendix/`, and `references.bib` for revisions.

Build from this directory:

```bash
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

The manuscript follows the supplied experiment report. It does not report new experiments. Tables of extended intervals are transcribed programmatically from `../results/paired_comparisons.csv`. The dose figure is replotted from cached outputs with the original bootstrap convention; regenerate with `../.venv/bin/python figures/plot_dose.py` from this directory. Other result plots are copied from the experiment artifacts. The method diagram is native LaTeX.

The supplied `neurips_2025.sty` leaves its track-name macro undefined with the requested `[final]` option alone. `commands/macros.tex` defines a manuscript footer so that the exact requested package line compiles without implying conference acceptance.

The author block is exactly `Ari Holtzman and NeuriCo`. Bibliographic details derive from the supplied literature review and local paper manifest. Undated model cards use `n.d.` with the recorded access date. No example manuscripts were present in `paper_examples/`; its README was the only available file.

The evidence-mapped outline is `../paper/OUTLINE.md`. `REVIEW.md` records final checks; build logs and PDF page renders are retained for inspection.
