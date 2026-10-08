# Version-42 figure notation update

Date: 2026-10-06. Scope: figure annotations and their generating code only.
Notation source: `mainaistats_42_notations (1).tex`, SHA-256
`c10828271f48e7d01426dcd78b026ac4078ab7ea74934eebbf47095894eb9191`.
The manuscript is not included in this anonymous code archive.

## Changes

The manuscript defines the radial quantile level as kappa and the survival
mass as `s = 1-kappa`. Both the main-text standalone coverage figure and
panel (b) of the three-panel tail figure now use:

- horizontal label: `survival mass $1-\kappa$`;
- vertical label: `$\widehat q_\kappa(|Y|)/q_\kappa(|T_\nu|)$`.

The two plotting functions and their generated caption are updated in
`experiments/exp1/plotting.py`. The radial Student quantile docstring in
`experiments/exp1/theory.py` now uses kappa; its arguments and computation
are unchanged. No blanket replacement of beta has been performed: beta
continues to denote the diffusion schedule in the simulation code/configs.

The stable-density symbols `\varphi_\alpha`, `\varphi_{\alpha,d}` and
`c_\varphi,C_\varphi`, and the Gaussian-density symbol `\phi`, are not
displayed in these plots. No density label therefore needs replacement.

Updated outputs, with names unchanged for the manuscript's existing includes:

- `figures/experiment1_quantile_ratio.pdf` and its PNG preview;
- `figures/experiment1_main.pdf` and its PNG preview.

All curves, uncertainty bands, sample selections, axis orientation, and
logarithmic scales are retained. Numerical data are not recomputed.

## Provenance and checks

The input `artifact_data/exp1/tail_summary.csv` retains SHA-256
`0569fd40f6a53271459887b7e2c44fefbb379e264903e2ab125538f929acaf70`.
All 36 files under `artifact_data/` and `configs/` are byte-identical to the
preceding MIT release. The Experiment 2 and Experiment 4 PDF/PNG files were
also regenerated as a control and are byte-identical to the preceding release.
No simulation was launched; no historical execution identity was relabelled.

New regression tests cover both quantile plot functions. They assert the
new labels, absence of beta in quantile labels, decreasing log-scale horizontal
axis, log-scale vertical axis, and exact preservation of the supplied median
curves and 16th/84th-percentile bands. Both tests failed on the previous beta
labels and passed after this edit.

Targeted validation: **31 tests passed** in 15.51 seconds (plotting, Experiment 1
analysis/theory, and artifact safeguards); Ruff passed. The existing 1,848
Matplotlib/PyParsing deprecation warnings are not numerical test failures.
The saved-statistics verifier reproduced all 24 initialization aggregate rows,
including bootstrap intervals, and 94 tail-figure rows. The complete numerical
suite was not rerun for this annotation-only revision; earlier full-suite
results are historical.

The two new PDFs were rendered with Poppler and visually inspected. No clipped
or overlapping labels were observed. PDF text extraction confirms kappa and
no beta in both outputs. PDF metadata contains only the generic software
creator/producer fields, with no author identity. The anonymous MIT notice is
unchanged. Figure-source hashes and `SHA256SUMS` are refreshed for this release.

This update does not certify the mathematical content or page count of the
version-42 manuscript. Reports named for version 40, the licensing receipt,
and historical figure hashes remain dated records of earlier versions.

## Reproduction without simulation

From the extracted archive root, using the README's pinned Python environment
and source `PYTHONPATH`:

```bash
python scripts/replot_saved_results.py --experiment 1 --data-root artifact_data --output-dir reproduced_figures_v42
python -m pytest -p no:cacheprovider -q tests/unit/test_exp1_plot_notation.py tests/unit/test_exp1_analysis.py tests/unit/test_exp1_theory.py tests/unit/test_artifact_tools.py
python -m ruff check . --no-cache
python scripts/verify_artifact.py
python scripts/verify_saved_statistics.py
```

Choose a fresh output directory; the replot command refuses to overwrite
existing figures unless `--force` is explicitly supplied. The updated PDFs
can directly replace the same filenames in the manuscript's `figures/`
directory. No change to `\includegraphics` is required.

Plotting-source edits change the current execution-source hash. Use saved
tables for replotting; do not bypass simulation version-mixing checks to
resume an old simulation with the revised source.
