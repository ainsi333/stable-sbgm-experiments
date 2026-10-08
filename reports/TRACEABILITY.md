# Experiment-to-manuscript traceability

## Authority and scope

The current manuscript audited for inclusion is
`mainaistats (40).tex`, SHA-256
`4713d4441e62491322371b02c42d11673fc615b8c06d626560bfd36aac9ca88c`.
It is not redistributed in this code archive. The final simulations predate
that source and embed the execution snapshot
`mainaistats_no_A2 (2).tex`, SHA-256
`a2ceb81155566b43030677c90f5a62437146449118dacc17f0ec2b963dce56bf`.
The current source was checked separately against the formula reports in this
directory. This distinction is provenance, not a claim that the archived runs
were rerun after every prose edit.

## Figures used by the current manuscript

| Manuscript output and claim | Frozen numerical input | Figure-producing code | Frozen configuration | Historical final receipt | Verification status |
| --- | --- | --- | --- | --- | --- |
| `figures/experiment4_ct.pdf`: deterministic comparison of the truncated denoiser modulus `C_hat_t(40)` for stable and VP forward models on Student-t4, with `1/a(t)` | `artifact_data/exp4/ct_curve.csv` (`f055865a...7e80236`); diagnostic JSON/NPZ in the same directory | `experiments/exp4/run.py::render_figure`; archive-only entry point `scripts/replot_saved_results.py` | `configs/final/exp4.toml`; source tables are tied to Experiment 2 hash `eaaf84c...ef737` | run `exp4-8db0d318c774`; configuration `8db0d318...f3ce7`; source-code hash `a0576214...2a5b`; all numerical gates passed | Regenerated from the frozen curve and visually inspected. Deterministic numerical evaluation: no Monte Carlo bars. Not a closed-form supremum because the spatial domain is truncated. |
| `figures/experiment2_initialization_comparison_cropped.pdf`: initialization decay for stable and VP samplers in empirical `W_1.1` and `W_1.4`; dotted exact--exact sampling reference | `artifact_data/exp2/initialization_wp_summary.csv` (`711caf29...73cb6`) and seed table (`0f9913e2...fc60a`) | `src/levy_experiments/exp2_initialization.py::render_initialization_figure`; archive-only entry point `scripts/replot_saved_results.py` | simulation `configs/final/exp2.toml`; reanalysis `artifact_data/exp2/requested_analysis_config.toml` | run `exp2-eaaf84c93847`; configuration `eaaf84c...ef737`; source-code hash `e160f3e2...fe52`; 192/192 tasks and final gate passed; postprocess `exp2-initialization-fd5e654500a5` | Numerical tables are byte-identical to the final postprocess. All method markers are filled and the abandoned sampling-resolution legend is absent. Diagnostic columns are retained. Both displayed orders satisfy `1 < p < alpha=1.5`. |
| `figures/experiment1_quantile_ratio.pdf`: generated-to-target extreme-quantile ratios for Student targets, stable at `nu={1.2,1.5,3}` and VP at index match | `artifact_data/exp1/tail_summary.csv` (`0569fd40...acaf70`) | `experiments/exp1/plotting.py::plot_quantile_ratio`; called by `experiments/exp1/aggregate.py` and `scripts/replot_saved_results.py` | `configs/final/exp1.toml` | run `exp1-c3680150ac3c`; configuration `c3680150...e2802`; source-code hash `d4e23ab3...c0f79`; 132/132 tasks and final gate passed | New standalone export from the same frozen table; removes the accidental sliver from a previously cropped multipanel PDF. Values and uncertainty ranges are unchanged. |
| `figures/experiment1_main.pdf`: Hill index, extreme-quantile ratio, and normalized tail-constant diagnostics | `artifact_data/exp1/tail_summary.csv`; supporting per-seed decisions, slopes, mean-excess summaries, and NPZ arrays in `artifact_data/exp1/` | `experiments/exp1/plotting.py::plot_main`; called by `experiments/exp1/aggregate.py` and `scripts/replot_saved_results.py` | `configs/final/exp1.toml` | same Experiment 1 receipt as above | Rebuilt PDF is byte-identical to the historical final PDF (`44b8bcf2...b7cf55`). These diagnostics are finite-sample illustrations, not proofs of tail equivalence. |

Ellipses in the table abbreviate hashes for readability. Full hashes are in
`figures/figure_manifest.json`, the frozen configs, and `SHA256SUMS` at the
archive root.

## Implemented experiment not used by the current manuscript

Experiment 3 tests common marginals and eta-dependent joint laws. Its archived
inputs are `artifact_data/exp3/metrics_per_seed.csv`
(`2510a7c6...dccc9`), `metrics_summary.csv`, `diagnostics.json`, and
`aggregate_manifest.json`; its final config is `configs/final/exp3.toml`.
The historical run `exp3-e751bb4269e9` used configuration hash
`e751bb42...7498`, source-code hash `2fe0e87c...9ba10`, completed all 288 tasks,
and passed the declared gates. No Experiment 3 PDF is referenced by the current
manuscript, so none is placed in `figures/`.

## Reproduction commands

Replot the four manuscript figures, without simulation:

```bash
python scripts/replot_saved_results.py \
  --data-root artifact_data \
  --output-dir reproduced_figures
```

Verify frozen scientific-table hashes, included figure hashes, anonymity, and
the Experiment 2 marker convention:

```bash
python scripts/verify_artifact.py
python scripts/verify_saved_statistics.py
```

Publication-scale simulation commands are listed in `README.md`. They are
separate from replotting and retain the pilot gates. Re-running with the current
source creates a new code/provenance hash; it must not be silently merged with
the historical receipts above.

Version-40 source locations: denoiser figure line 613 and protocol 3111--3147;
initialization figure 659 and protocol 3151--3188; tail ratio 854 and protocol
3190--3336; three-panel appendix figure 3286. The displayed curves and intervals
were compared back to raw/seed-level sources in the version-40 audit. The
initialization score-sensitivity gate predates the displayed orders; see
`MATH_V40.md` and `MANUSCRIPT_ACTIONS.md` for the exact scope, not an assertion
that all order-specific controls passed. `artifact_data/source_inventory.json`
includes full hashes for the 612 historical tasks and 12 Student score tables.
