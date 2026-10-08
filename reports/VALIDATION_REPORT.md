# Final supplementary-code audit: manuscript version 40

Audit dates: 2026-09-30--2026-10-01. Authority:
`mainaistats (40).tex`, SHA-256
`4713d4441e62491322371b02c42d11673fc615b8c06d626560bfd36aac9ca88c`.

## Verdict and most important remaining issues

The numerical core and archived figure data agree with the current equations
and reported configurations. In particular the Student samplers use the
correct Popov eta=0.5 in BOTH drift and noise. The wrong-noise hybrid is only
a labelled negative control in Experiment 3. Experiment 4 is deterministic.

The code distribution has been repaired without changing final simulation
outputs, scientific configurations, displayed data, or the four PDF figures.
**This does not certify the entire submission ready to upload.** Authors must:

1. resolve the active AI Use Statement's narrower account versus the actual
   experimental-design assistance and the broader mathematical-assistance
   account commented out in the source; independently confirm human checks;
2. fix the tiny main-text figure insertion scales (estimated legends about
   2.3 pt for the half-column denoiser and 3.1 pt for the single-column
   two-panel initialization plot);
3. narrow the exp2 score-resolution-control claim: controls at p1/1.25
   do not certify p1.4;
4. inspect the exact compiled submission PDF and its page count/citations.

The exact proposed changes are in `MANUSCRIPT_ACTIONS.md`; the 18 checklist
items are audited individually in `CHECKLIST_V40.md`. The earlier warning
about visible editorial comments is withdrawn: version40 disables the macros.
No-GPU execution is not asserted to be an automatic AISTATS rejection ground.
The missing-license finding was resolved by the authorized MIT release on
2026-10-05. See `LICENSING_RELEASE.md` for the license scope and checks. This
administrative update does not resolve the scientific or presentation caveats above.

## Findings and disposition

| ID | Severity | Evidence | Action and status |
| --- | --- | --- | --- |
| R01 | Major | Old `exp4/source.py::validate_source` required the installed exp2 code hash to equal the historical run hash; metadata-only packaging already changed it | Corrected: accept exactly the frozen historical source or the current distribution, retaining fixed configuration, successful gate and four content-checked table hashes. Actual historical source successfully loaded. Unknown/mixed versions rejected by tests. |
| R02 | Major | Old `exp2_initialization.py::_validate_source_run` also rejected the frozen execution after packaging | Corrected for read-only reanalysis only: explicit historical (config,code) pair or current code; all task identities, raw hashes and configuration checks remain. Simulation resume still rejects mixed versions. |
| R03 | Major | Docker/Apptainer ran the full suite while omitting `slurm`, some `scripts`, and exp3's `experiments` | Corrected six definitions; include source, tests and required reports. Static regression coverage added. Linux/CUDA builds not executed on this Windows host. |
| R04 | Major | README `pip install --no-build-isolation -e .` failed because Hatchling editable mode requires unlisted `editables` | Corrected documented procedure to noneditable wheel install plus explicit source PYTHONPATH, as used in containers. Successful install and pip check. No extra unpinned installer dependency. |
| R05 | Major | Exp3 Slurm preflight used bound source, but container array/table/aggregate could use an installed image version | Corrected all three stages to bound source PYTHONPATH; propagate GPU visibility and CPU thread settings. Static tests, not an actual Slurm job. |
| R06 | Moderate | Authority and active reports still cited version38; exp2 historical notes said finals unexecuted | Updated authority/test to version40. Historical records labelled; current reports supersede earlier conclusions without changing historical receipts. |
| R07 | Moderate | Artifact checker covered only five numerical tables and output images, not all shipped code/configuration | Expanded to every SHA256SUMS entry, unsafe/missing/duplicate/unlisted file detection and figure-source hashes. Regression tests for corruption/path escape. |
| R08 | Moderate | Replot could write figures then fail on an existing manifest | Manifest now preflighted before any rendering; default output is separate reproduced_figures. Regression test confirms no partial writes in this case. |
| S01 | Major qualification | `resolved_analysis_spec.json` says post-hoc orders and score gate not recomputed; high-resolution exp2 particles are not stored | Disclosed; 288 displayed rows independently rechecked from raw, 144 temporal-refinement rows added; no invented p1.4 score gate or unauthorized rerun. |
| P01 | Major presentation | v40 lines613/659/853--854 scale wide PDFs into half/single columns | PDF geometry and reduced-size rendering inspected. Recommended float/layout changes only; attached manuscript not edited. |
| P02 | Author action | v40 lines903--905 and commented908--931 give different AI-use scopes | Explicit disclosure/verification confirmation required from authors; no attestation fabricated. |

## Numerical invariants and provenance

The prior artifact and all historical runs remain unchanged. These retained
plot inputs are byte-identical to the prior distribution and raw-run source:

| Input | SHA-256 |
| --- | --- |
| exp1 tail summary | `0569fd40f6a53271459887b7e2c44fefbb379e264903e2ab125538f929acaf70` |
| exp2 per-seed display | `0f9913e27d79d539398efc9300958755b1c4e7baf59f87fb21e25e24392fc60a` |
| exp2 display summary | `711caf2907c3e3914173dba752e22f79c4dbc8b05bd40b243161d21d8a873cb6` |
| exp3 per-seed metrics | `2510a7c6226d29b42321af881ad26062b38e4e3a343a18824b10c4f9142dccc9` |
| exp4 curve | `f055865a7e395bb1ff4c6fa3b43c70a5eea830743ac11e508f1f3b7027e80236` |

All 612 historical task directories were checked against completion markers,
configuration/code identities, raw NPZ checksums, metadata checksums and log
checksums: exp1 132, exp2 192, exp3 288. Exp2 additionally passed the complete
task loader's per-array content validation during read-only metric reanalysis.
The 8 exp1 and 4 exp2 main/high-resolution Student score tables were loaded
through their validating loaders. No corruption was detected.
Sanitized receipts and full hashes are in `artifact_data/source_inventory.json`.

The newly included T2 refinement table is an explicitly labelled reanalysis
of existing exp2 particles, not a new run or change to a displayed figure.
The source-code execution hashes remain the historical values; changing an
authority label does not relabel old observations as newly simulated.

## Statistical and graphical validation

- Recomputed all 288 displayed exp2 seed/order/horizon rows from original
  particle arrays at p1.1/1.4, matching the saved numerical values.
- Recomputed all 24 exp2 aggregate rows including 20,000-resample bootstrap
  intervals using the saved bootstrap seed.
- Independently recomputed all 94 tail-figure medians and 16%/84% ranges from
  the archived per-seed metrics. Primary sample sizes and distinct seed
  counts match the manuscript. Tail ranges are not confidence intervals.
- Regenerated all four PDFs into an external validation directory. All four
  are byte-identical to the distributed PDFs.
- Inspected all four rendered pages, their metadata and reduced-size versions.
  No PDF author field or JavaScript was found. No abandoned exp2 open-marker
  convention remains; dotted exact--exact references are retained.
- Native PDFs are readable; the version40 LaTeX insertion scales are NOT
  thereby certified readable. This is the separate P01 finding.

PDF SHA-256 values remain:

| Figure | SHA-256 |
| --- | --- |
| experiment1_main.pdf | `44b8bcf2218068d5cd33325934764490c17186fb629ffbd26581095104b7cf55` |
| experiment1_quantile_ratio.pdf | `ec6ed1425baa3882e7c5049900788100daeeac248eb43f89f37f7ddaf4708edf` |
| experiment2_initialization_comparison_cropped.pdf | `e60d162184481a74739ef8fa237463cd196818454ba86a2c4412656edccde425` |
| experiment4_ct.pdf | `52d2a23ac942f3e68a0e6e4e30c93127813c30392932869c758a7a619636ea4c` |

## Execution and verification boundaries

Validation host: Windows, Python3.12.10, CPU, no NVIDIA device. Pinned environment:
JAX/jaxlib0.7.2, NumPy2.3.3, SciPy1.16.2, Matplotlib3.10.6.

The full pre-install-repair suite passed 244 tests with 2 GPU skips.
The newly scoped historical exp2 source regression then passed in the
10-test postprocessing subset. Final release test/checksum results are
recorded in `RELEASE_CHECKS.md` after the installation repair.

A deliberately environment-clean invocation exposed an older installed wheel
and the failed editable install, rather than a numerical failure. This
motivated R04 and the test bootstrap which explicitly selects this checkout.
The successful documented wheel install reported no broken dependencies.
Matplotlib/PyParsing deprecation warnings are dependency API warnings, not
failed numerical gates. They were not hidden to obtain a clean result.

The source configurations of all four finals are unchanged. No pilot or final
publication-scale simulation was launched by this audit. Integration tests use
tiny smoke configurations. Raw particles (about3.5GiB) are not distributed;
full configs, seed summaries, code and commands are included. The source
manuscript and private raw logs are not included in the anonymous code ZIP.

CUDA/Apptainer/Slurm are inspected and statically tested only. Their target-site
execution, GPU precision parity, memory and performance still require a pilot.
The GPU dependency file does not freeze every transitive NVIDIA wheel.
There is no new GPU timing claim. Figure bitwise identity is only tested on
this CPU/font stack, not promised across platforms.

## Commands and interpretation

From the extracted root after the README installation and PYTHONPATH setting:

```bash
python -m pytest -p no:cacheprovider -q
python -m ruff check . --no-cache
python scripts/verify_artifact.py
python scripts/verify_saved_statistics.py
python scripts/replot_saved_results.py --data-root artifact_data --output-dir reproduced_figures
```

The main figure/code/config/run map is `TRACEABILITY.md`; formula and evidential
limits are in `MATH_V40.md`. Keep exact-versus-exact sampling references,
finite-epsilon/numerical-score caveats, and the manuscript's inconclusive
unmatched-tail diagnostics. No software gate proves a theorem.
