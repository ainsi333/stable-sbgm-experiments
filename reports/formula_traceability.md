# Experiment 3 formula traceability

Historical implementation record. Version 40 does not display an Experiment 3
figure. This code is retained as a validation of the backward-family distinction,
not as an additional empirical claim in the manuscript. See `MATH_V40.md`.

Authoritative source: `mainaistats_no_A2 (2).tex`, SHA-256
`A2CEB81155566B43030677C90F5A62437146449118DACC17F0EC2B963DCE56BF`.
The physical SP-SDE, Popov-family, and EI formulas referenced below are
unchanged in this authority; legacy line numbers are retained only as historical
locators.

| Element | Manuscript | Implementation | Verification |
|---|---|---|---|
| Stable convention | lines 133-153 and 2427-2435: `E exp(iuZ)=exp(-|u|^alpha)` | `stable.py` CMS sampler | ECF, stability, symmetry, alpha=2 tests |
| Forward scales | lines 145-153 | `scores/discrete.py::forward_scales` and exact-reference kernel | analytic CF marginal tests |
| Backward drift | lines 167-183 | `integrators.py::popov_remainder` plus dissipative linear part | coefficient and stationary-invariance tests |
| Time orientation | `t=T-tau`, lines 169-183 | score at `T-kh`, checkpoint mapping after update | exact grid-index test |
| Fractional score | lines 203-210 and 2447-2457 | weighted three-atom posterior, then Tweedie identity | direct SciPy-weight comparison |
| EI drift | lines 2471-2486 | `stable_ei_coefficients` | exact factor test |
| EI noise | lines 2492-2511: `sigma^alpha=1-exp(-eta beta h)` | same function, implemented with `expm1` | coefficient and stationary-invariance tests |
| Density table | lines 2436-2445 | float64 positive log1p grid; explicit three-term alpha=1.5 asymptotic tail | independent Fourier points, convergence and seam tests |
| Stationary joint CF | consequence of stationary stable OU | `theory.py::stationary_joint_cf` | empirical/analytic smoke comparison |
| True reversed pair | distinction emphasized lines 162-187 | stationary and discrete analytic reversed-pair CFs | marginal-recovery tests |
| Invalid hybrid | defect documented lines 2408-2423 | drift eta 0.5, noise eta 1 with exact OU attenuation | automatic stationary failure test |
| Nested refinement coupling | protocol choice; it does not modify the SDE | shared finest-grid primitives with weights `q_f d_f^(m-1-j)` | alpha-power identity and pure-OU path equality tests |
| Refinement decision | frozen audit rule `0.25/sqrt(n)` | paired ECF change on the two finest levels, median by independent seed in every eta/checkpoint cell | pass/fail/missing-cell tests and publication-render guard |
| Stationary truth gate | frozen audit rules: component error `<=0.006`, joint separation `>=0.02`, `n>=200000` | analytic and empirical marginal/joint CF checks, including true-reverse distinction | inaccurate, nondiscriminating, and insufficient-budget mutations fail |
| Invalid-hybrid gate | simultaneous ECF family error `0.05`, rejection in at least `75%` of seeds | checkpoint/frequency-corrected Hoeffding radius followed by a robust seed rule | one-outlier and missing-checkpoint mutations fail |
| Pilot/final identity | same scientific protocol outside execution budgets | canonical compatibility payload/hash stored in diagnostics and aggregate manifest | pilot and final payload equality test |
| Aggregate provenance | every displayed metric must trace to raw tasks | atomic manifest hashes four aggregates plus every `arrays.npz` and `metadata.json` | diagnostic and raw-array corruption tests |

The three-atom target and the sweeps in `eta`, lag, and `N` are protocol choices,
not manuscript equations. They are frozen in TOML and included in the canonical
configuration hash.
