# Experiment 4: forward denoiser modulus

## Status

The deterministic final run is complete and all predeclared numerical gates
pass. It is stored outside the source tree at:

```text
../run_exp4_publication_20260829/exp4-8db0d318c774
```

Configuration hash:
`8db0d318c77493b0453050e2e43d9d4871772e096b70ac5e61a17ffa434f3ce7`.
Code hash:
`a0576214ee0573e5355a793af607fa31ce70e8415f54676d8152890a0ade2a5b`.

## Quantity and formulas

For the one-dimensional Student-t(4) target, the experiment evaluates the
truncated forward denoiser modulus

```text
C_hat_t(40) = max_{|x|<=40} |m_t'(x)|.
```

With `beta=1`, the stable attenuation is
`a_alpha(t)=exp(-t/alpha)` at `alpha=1.5`, while the Brownian VP attenuation is
`a_VP(t)=exp(-t/2)`. The dashed comparison is therefore
`1/a_VP(t)=exp(t/2)`. It is the Brownian limiting spatial slope and a lower
bound on the population VP modulus, not an equality forced on the numerical
curve. The backward parameter `eta` is absent because `C_t` is defined from
the forward denoiser.

The exact forward characteristic functions from Experiment 2 are reused. The
density, score numerator, and their spatial derivatives are evaluated with
their exact Fourier multipliers before the quotient derivative is formed.
This avoids finite-differencing the score tables. The four immutable
Experiment 2 main/high-resolution tables are nevertheless hash-validated and
used as independent overlap checks on their common interval `[0.05,2]`.

## Validation results

- Time grid: 64 points on `[0.05,4.5]`.
- Stable FFT: half-width 8192, `N=1,048,576`.
- VP FFT: half-width 1024, `N=524,288`.
- Refinement times: `0.05, 0.25, 0.5, 1, 2, 3, 4.5`.
- Nested spatial radii: `8, 16, 24, 40`.
- Maximum main/refined relative discrepancy: `1.33094e-6`.
- Maximum discrepancy against the cached high-resolution table proxy: `0.0237726`.
- Every numerical maximizer lies inside `|x|<=8`.
- At `t=4.5`, the stable boundary derivative is `2.14913e-5`, or
  `2.96688e-4` of the stable maximum.
- At `t=4.5`, the VP derivative at `x=40` differs from `exp(t/2)` by
  `0.00312475` relatively.
- Stable displayed endpoints: `0.967202` and `0.0724373`.
- VP displayed endpoints: `1.03352` and `40.3070`.

## Why the figure has no error bars

There is no Monte Carlo layer in this experiment: no particles, no random
seeds, no fitted score, and no bootstrap. At fixed configuration, the curve
and figure are deterministic, and the test suite verifies byte-identical PDF
and PNG rendering. Statistical error bars would therefore be artificial.

This does **not** make the displayed values closed-form exact values of the
population supremum. The exact ingredients are the Student-t(4) characteristic
function, the forward convolution identities, the Fourier multipliers, and the
Tweedie formulas. Their numerical evaluation still has deterministic error
from the Fourier cutoff and periodic domain, the spatial mesh, `float64`
roundoff, and the replacement

```text
sup over R  ->  maximum over the grid |x| <= 40.
```

Those effects are checked by doubling the FFT size, comparing nested domains,
checking the boundary asymptotics, and using the cached high-resolution score
tables as a secondary overlap diagnostic. The observed main/refined difference
(`1.33094e-6` relative) is a sensitivity diagnostic, not a confidence interval
or a rigorous bound on every numerical error. The `2.37726%` table-overlap
difference compares two numerical routes and must not be plotted as an error
bar either. At the scale of the main figure, the direct refinement difference
would be visually smaller than the line width.

The precise wording is therefore: **deterministic numerical evaluation of
`C_hat_t(40)` from the exact forward characteristic function, converged at the
tested resolutions**. It should not be described simply as “exact computation
of `C_t`.”

If a numerical-precision visualization is requested, the seven pointwise
main/refined discrepancies may be shown in a supplementary diagnostic panel
labelled “FFT refinement discrepancy.” They must not be labelled error bars or
confidence intervals; the maximum is below the visible line width in the main
panel.

Measured on the local 16-logical-CPU Windows machine, the final deterministic
calculation took 95.366 seconds: 72.175 seconds for the 64-point curves and
23.191 seconds for the doubled-resolution checks. This workload is CPU/FFT
post-processing; it should not be submitted as a GPU job.

## Commands

From the project root with the environment activated:

```bash
make smoke-exp4 OUTPUT_ROOT=audit_outputs/exp4_smoke
make final-exp4 OUTPUT_ROOT=../run_exp4_final
```

Equivalent explicit final command:

```bash
python -m experiments.exp4.run \
  --config configs/final/exp4.toml \
  --device cpu --precision float64 \
  --output-dir ../run_exp4_final --resume
```

If the verified Experiment 2 run has been copied elsewhere, provide it without
changing the scientific configuration hash:

```bash
python -m experiments.exp4.run \
  --config configs/final/exp4.toml \
  --source-run /absolute/path/to/exp2-eaaf84c93847 \
  --device cpu --precision float64 \
  --output-dir /absolute/path/to/new-output --resume
```

The PDF/PNG figure is rebuilt from the deterministic curve CSV/NPZ during the
same command. A repeated `--resume` verifies every artifact hash and refuses a
run produced by a different code hash.

## Interpretation boundary

The observed separation is a numerical illustration of the forward-denoiser
dichotomy for Student-t(4) at the declared parameters. It is not a proof of
the asymptotic theorem, a comparison of backward path laws, or evidence of a
universal mixing advantage.
