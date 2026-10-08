# Experiment 1 formula traceability

Historical implementation record. For the current version-40 manuscript,
use `MATH_V40.md` and `TRACEABILITY.md`; source-line references below identify
the earlier execution snapshot, not the current submission.

## Authority and scope

- Execution-time authoritative snapshot: `mainaistats_no_A2 (2).tex`
- SHA-256: `A2CEB81155566B43030677C90F5A62437146449118DACC17F0EC2B963DCE56BF`
- The supplement is included in that source (lines 756--2651). The separate old
  supplement is not used because it fixes an obsolete backward member.
- Old audit inputs: `levy (1).py` SHA-256
  `FCA9E677A3C8FCBD199C698BAC2B189EC109D5B20E4842E31D3ABDB9B38C70E1`;
  `run_tails (2).py` SHA-256
  `CA611A75010E90C32FB71F1AA0A90C00B9C3675B52D8C56F4A922314E42AA43C`.
  They are evidence only and are never imported.

## Mathematical map

| Element | Manuscript specification | Implementation | Verification |
|---|---|---|---|
| Target | Standard Student (t_\nu), (\nu\in\{1.2,1.5,3\}) | `theory.student_density`, `student_cf`, `student_abs_quantile` | normalization, CF quadrature, tail constant, and quantile unit tests |
| Stable forward | (a(t)=e^{-t/\alpha}), (\gamma(t)^\alpha=1-e^{-t}), (X_t=aX_0+\gamma Z_\alpha) | `theory.forward_coefficients`, `sample_exact_forward_marginal` | analytic-CF and sampler tests |
| VP forward | (a_2(t)=e^{-t/2}), (\gamma_2(t)^2=1-e^{-t}) | same functions with `model="vp"` | model-specific tests |
| Stable score | (S_t=\mathcal F^{-1}[-i\xi|\xi|^{\alpha-2}\phi_t]/p_t) | `score_tables._spectral_values` | direct oscillatory quadrature in the bulk and embedded (2\times) FFT over the complete blend |
| VP score | (s_t=\mathcal F^{-1}[-i\xi\phi_t]/p_t) | separate VP multiplier and table | same independent checks; table hash differs from all stable tables |
| Stable tail score | leading Student-plus-stable density denominator and fractional numerator | `theory.stable_tail_score`; JAX equivalent in `interpolate_score_jax` | worst configured regime (\nu=1.2) checked against an embedded doubled-domain FFT |
| Stable backward | (dY=[Y/\alpha+(1+\eta)S_s(Y)]d\tau+(\eta)^{1/\alpha}dL_\tau^\alpha), (s=T-\tau) | `simulation._update_state`, `integrators.popov_remainder` | coefficient tests and true integration smoke |
| Stable EI | (e^{-\eta h/\alpha}Y_k+\frac{1-e^{-\eta h/\alpha}}{\eta/\alpha}F(Y_k,S_{s_k})+(1-e^{-\eta h})^{1/\alpha}\xi_k) | `stable_ei_coefficients`; nested exact OU innovation aggregation | unit test fixes every factor in (\alpha,\eta,\beta,h) |
| VP backward/EI | drift (Y/2+s_s(Y)), decay (e^{-h/2}), noise (\sqrt{1-e^{-h}}) | `vp_ei_coefficients`, model branch in simulation | unit and integration tests |
| Initialization | primary stable (S\alpha S(1)), VP (N(0,1)); diagnostic exact (p_T) | separate named random streams | exact control arrays A/B/C and stream-manifest tests |
| Stable generator | (E[e^{iuZ}]=e^{-|u|^\alpha}); (\alpha=2\) gives (\sqrt2N(0,1)) | backend-native CMS in `levy_experiments.stable` | shared ECF, scaling, stability, symmetry, and Brownian-limit tests |
| Tail constant | (c_{N,h}=C_{1,\alpha}\rho_{N,h}^\alpha) with the EI recurrence | `theory.discrete_stable_scale_power` and `discrete_stable_tail_constant` | recurrence and scale-equivariance tests; stored per task |
| Continuous comparison | (c=C_{1,\alpha}\rho^\alpha) from the manuscript ODE | `continuous_stable_scale_power` | high-accuracy DOP853 reference, secondary only |
| Quantile ratio | (\mathcal M_\nu(\beta)=q_\beta(|Y|)/q_\beta(|T_\nu|)) | exact Student denominator and per-seed log-log fit | synthetic-sign and exact-quantile tests |
| Marginal check | (W_1) and ECF, numerical exact-(p_T) arm against exact (p_s) | median versus A/B/C; maximum pairwise exact-exact envelope | three-reference loader and integration test |

## Time and noise conventions

For every dynamic task, `h=(T-epsilon)/N`, `s_k=T-kh`, and the score is
evaluated before update at `s_k`. Checkpoint index `j` represents the post-update
state at `s=T-jh`. Arrays are reversed only when written so
`control_forward_times` is increasing; no trajectory orientation is changed.

Refinement levels are coupled through `coupling_steps=max(N)`. A coarse noise
is the exact OU-weighted aggregation of the same fine standardized innovations,
not a fresh draw. Stable aggregation uses the stability property and VP uses
Gaussian variance addition. Initializations and exact references omit `N` from
their semantic stream labels.

## Why the former tail outputs are superseded

The old code fixes the drift member at `eta=alpha-1` but injects noise with
power `beta*h`, which is the `eta=1` Euler noise. At `alpha=1.5`, `h=0.075`, the
old amplitude is approximately `1.607` times the correct EI amplitude and its
stable power is approximately `2.038` times too large. Experiment 1 never reads
those outputs. It also replaces the old one-generator/three-external-target
quantile plot by three target-specific generators and exact Student quantiles.

## Interpretation boundary

The primary reference is already alpha-heavy. The experiment validates the
complete exact-score-tabulated finite-EI sampler and its predicted output-tail
diagnostics; it does not establish that jumps create a tail from a light law.
Finite samples cannot prove regular variation, tail equivalence, or a theorem.
The VP baseline at `nu=1.5` is a minimal index-matched baseline and does not
support a universal stable-over-VP claim. Only (W_1) is used, so all displayed
Wasserstein quantities remain in a common admissible moment domain.
