# Mathematical consistency with manuscript version 40

Authority: `mainaistats (40).tex`, with SHA-256 recorded in `authority.py`.
This is an equation/protocol/code audit, not certification of all proofs.

## Main result

The retained final Student samplers use the current Popov convention:
drift `beta*x/alpha+(1+eta)*beta*S` and noise `(eta*beta)^(1/alpha)`.
The EI decay is `exp(-eta*beta*h/alpha)` and innovation scale is
`(1-exp(-eta*beta*h))^(1/alpha)`. Both use eta=0.5. The old drift/noise
mismatch is not present in these archived finals. The hybrid in Experiment 3
is deliberately invalid and labelled as a negative control. Experiment 4
contains no backward propagation or stochastic noise.

## Formula-to-code correspondence

| Manuscript source and element | Formula / implementation | Verdict |
| --- | --- | --- |
| Driver, line181; appendix3338 | CF exp(-t abs(u)^alpha); `stable.py` CMS and sqrt(2) normal at alpha2 | Consistent. alpha1 explicitly unsupported. All reported runs are 1D, not an implementation claim for multivariate isotropic noise. |
| Forward, around235;3055--3064 | a=exp(-beta*t/alpha), gamma^alpha=1-exp(-beta*t); `exp2/simulation.py::forward_coefficients` and both Student theory modules | Same convention for exact reference draws and forward CF. |
| VP forward | a=exp(-beta*t/2), Gaussian variance=1-exp(-beta*t) | Unit-variance stationary Gaussian; intentionally different from variance-two stable alpha2 normalization. |
| Popov,268--296 | b=beta*x/alpha+(1+eta)*beta*S at t=T-tau; noise=(eta*beta)^(1/alpha) | `integrators.py::popov_remainder` and both Student simulation modules use the correct sign and factors. Only beta=1 is tested by final experiments. |
| EI,3032--3053 | lambda=eta*beta/alpha, F=(1+eta)*beta*(x+alpha*S)/alpha; decay exp(-lambda*h), remainder multiplier (1-exp(-lambda*h))/lambda, noise power1-exp(-eta*beta*h) | `stable_ei_coefficients`: -lambda*x+F recovers the backward drift; exact linear stochastic convolution, stable 1/alpha root, `expm1` for small steps. |
| VP EI,3051--3053 | lambda=beta/2; F=beta*(x+s); noise variance1-exp(-beta*h) | `exp2/simulation.py::vp_ei_coefficients`, also used by exp1, consistent. |
| Grid,3033 | t_k=T-kh, h=(T-epsilon)/N | Both Student kernels evaluate the score before the update; post-update checkpoints. N steps, stop at epsilon, not zero. Coupled refinement uses weighted fine-grid primitives. |
| Scores,3055--3074 | Target/model-specific forward CF, deterministic inversion, interpolation of the denoiser residual | Both score-table modules use separate tables; exact CF does not make numerical/interpolated scores exact. Independent off-grid and doubled-resolution checks assess approximation. |
| Fourier signs,3115--3127 | transform exp(+ixu); derivative -iu; stable score numerator -iu abs(u)^(alpha-2) | `exp4/computation.py::spectral_denoiser_profile` and table builders agree. |
| Denoiser,3128--3140 | m'_stable=[1+alpha*(1-exp(-t))*S']/a; m'_VP=[1+(1-exp(-t))*s']/a | `model_scales` and Fourier quotient differentiation agree; maximum is truncated/discretized. |
| Initialization,3151--3188 | independent stationary and exact-pT EI populations; I_p=Wp(EI_stat,EI_pT); F_p=Wp(exact_eps_A,exact_eps_B) | Simulation and `_task_metric_rows` agree. Same table/step size but independent innovations. I_p is a finite-step proxy; F_p a sampling reference, not a lower bound. |
| Wp,3164--3172 | (mean abs(sort(X)-sort(Y))^p)^(1/p) | Exact equal-weight 1D empirical transport, not exact population distance. Displayed p1.1/1.4 both below alpha1.5. |
| Hill,3204--3222 | reciprocal mean log(R_j/R_(k+1)); k=floor(n*fraction) | `exp1/metrics.py::hill_grid`, `tail_count`; correct order statistic and rounding. Primary k1572; finite threshold bias remains. |
| Student quantiles,3224--3243 | radial quantile Student.isf(s/2) | `student_abs_quantiles`, exact target CDF denominator, no target sampling noise. At s=.001 about262 upper observations/primary seed. |
| Tail slope,3245--3252 | slope log(M) against log(s) =1/nu-1/alpha | Per-seed fits; x-axis decreases rightwards. Index matching does not imply ratio1. |
| Tail constant,3254--3282 | c_hat=(k/n)R_(k+1)^alpha; finite-chain recurrence abs(r_k)^alpha*c_k+C_alpha*(1-exp(-eta*h)) | `tail_constant_at_k` and `exp1/theory.py`; denominator is the EI-chain constant, not target or continuous-time constant. |
| Uncertainty,3184--3186;3290--3301 | exp2 median/90% seed-bootstrap; exp1 median/16--84% ranges | All displayed medians/bands recomputed; exp2 bootstrap also reproduced. No pseudoreplication from seed pooling. |

Clamps inside interpolation protect array indices and blend weights. Separate
tail branches are selected outside the space grid; no particle is clipped.
Nonfinite trajectories fail the run. Independent-coordinate sampling in higher
dimension would not be isotropic for alpha<2; no such simulation is claimed here.

## Experiment 3, additional implementation control

No exp3 figure is referenced in version40. Its exact stationary transition is
Y_lag=rY0+(1-r^alpha)^(1/alpha)Z with r=exp(-eta*beta*lag/alpha).
The joint CF is exp(-abs(u+r*v)^alpha-(1-r^alpha)*abs(v)^alpha).
The true reversed forward pair instead uses
exp(-abs(r_f*u+v)^alpha-(1-r_f^alpha)*abs(u)^alpha),
r_f=exp(-beta*lag/alpha). `theory.py` implements both. Common marginals do not
imply identical path laws. The finite-mixture posterior formula is analytic,
but its stable density values use validated interpolation, not an exact
floating-point oracle. The hybrid eta_drift=.5/eta_noise=1 is intentionally
invalid; tests and the archived gate reject it as a valid-family approximation.

Final configuration: alpha1.5, beta1, eta{.25,.5,1,2},12 seeds, lags{.25,.5,1};
atoms{-3,.5,2}, weights{.25,.5,.25}, T3, epsilon.25, N1760/3520. This validates
the family distinction and integrator, not a new displayed manuscript claim.

## Evidential strength and remaining limitation

| Figure | Verdict | Permitted conclusion |
| --- | --- | --- |
| Denoiser modulus | Valid with numerical qualification | Deterministic 64-time, radius40 calculation; resolution/domain checks, no Monte Carlo bars; not certified global supremum. |
| Initialization | Valid with qualification | Empirical loss of initialization dependence on Student-t4 in both models; no rate fit or universal stable superiority. Score error at displayed p1.4 is not specifically certified. |
| Main tail ratio | Valid qualitative finite-sample illustration | Correct direction of target-specific mismatch; index match does not mean exact coverage. |
| Appendix tail diagnostics | Valid with inconclusive unmatched regimes | Existing v40 caution matches threshold drift; no inferred exact tail equivalence. |
| Extra exp3 control | Valid supplementary implementation check | Joint-law difference versus shared marginals, not exact reversal of paths. |

The original exp2 analysis used p1/1.25; p1.1/1.4 was selected in post-processing.
All288 displayed seed/order/horizon rows were recovered from immutable raw
particles. The144 T2 refinement rows were similarly recalculated and included
separately. However, `execute_task_batches` stores only scalar high-resolution
score distances at the original orders, not `numerical_hires`. W1.4 cannot be
reconstructed from W1 and W1.25. No new passed order-1.4 score gate is claimed.
The manuscript control sentence must be narrowed as in `MANUSCRIPT_ACTIONS.md`.

No global h-rate, joint h/epsilon/T limit, population moment existence from
empirical finiteness, or proof of regular variation is inferred here. Several
v40 theorem statements permit p=1; the chosen displayed orders do not make W1
mathematically invalid. No theoretical conclusion was changed to fit results.
