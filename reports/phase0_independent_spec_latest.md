# Phase 0 — independent mathematical specification

Status: frozen before inspection of the implementation for the present audit.

## Authority and scope

The mathematical authority used when the final simulations were executed was the
anonymized manuscript snapshot `mainaistats_no_A2 (2).tex`, SHA-256
`A2CEB81155566B43030677C90F5A62437146449118DACC17F0EC2B963DCE56BF`
(3,144 lines). Older manuscript sources and old numerical outputs are historical evidence only.

The submission artifact additionally records the current manuscript source hash in
`levy_experiments.authority`. The formulas and experimental parameters relevant to the
archived outputs were checked against that current source; the manuscript itself is not
distributed in this code archive.

The manuscript's published numerical protocol and the three new experiments have to be
distinguished. The former is described at lines 2898–3142 and is explicitly declared
provisional for every backward-trajectory quantity (lines 2904–2918). The latter are the
three target-specific protocols required by the author's subsequent specification. They
are not already validated by the manuscript and must not inherit numerical values from
the provisional figures.

## Common mathematical objects

### Stable convention and forward flow

For `1 < alpha < 2`, the isotropic stable driver is normalized by

`E exp(i <xi,L_t>) = exp(-t ||xi||_2^alpha)`.

Consequently, a standard one-dimensional symmetric stable variable `Z` has characteristic
function `exp(-|u|^alpha)`, and `L_h = h^(1/alpha) Z`. At `alpha=2`, this convention gives
`L_t^2 = sqrt(2) W_t`, not a standard Brownian motion. In more than one dimension the
symbol is `||xi||_2^alpha`; independent coordinate-wise stable draws would instead have
symbol `sum_j |xi_j|^alpha` and would be a different model.

For a positive schedule `beta`, with `B(t)=integral_0^t beta(s) ds`,

- `a_alpha(t) = exp(-B(t)/alpha)`,
- `gamma_alpha(t)^alpha = 1-exp(-B(t))`,
- `X_t =_law a_alpha(t) X_0 + gamma_alpha(t) Z`.

These definitions are fixed by manuscript lines 123–143.

### Fractional score and denoiser

With Fourier convention `F[f](xi)=integral f(x) exp(i<x,xi>) dx`, the fractional-gradient
multiplier is `-i xi ||xi||^(alpha-2)`. The marginal fractional score is

`S_t^(alpha)(x) = Delta^((alpha-2)/2) grad p_t(x) / p_t(x)`.

The denoiser `m_t(x)=E[X_0 | X_t=x]` obeys

`m_t(x) = [x + alpha gamma_alpha(t)^alpha S_t^(alpha)(x)]/a_alpha(t)`,

or equivalently

`S_t^(alpha)(x) = -(x-a_alpha(t)m_t(x))/(alpha gamma_alpha(t)^alpha)`.

These are marginal quantities, not conditional scores and not ordinary log-density scores
unless `alpha=2` under the corresponding Brownian convention.

### Matching-marginal backward family

Let backward time be `tau`, forward time `s=T-tau`, and
`bar_beta(tau)=beta(T-tau)`. For each `eta>0`, the additive process is

`dY_tau = [bar_beta(tau) Y_tau/alpha + (1+eta) bar_beta(tau) S_s^(alpha)(Y_tau)] d tau`
`          + [eta bar_beta(tau)]^(1/alpha) dL_tau^alpha`,

initialized from `Y_0 ~ p_T`. Under the stated well-posedness/Fokker–Planck uniqueness
conditions, only its one-time marginals are asserted to satisfy
`Law(Y_tau)=p_(T-tau)`. It is generally not the exact pathwise time reversal. Different
values of `eta` may have identical marginals and different joint path laws.

This matching-marginal statement is for the continuous exact-score process with exact
initialization `p_T`; it is not an exact property of a finite-step EI chain or of the
stationary-reference initialization. The additive member's jump compensator is
`eta*bar_beta(tau)*nu_alpha(dz) d tau`, whereas the genuine time reversal has a
state-dependent density-ratio tilt. Thus jump counts above a fixed radius provide an
additional exact path-law diagnostic even when all one-time marginals agree.

### Exponential integrator

For constant `beta0`, write

- `lambda = eta beta0 / alpha`,
- `F(x,s) = (1+eta) beta0 [x+alpha S_s^(alpha)(x)]/alpha`,
- `h=(T-epsilon)/N`, `s_k=T-kh`.

The announced update freezes `F` at `(x_k,s_k)` and integrates the dissipative linear
part and additive noise exactly:

`x_(k+1) = exp(-lambda h)x_k + (1-exp(-lambda h))/lambda F(x_k,s_k)`
`          + [1-exp(-eta beta0 h)]^(1/alpha) xi_k`,

where `xi_k ~ S alpha S(1)`. The continuous coefficient and the EI attenuator contain the
same `eta`. The Euler scale `[eta beta0 h]^(1/alpha)` is not the announced EI scale.

For the VP baseline,

- `a_2(t)=exp(-B(t)/2)`, `gamma_2(t)^2=1-exp(-B(t))`,
- `s_t(x)=-(x-a_2(t)m_t(x))/gamma_2(t)^2`,
- `lambda=beta0/2`, `F=beta0(x+s_t(x))`,
- EI noise scale `sqrt(1-exp(-beta0 h))` for a standard normal innovation.

The state after update `k` corresponds to forward time `s_(k+1)`; the last score is
evaluated at `epsilon+h`, and the terminal state is at `epsilon`.

### Distributional distances and admissible moments

For equal-size one-dimensional empirical samples, the empirical optimal-transport
estimator is

`W_p = [n^(-1) sum_i |x_(i)-y_(i)|^p]^(1/p)`

after sorting both samples in the same order. Stable comparisons may use only `p<alpha`
when the population moment condition is required. A synchronous mean distance is merely a
coupling cost and therefore an upper bound on `W_1`, not `W_1` itself.

## New experiment 3 — common marginals, different process laws

### Stationary analytic case

Target: `S alpha S(1)` in one dimension. Its fractional score is `-x/alpha`. For constant
`beta0`, the backward member reduces to the stationary stable OU process

`dY = -(eta beta0/alpha)Y d tau + (eta beta0)^(1/alpha)dL^alpha`.

For lag `Delta`, let `c_eta=exp(-eta beta0 Delta/alpha)`. Its exact transition is

`Y_(tau+Delta)=c_eta Y_tau + (1-c_eta^alpha)^(1/alpha) Z`,

and, in stationarity, the joint characteristic function is

`Phi_eta(u,v)=exp(-|u+c_eta v|^alpha-(1-c_eta^alpha)|v|^alpha)`.

Both marginals are exactly `exp(-|u|^alpha)` for every `eta`, whereas the joint CF varies
with `eta`. The true reversed pair of the stationary forward OU process has

`Phi_rev(u,v)=exp(-|v+c_1 u|^alpha-(1-c_1^alpha)|u|^alpha)`,

with `c_1=exp(-beta0 Delta/alpha)`. Equality of the additive member's one-time marginals
must not be reported as equality to this reversed joint law.

Outputs: direct exact-transition samples; marginal ECF errors; joint ECF errors against
`Phi_eta`; pairwise joint-law differences across `eta`; comparison with `Phi_rev` where
specified. No artificial time discretization is permitted here.

Scientific claim: the Popov family shares marginal flow while `eta` changes process law.
This is a numerical validation/diagnostic of formulas, not proof of the proposition.

### Nonstationary three-atom case

The protocol target is the probability measure with atoms `(-3, 0.5, 2)` and weights
`(1/4, 1/2, 1/4)`. For each forward time `s`,

`p_s(x)=sum_i w_i gamma(s)^(-1) q_alpha((x-a(s)x_i)/gamma(s))`,

`m_s(x)=sum_i x_i W_i(x,s)`, with normalized weights proportional to the summands, and
`S_s=-(x-a(s)m_s)/(alpha gamma(s)^alpha)`.

The backward process uses the EI above, starts either at exact `p_T` or at the reference as
explicitly labelled, and records only requested checkpoints. The exact marginal reference
at a checkpoint is sampled independently as `a(s)X_0+gamma(s)Z`.

Primary outputs: marginal ECF discrepancy to the exact forward marginal, empirical `W_1`
with an independent exact-reference floor, and a joint two-time characteristic statistic.
A bounded-kernel MMD is secondary. “Exact-versus-exact” means two independent samples of
the same analytic marginal and supplies the Monte-Carlo envelope; it is not zero at finite
sample size.

The deliberately invalid hybrid uses different `eta` values in drift and noise. It must be
labelled as a negative control, must fail marginal preservation, and must be impossible to
aggregate as a valid family member.

## New experiment 2 — direct initialization error in distribution

Target: standard Student `t_4` in one dimension. Stable model: `alpha=1.5`, `eta=0.5`,
`beta0=1`; VP is a separate Gaussian model. The stable and VP scores are distinct marginal
scores of their own forward corruptions and must be tabulated separately from the exact
target characteristic function. “Exact score” means the exact mathematical score evaluated
through a numerically validated table, not exact floating-point arithmetic and not a learned
network.

For each model, horizon and step count, two backward arms are required:

1. exact initialization `Y_0~p_T`, which isolates score-table plus discretization error;
2. stationary-reference initialization, which additionally contains initialization error.

At each requested forward checkpoint `s`, independent exact references are sampled from
`p_s`. The primary distance is `W_1.25` for the stable model (`1.25<1.5`); `W_1` and a
weighted characteristic-function distance are secondary. Independent exact-versus-exact
samples define a finite-sample floor. That floor is reported, not subtracted.

The horizon sweep addresses decay of initialization mismatch. The `h` refinement is only at
fixed `(epsilon,T)` and diagnoses discretization; it must not be described as a joint limit
in which `epsilon` decreases or `T` increases, and no unproved global rate in `h` may be
claimed. Fits are performed per independent seed and only when the signal exceeds the
predeclared Monte-Carlo/score-table resolution.

Scientific claim: quantify, rather than infer from a synchronous coupling, the distributional
cost of replacing `p_T` by the stationary reference, separately from discretization and
finite-sample error. A finite empirical distance does not establish a population `W_p` when
the relevant moment is absent.

## New experiment 1 — target-specific tail coverage

Targets are standard Student laws `t_nu` with `nu in {1.2,1.5,3}`. The stable model has
`alpha=1.5`, `eta=0.5`, `beta0=1`; the minimum VP baseline is target `nu=1.5`. Each target
must drive its own exact-mathematical, numerically tabulated score. Target absolute quantiles
are exact:

`q_beta(|T_nu|)=t_nu^(-1)((1+beta)/2)`.

The primary generated arm starts from the model's stationary reference because that is the
law covered by the fixed-time tail proposition. An exact-`p_T` arm is a labelled marginal
control, not a substitute for the theorem's initialization. The recommended finite-horizon
design fixes `T=2`, `epsilon=0.5`, `beta0=1`; using the old `T=8` design would move relevant
crossovers beyond finite-sample resolution for `nu<alpha`.

For a terminal sample, with descending absolute order statistics `R_(1)>=...>=R_(n)`:

- Hill estimates `xi_hat(k)=k^(-1) sum_(i<=k) log(R_(i)/R_(k+1))` and
  `alpha_hat=1/xi_hat`; thresholds/fractions are prespecified.
- Pickands and GPD/POT estimates are sensitivity diagnostics, never silently substituted for
  the primary estimator.
- The exceedance count and threshold accompany every mean-excess estimate
  `e_hat(u)/u = sum_i (R_i-u)_+ / [u sum_i 1{R_i>u}]`.
- The tail-survival constant at fixed exponent is estimated as
  `c_hat=(k/n) R_(k+1)^alpha`; it is distinct from the tail index.
- The extreme-mass ratio is
  `M_nu(beta)=q_hat_beta(|Y|)/q_beta(|T_nu|)`.

For the stable output, theory predicts tail index `alpha`, relative mean excess
`1/(alpha-1)=2`, and

`M_nu(beta) ~ A_nu (1-beta)^(1/nu-1/alpha)`.

Thus the log-log slopes are `+1/6` for `nu=1.2`, `0` for `nu=1.5`, and `-1/3` for
`nu=3`. At index match, a slope near zero is insufficient: the nonzero finite plateau and
tail constant must also be assessed. The VP output at `nu=1.5` is predicted to have
vanishing relative excess and `M->0`, but a finite sample cannot prove a sub-Gaussian tail.

Uncertainty is aggregated at the independent-seed level. The pooled mean-excess estimate is
descriptive under infinite variance; an ordinary bootstrap confidence interval for that
mean is invalid. Threshold sensitivity, effective exceedance counts, seed-level bootstrap
for finite-variance summaries, independent pilot/final seed namespaces, Pareto and exact
stable positive controls, and a light-tailed negative control are required. A missing
plateau or insufficient exceedances yields “inconclusive,” never a forced compatible result.

Scientific claim: finite-sample, finite-step validation of the target-specific consequences
of the fixed-time tail theorem. It cannot prove regular variation, exact tail equivalence,
or universal superiority of stable sampling.

The authority proves these tail statements for the ideal continuous exact-score sampler
initialized from the reference. Its introductory wording about tails of the numerical law
must not be used to silently extend the theorem to finite-step EI; the discrete-chain tail
constant is a separate, explicitly derived numerical reference.

## Pre-implementation logical checks

The following checks are consequences of this specification and are fixed before code audit:

1. Replacing EI stable noise by `(eta beta h)^(1/alpha)` must break exact stationarity at
   finite `h`; omitting `eta` must break it more strongly whenever `eta != 1`.
2. Using `beta(tau)` instead of `beta(T-tau)` must be detected on a nonconstant schedule.
3. In the stationary analytic case, all marginal CFs must be eta-invariant while at least
   one nonsymmetric joint-CF point must vary with eta.
4. Reusing one Student score for two values of `nu` must fail an independent marginal-CF or
   exact-reference test.
5. Replacing stable isotropic draws by independent coordinates must fail rotation
   invariance in dimension at least two.
6. Any `W_p` request with `p>=alpha` in the stable population comparison must be rejected.
7. Clipping, boundary pinning, or silent removal of non-finite tail samples is forbidden.
8. Permuting a sample must leave all distributional and tail estimators unchanged.
9. Changing a seed must change the stream while exact repetition of a seed must reproduce
   the artifact bit-for-bit within one fixed backend/precision contract.
10. A hybrid drift/noise eta mismatch must be caught by a test and by artifact metadata.

## Claims explicitly not supplied by the authority

- Matching one-time marginals is not equality of process laws or exact time reversal.
- A coupling gap is not an exact Wasserstein distance.
- A finite empirical `W_p` is not evidence that the population moment exists.
- A tail-index estimate does not identify the tail constant or exact extreme-mass coverage.
- Finite samples cannot prove an asymptotic equivalence.
- The three new protocols are not the old figures already reported at manuscript lines
  592–648; their publication-scale numerical outcomes remain to be generated and verified.
