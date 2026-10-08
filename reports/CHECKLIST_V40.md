# Checklist and AI-disclosure audit: manuscript version 40

Audit dates: 2026-09-30--2026-10-01. Scope: the source named `mainaistats (40).tex`,
the preserved numerical artifact, its configuration files, saved numerical
tables, and implementation. The manuscript was not edited. No publication-scale
simulation or training was run; integration tests use tiny smoke cases. Line numbers refer
to version 40, not to the older manuscript audited during initial packaging.

## Outcome

The current checklist is substantially supported by the artifact. Its honest
**No** answer on complete complexity analysis should remain **No**. The
2026-10-05 authorized licensing revision supplies an MIT license and supports
changing item 4(b) to **Yes**; see `LICENSING_RELEASE.md` for replacement text.
This is an artifact update, not an edit already made to the manuscript. The CPU-only
execution claim is supported and does not require a GPU validation. The main
remaining disclosure issue is the AI Use Statement: it should explicitly cover
the substantive mathematical and experimental-design/methodology assistance
documented during preparation, and its assertions about independent human
verification require the authors' own confirmation.

## Question-by-question evidence

“Keep” means that the answer is supported at the level audited here. In
particular, this checklist audit is not an independent certification of every
proof or an attestation about what a human author personally verified.

| Item | Current answer | Verdict and evidence |
| --- | --- | --- |
| 1(a): mathematical setting, assumptions, algorithm/model | Yes | Keep. The setup, backward family, exponential-integrator scheme and assumptions are stated in the main text; the numerical protocol is at lines 3020–3364. |
| 1(b): properties and time/space/sample complexity | No | Keep. The convergence analysis and the propagation cost `O(nN)` and sorted one-dimensional Wasserstein cost `O(n log n)` are present (lines 3343–3346); sample counts appear in the protocol table. A complete space/precomputation-complexity analysis is not supplied, exactly as the answer acknowledges. |
| 1(c): optional anonymized source/dependencies | Yes | Keep for the submitted artifact. Simulation, post-processing, plotting and tests are included, with `pyproject.toml`, pinned CPU dependencies, and README commands. The unvalidated optional CUDA path does not negate the disclosed CPU result. |
| 2(a): full assumptions for theoretical results | Yes | Structurally supported; retain subject to the mathematical audit. The statements specify or refer to assumptions, moment restrictions and tail-index regimes. Code consistency alone cannot certify that no mathematical hypothesis is missing. |
| 2(b): complete proofs | Yes | Structurally supported; retain subject to the mathematical audit. The cited appendices contain proofs and the backward-construction discussion. This answer is not established merely by automated tests or the existence of an appendix. |
| 2(c): explanations of assumptions | Yes | Keep. The text explains the target, score and denoiser conditions and connects them to regularity, contraction, moments and tails. |
| 3(a): code/data/instructions for the main empirical results | Yes | Keep with the README's existing data boundary. All four figures have saved source tables and replotting commands; full configurations and simulation commands are included. Raw terminal particles are excluded (about 3.5 GiB), not falsely described as distributed. Replotting saved results and rerunning the synthetic simulations are separate workflows. |
| 3(b): training details | Not Applicable | Keep. No model is trained, no external dataset is split, and scores are computed from forward characteristic functions by numerical Fourier inversion/interpolation. Numerical settings remain disclosed even though training is inapplicable. |
| 3(c): estimators/statistics/error bars | Yes | Keep. Experiment 2 uses 12-seed medians and pointwise 90% percentile-bootstrap intervals; tail figures use medians and descriptive 16%–84% inter-seed ranges. The deterministic denoiser calculation has no sampling error bars. See the checks below. |
| 3(d): computing infrastructure | Yes | Keep. Lines 3348–3355 report CPU `float64`, Windows 11, Python/JAX/NumPy/SciPy versions, and 16 logical CPU cores, consistent with the archived execution report. No GPU result is claimed. An exact processor model and RAM amount would improve detail but are not asserted here or required to change this answer. |
| 4(a): citations for creators of existing assets | Yes | Supported for the named core numerical libraries: the manuscript cites JAX, NumPy and SciPy; `THIRD_PARTY.md` identifies all direct dependencies and their upstream projects. Matplotlib is also used for figures and documented in the artifact; adding its scholarly citation in the paper would make attribution more complete. No external dataset or pretrained model is used. |
| 4(b): asset-license information | No in v40; change to Yes with this revision | The authorized 2026-10-05 revision adds `LICENSE` (MIT) for the original supplementary implementation and accompanying materials, and declares MIT in `pyproject.toml`. JAX/NumPy/SciPy project licenses are identified in the paper; direct dependencies and upstream terms are documented in `THIRD_PARTY.md`. No third-party source or binary package is redistributed. Dependency licenses remain separate. See `LICENSING_RELEASE.md` for exact replacement text. |
| 4(c): released new assets in supplement/URL | Yes | Keep when this archive is actually uploaded. The code, configurations, saved tables and four manuscript figures are present. |
| 4(d): provider/curator consent | Not Applicable | Keep. The experiments generate synthetic numerical samples rather than using provider-supplied personal data. |
| 4(e): sensitive/personal/offensive content | Not Applicable | Keep for the scientific data, which are synthetic numerical samples. Archive anonymity remains a separate packaging obligation. Preserve the template's original question wording, including its word “sensible.” |
| 5(a): participant instructions/screenshots | Not Applicable | Keep. No crowdsourcing or human-subject experiment is described or used. |
| 5(b): participant risks/IRB | Not Applicable | Keep, for the same reason. |
| 5(c): participant wages/compensation | Not Applicable | Keep, for the same reason. |

## Empirical cross-checks

- **Initialization:** `configs/final/exp2.toml` specifies 12 seeds
  (`1000`–`1011`), 16,384 particles, epsilon 0.05, horizons
  `0.1, 0.15, 0.25, 0.5, 1, 2`, and step counts
  `4, 8, 16, 36, 76, 156`, giving `h=0.0125`. The distributed per-seed table
  contains 288 rows: 2 models × 6 horizons × 12 seeds × 2 orders, all with
  sample count 16,384. The displayed orders 1.1 and 1.4 come from
  `configs/postprocess/exp2_initialization.toml`; the initial simulation
  aggregation orders 1 and 1.25 are not the displayed post-processing orders.
  This is not a contradiction.
- **Initialization uncertainty:** the post-processing specification records
  20,000 bootstrap resamples, confidence level 0.90 and seed 20260828.
  `src/levy_experiments/exp2_initialization.py` resamples seed labels jointly
  across horizons and takes pointwise percentile intervals of bootstrap
  medians. The dotted exact–exact distances use independent exact samples and
  are sampling-resolution references, not lower bounds or quantities to
  subtract.
- **Tail figures:** `configs/final/exp1.toml` confirms 12 primary seeds,
  262,144 primary particles, stable alpha 1.5/eta 0.5, targets with nu
  1.2/1.5/3, a VP baseline only at nu 1.5, `T=2`, epsilon 0.5 and `N=80`.
  Displayed Hill, quantile-ratio (`mass_ratio`) and tail-constant rows have
  12 seeds and sample size 262,144. The analysis and plotting code use the
  median and 0.16/0.84 empirical seed quantiles; these bands are not confidence
  intervals. The 10,000 bootstrap resamples mentioned in the tail-protocol
  paragraph belong to the separate tail diagnostics, not Experiment 2.
- **Denoiser modulus:** `configs/final/exp4.toml` confirms 64 deterministic
  times in `[0.05,4.5]`, spatial maximum 40, FFT sizes 2^20/2^19 for stable/VP,
  and seven doubled-resolution checks. The saved diagnostic gates pass.
  This remains a finite-domain, discretized maximum, not a certified global
  population supremum; absence of statistical error bars is appropriate.
- **No new scientific claims:** the manuscript's cautious interpretation of
  unresolved mismatched-tail diagnostics should be retained. Numerical gates
  and qualitative agreement do not prove the asymptotic theorems.

For provenance clarity, the authors may also mention that the displayed
Experiment 2 orders were evaluated in post-processing: the resolved analysis
file explicitly records `post_hoc: true` and says its score-sensitivity gate
was not recomputed. The manuscript does not currently claim these orders were
pre-registered, and no checklist-answer change is needed. Do not convert the
existing numerical controls into a new claim of a passed order-specific gate
at 1.1 and 1.4.

## AI statement: necessary substantive correction and author confirmation

The section is already present before the references (lines 901–907), which
is the correct location. Its active wording describes English editing,
literature checks, “mathematical proofreading,” and numerical implementation
and verification. That scope is narrower than the mathematical-statement,
proof-development, interpretation, and experiment-design/methodology
assistance documented during manuscript preparation. The older commented
statement at lines 908–931 also explicitly describes substantive mathematical
assistance; comments are not a public disclosure in the compiled PDF.

The broad denial that AI was used to “determine the scientific conclusions”
should not substitute for disclosure of assistance with interpreting results.
Likewise, the assertion that *all* AI-assisted arguments were independently
checked by the authors cannot be established by an AI audit or a passing code
test suite. These are author-confirmation issues, not facts certified here.

Proposed exact English replacement for the **scope paragraph only**, to be
reviewed against the authors' complete usage history:

> In this work, we used OpenAI's ChatGPT and Codex to assist with formulating
> and revising mathematical statements and assumptions, exploring proof
> strategies and intermediate estimates, drafting and revising proofs, and
> critically examining mathematical arguments. We also used these tools for
> feedback on experimental design and methodology, implementing numerical
> methods and simulation code, checking numerical diagnostics, and discussing
> the interpretation and scope of theoretical and numerical results.
> Additional assistance included literature searches and checks, English
> writing and editing, manuscript organization, reference formatting, and
> scientific-figure code and presentation.

This text describes assistance, not transfer of scientific responsibility.
Confirm the complete tool list and add any other actual required-disclosure
uses before submission. Deterministic/probabilistic simulation code generated
the numerical observations; language-model responses are not experimental
measurements. If useful to avoid ambiguity about synthetic-data generation,
add this exact sentence:

> The experimental observations were generated by the probabilistic models,
> numerical simulations, and deterministic calculations described in the
> numerical protocol, using AI-assisted implementation; language-model
> responses were not used as experimental observations.

Keep the existing final responsibility paragraph. Finalize the verification
paragraph only after the authors confirm what they personally checked:
mathematical arguments, primary references, simulation code, diagnostics and
reported outputs. Retain “independently” or “all” only if those stronger
statements are accurate. Do not replace an unverified assurance with another
generic assurance, and do not submit author-confirmation placeholders.

## Official requirements versus recommendations

The official [AISTATS 2027 Call for Papers](https://virtual.aistats.org/Conferences/2027/CallForPapers)
and [Submission FAQ](https://virtual.aistats.org/Conferences/2027/SubmissionFAQ)
were checked on 2026-09-30 and rechecked on 2026-10-01. The official
[paper template](https://aistats.org/aistats2027/AISTATS2027PaperPack.zip)
was inspected from its distributed `sample_paper.tex`.

- **Required AI disclosure:** a statement before the references, also
  reflected in the submission form. Missing the statement causes desk
  rejection. Substantive proof/claim, methodology/experiment, implementation,
  and interpretation assistance must be disclosed. The authors remain
  responsible for the submission. [CFP](https://virtual.aistats.org/Conferences/2027/CallForPapers)
- **Recommended AI disclosure:** literature, editorial, figure and similar
  support should also be described. AISTATS permits AI assistance; disclosure
  is not an admission of a prohibited research practice.
  [FAQ](https://virtual.aistats.org/Conferences/2027/SubmissionFAQ)
- **Checklist format:** retain the questions; choose Yes, No or Not
  Applicable. Justifications are encouraged. The code item is explicitly
  optional. The template states that missing the initial checklist alone
  does not cause desk rejection, although later submission is requested.
  All 18 version-40 questions match the template.
  [Template](https://aistats.org/aistats2027/AISTATS2027PaperPack.zip)
- **Not stated as conference blockers:** an honest No answer, lack of a GPU
  run, and absence of a standalone implementation license are not identified
  as automatic rejection grounds in these rules. License selection remains
  an author distribution decision; this audit does not grant rights.
- **Submission packaging:** keep the supplement anonymous and consistent
  with the paper; upload it by the full-paper deadline. The appendix,
  checklist and AI statement are outside the main-text page limit.
  [CFP](https://virtual.aistats.org/Conferences/2027/CallForPapers)

## Version-specific hygiene

Version 40 redefines both `\cb` and `\ih` to empty macros before
`\begin{document}`. Their contents therefore do **not** render as visible
author annotations. Older artifact warnings about visible annotations refer
to version 38 and must not be applied to version 40. If manuscript source is
ever distributed, remove private editorial exchanges and unresolved author
notes from that source; this is distinct from checking the rendered PDF.

This audit does not attest that the source has been compiled successfully,
that every bibliography entry resolves, that the final main-text page count
is compliant, or that the authors completed independent verification. Those
are separate final-document and author checks.
