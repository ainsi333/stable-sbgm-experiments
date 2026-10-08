# Version-40 author actions (manuscript not edited)

Authority: `mainaistats (40).tex`, SHA-256
`4713d4441e62491322371b02c42d11673fc615b8c06d626560bfd36aac9ca88c`.
Audit completed 2026-10-01. These are proposed changes, not edits already made
to the attachment. The detailed checklist audit is in `CHECKLIST_V40.md`.

## 1. Complete the AI disclosure and confirm human verification

The statement is correctly placed before references. However, its active
paragraph at line 903 describes mathematical proofreading and implementation/
verification, whereas the work record includes assistance with proposing
experiments, refining protocols and discussing interpretation. Add at minimum:

> We also used these tools for feedback on experimental design and methodology,
> numerical implementation, and the interpretation and limitations of numerical
> diagnostics.

The commented source at lines 908--931 describes broader mathematical/proof
assistance than the active text. Authors must resolve which account accurately
describes actual use; comments are not public disclosure. The fuller candidate
in `CHECKLIST_V40.md` is conditional on that confirmation. Do not add claims
about proof assistance merely because an audit proposed them. Conversely,
do not retain negative statements about methodology that omit actual help.

Line 905 asserts independent author verification of all assisted arguments.
Tests performed by this assistant cannot establish that humans did those
checks. Authors must confirm or accurately narrow the statement and mirror
the finalized disclosure in OpenReview. No human verification is attested here.

## 2. Correct figure insertion scale

The PDFs have no cropped objects or overlaps, but the current LaTeX reduces
their text severely. Official style: 6.75-inch text width, 0.25-inch separation,
thus a 3.25-inch column. Geometry from the actual PDFs gives:

| Source | Current width | PDF width (points) | Approximate legend size after scaling |
| --- | --- | ---: | ---: |
| 613, denoiser | half-column wrap, image at 99% | 346.8 | 2.3 pt, originally 7 |
| 659, initialization | one column, ordinary figure | 516 | 3.1 pt, originally 6.8 |
| 853--854, tail ratio | half-column wrap | 188.416 | 4.3 pt, originally 7 |
| appendix tail panels | full-width one-column appendix | 523.2 | 6.5 pt |

These are source-based geometric estimates, corroborated by reduced-size
raster inspection, not measurements from a compiled version-40 main PDF.
They are readability concerns, not assertions of automatic desk rejection.

Recommendation: initialization in `figure*` at `width=\textwidth`; denoiser
and standalone ratio in ordinary full-column `figure` floats. A native
column-width re-export of the denoiser with larger fonts would improve its
remaining small legend. Do not alter data or hide uncertainty for typography.
Recheck the eight-page main limit after moving floats. No manuscript or
distributed figure was silently changed for this recommendation.

## 3. Qualify the Experiment 2 control sentence

Lines 3186--3187 can be read as certifying interpolation error at both displayed
orders. The historical score-sensitivity arrays concern only p=1 and1.25.
The postprocessing specification explicitly records
`score_sensitivity_gate_recomputed: false`. The high-resolution particles
were not retained, so W1.4 cannot be recovered from those scalar distances.
Norm monotonicity bounds W1.1 by W1.25, but gives no such bound for W1.4.

Suggested replacement:

> The displayed orders were evaluated in post-processing from saved terminal
> samples. At T=2, temporal refinements N=156,312,624 are available at both
> displayed orders. The doubled-score-table sensitivity controls were
> originally evaluated at p=1 and p=1.25; they do not constitute an
> order-specific validation at p=1.4.

Add `using 20,000 bootstrap resamples` to the preceding uncertainty sentence.
The new `artifact_data/exp2/temporal_refinement_recomputed.csv` records all
144 corresponding rows without propagation. Do not treat them as an
unannounced continuous-time rate estimate. A stronger score claim requires
an explicitly authorized, separately recorded high-resolution rerun.

## 4. Make the main-text captions self-contained

Suggested additions, retaining existing model and seed information:

- Denoiser: "The maximum is evaluated numerically on a finite grid in
  [-40,40]; the dashed curve is the Brownian reference exp(t/2).
  No sampling uncertainty is involved." It is not an exact global supremum.
- Tail ratio: "The VP baseline is shown only for nu=1.5. Shaded bands are
  descriptive 16%--84% inter-seed ranges, not confidence intervals."
  These details are currently commented out in the main caption but present
  in the appendix.

## 5. Checklist and final-document boundaries

The 18 checklist questions match the official template. Keep the honest No answer
for complete complexity analysis. With the authorized MIT release of 2026-10-05,
change the asset-license item to Yes using `LICENSING_RELEASE.md`. Keep
Not Applicable for training and human subjects. No GPU result is claimed;
GPU execution is not a prerequisite for the reported CPU results.
The license covers original supplementary material; dependency terms are unchanged.

Version 40 disables both editorial macros before the document starts.
The old warning about visible author annotations is withdrawn. If source is
later shared, omit private editorial exchanges from it.

An identified compiled version-40 PDF with matching bibliography inputs was
not supplied. Page count, resolved citations, final float placement and
complete manuscript PDF metadata cannot be certified from code and source
alone. The four filenames in `TRACEABILITY.md` must match the compiled paper.
Do not upload contradictory supplementary versions. Official links and the
question-by-question evidence are in `CHECKLIST_V40.md`.
