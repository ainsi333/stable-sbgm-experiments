# MIT licensing revision

Date: 2026-10-05. Scope: a separately versioned copy of the version-40 code
supplement. The preceding archive and the manuscript attachment are unchanged.
The requesting author explicitly authorized selection of distribution terms.

## Decision and scope

The MIT License is supplied in `LICENSE`, using the
[standard MIT text](https://opensource.org/license/mit). It covers the original
implementation and accompanying original repository material: configurations,
documentation, generated numerical outputs, and figures. It permits reuse,
modification, redistribution, and commercial use, subject to preservation of
the copyright and permission notices, without warranty. It does not impose
copyleft or an additional citation condition.

The anonymous copyright designation is "The authors of the accompanying
manuscript". The authors remain responsible for ensuring that they hold the
necessary rights, including any applicable coauthor/institutional permissions.
An identified public release should name the actual rights holders. This file
does not certify ownership or scientific validity. The manuscript is not
distributed in this archive and is not licensed by this grant.

Third-party dependencies retain their upstream terms in `THIRD_PARTY.md`.
No third-party source package, binary wheel, external dataset, or pretrained
model is included. Dependencies are installed separately. Distribution of a
built container or bundled environment would require retaining those
dependencies' own notices; the present source archive is not such an image.

## Changes and provenance

- Add `LICENSE` and this report; declare `license = "MIT"` and
  `license-files = ["LICENSE"]` in `pyproject.toml`.
- Update the README, third-party inventory, and active checklist/audit notes.
- Include `LICENSE` and `THIRD_PARTY.md` in all six Docker/Apptainer recipes.
- Extend wheel and container-input tests to check license inclusion.
- Refresh the archive's checksum manifest; keep the older archive intact.

The historical 2026-10-01 receipt in `RELEASE_CHECKS.md` remains a historical
receipt, not a claim that this licensing revision generated the original runs.
All 99 files under `src`, `experiments`, `configs`, `artifact_data`, and `figures`
were compared by SHA-256 and are identical to the preceding version-40 artifact.
No simulations or figures require regeneration for this licensing change.

Changing `pyproject.toml` changes the current execution-source hash. Historical
execution identities in the saved provenance are intentionally not rewritten.
Replot the frozen tables normally; do not resume a historical simulation with
this new source identity or bypass version-mixing safeguards.

## Validation of this revision

On the existing pinned Windows CPU environment (Python 3.12.10):

- Full suite: **245 passed, 2 skipped**, 132.24 seconds. Skips concern GPU
  availability; no GPU, Docker, or Apptainer execution is claimed.
- Ruff: all checks passed.
- The wheel build test verifies `License-Expression: MIT`, `License-File:
  LICENSE`, and exact equality between the packaged license and root `LICENSE`.
- Static tests check license/third-party-notice inclusion in all six container
  recipes. These do not replace a target-platform container build.
- Scientific source, configuration, data, and figure identity: 99/99 files
  unchanged by SHA-256 comparison with the preceding version-40 artifact.
- Saved-statistics verification passed: 24 initialization aggregate rows,
  including bootstrap intervals, and 94 tail-figure rows reproduced.
- Existing Matplotlib/PyParsing deprecation warnings remain visible (3,880
  warnings in the suite); they are not numerical test failures.

Commands, from the archive root after the README environment setup:

```bash
python -m ruff check . --no-cache
python -m pytest -p no:cacheprovider -q
python scripts/verify_artifact.py
python scripts/verify_saved_statistics.py
```

## Replacement checklist item

Use this answer only when the licensed archive accompanies the submission.
This replacement has not been applied to the attached manuscript source.

```latex
\item The license information of the assets, if applicable.
  \textbf{Yes.}
  The supplementary implementation and accompanying original materials
  are released under the MIT License, included as \texttt{LICENSE}.
  Appendix~\ref{app:exp:controls} identifies the licenses of JAX,
  NumPy, and SciPy; \texttt{THIRD\_PARTY.md} documents the dependencies
  and their upstream licenses. Third-party packages are installed
  separately and retain their own licenses.
```

Licensing resolves this checklist item only. The remaining scientific,
disclosure, and manuscript-layout qualifications in `MANUSCRIPT_ACTIONS.md`
are unchanged.
