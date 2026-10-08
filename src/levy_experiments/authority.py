"""Frozen identifiers of the mathematical sources audited by this code."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

_AUTHORITY: dict[str, dict[str, Any]] = {
    "current_manuscript": {
        "name": "mainaistats (40).tex",
        "sha256": "4713D4441E62491322371B02C42D11673FC615B8C06D626560BFD36AAC9CA88C".lower(),
    },
    "execution_manuscript_snapshot": {
        "name": "mainaistats_no_A2 (2).tex",
        "sha256": "a2ceb81155566b43030677c90f5a62437146449118dacc17f0ec2b963dce56bf",
    },
    "independent_specification": {
        "name": "phase0_independent_spec_latest.md",
        "sha256": "f32b6586f8b043f634c0f01412fad194914f78a6494ae7cd9740914bdd1d7fc0",
    },
}


def authority_manifest() -> dict[str, dict[str, Any]]:
    """Return the immutable source identifiers embedded in every run."""

    return deepcopy(_AUTHORITY)
