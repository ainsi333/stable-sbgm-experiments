"""Deterministic, role-separated JAX random streams."""

from __future__ import annotations

import hashlib


def label_token(label: str) -> int:
    """Return a stable uint32 token independent of Python's randomized hash."""

    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def named_key(seed: int, label: str, index: int = 0):
    """Derive a JAX key from an auditable semantic label and integer index."""

    import jax

    key = jax.random.PRNGKey(seed)
    key = jax.random.fold_in(key, label_token(label))
    return jax.random.fold_in(key, index)


def stream_manifest(labels: tuple[str, ...]) -> dict[str, int]:
    """Record the exact fold-in token used for every semantic stream."""

    return {label: label_token(label) for label in labels}
