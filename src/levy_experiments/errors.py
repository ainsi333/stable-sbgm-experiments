"""Explicit domain errors used by the experiment pipeline."""


class ExperimentError(RuntimeError):
    """Base class for a controlled experiment failure."""


class ConfigurationError(ExperimentError):
    """Raised when a configuration is incomplete or scientifically inconsistent."""


class ArtifactError(ExperimentError):
    """Raised when an output is missing, corrupt, stale, or would be overwritten."""


class NumericalError(ExperimentError):
    """Raised when a simulation produces a non-finite numerical value."""
