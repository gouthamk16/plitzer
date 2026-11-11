"""Deterministic, resumable dataloader for Parquet on local disk and S3."""

from importlib.metadata import version

from plitzer._plitzer import (
    Array,
    Batch,
    CheckpointError,
    ConfigError,
    DataError,
    Loader,
    PlitzerError,
    merge,
)

__version__ = version("plitzer")

__all__ = [
    "Array",
    "Batch",
    "CheckpointError",
    "ConfigError",
    "DataError",
    "Loader",
    "PlitzerError",
    "__version__",
    "merge",
]
