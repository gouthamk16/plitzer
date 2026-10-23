import sys
from collections.abc import Mapping, Sequence

if sys.version_info >= (3, 13):
    from types import CapsuleType
else:
    from typing_extensions import CapsuleType

class PlitzerError(Exception):
    """The base of every plitzer error."""

class ConfigError(PlitzerError):
    """A setting that cannot work, such as a world that does not divide lanes."""

class CheckpointError(PlitzerError):
    """A checkpoint that is damaged, or from another setting or dataset."""

class DataError(PlitzerError):
    """The data cannot be listed, read or decoded."""

class Array:
    """A C-order int32 array, read through DLPack or Arrow without a copy."""

    @property
    def shape(self) -> tuple[int, ...]: ...
    def __dlpack__(
        self,
        *,
        stream: object = None,
        max_version: tuple[int, int] | None = None,
        dl_device: tuple[int, int] | None = None,
        copy: bool | None = None,
    ) -> CapsuleType: ...
    def __dlpack_device__(self) -> tuple[int, int]: ...
    def __arrow_c_array__(
        self, requested_schema: object = None
    ) -> tuple[CapsuleType, CapsuleType]: ...

class Batch:
    """One step of one GPU."""

    @property
    def tokens(self) -> Array:
        """Shape (lanes, seq_len): one sequence per lane, padded with `pad`."""
    @property
    def lengths(self) -> Array:
        """Shape (lanes,): the real (not pad) tokens of each sequence."""

class Loader:
    """The lanes of GPU `rank` of `world`, from one epoch to the next.

    Iterate it for the rest of the current epoch; the next iteration gives
    the next epoch.
    """

    def __init__(
        self,
        location: str,
        *,
        seed: int,
        lanes: int,
        window: int,
        seq_len: int,
        docs: int,
        cap: int,
        pad: int,
        budget: int,
        rank: int = 0,
        world: int = 1,
        column: str = "tokens",
        cache: str | None = None,
        storage_options: Mapping[str, str] | None = None,
    ) -> None: ...
    def __iter__(self) -> Loader: ...
    def __next__(self) -> Batch: ...
    def set_epoch(self, epoch: int) -> None:
        """Go to the start of `epoch` if it is later than the current one."""
    def checkpoint(self) -> bytes:
        """This GPU's part of a checkpoint. `merge` joins the parts of all GPUs."""
    def resume(self, checkpoint: bytes) -> None: ...
    @property
    def epoch(self) -> int: ...
    @property
    def lanes(self) -> range: ...

def merge(parts: Sequence[bytes]) -> bytes:
    """One global checkpoint from the parts of all GPUs, in any order."""
