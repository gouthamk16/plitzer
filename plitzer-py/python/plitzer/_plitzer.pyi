import os
import sys
from collections.abc import Sequence
from typing import SupportsIndex

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
    """A C-order int32 array, read through DLPack or Arrow without a copy.

    Every export of one array shares its memory, as numpy views do: a write
    through one is seen by the others. Use `copy=True` for separate memory.
    """

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
        location: str | os.PathLike[str],
        *,
        seed: SupportsIndex,
        lanes: SupportsIndex,
        window: SupportsIndex,
        seq_len: SupportsIndex,
        docs: SupportsIndex,
        cap: SupportsIndex,
        pad: SupportsIndex,
        budget: SupportsIndex,
        rank: SupportsIndex | None = None,
        world: SupportsIndex | None = None,
        column: str = "tokens",
        cache: str | os.PathLike[str] | None = None,
        storage_options: dict[str, str | bool] | None = None,
    ) -> None: ...
    def __iter__(self) -> Loader:
        """The rest of the current epoch, or the next epoch if it has ended."""
    def __next__(self) -> Batch:
        """The next step; `StopIteration` from the end of the epoch on."""
    def set_epoch(self, epoch: SupportsIndex) -> None:
        """Go to the start of `epoch` if it is later than the current one."""
    def checkpoint(self) -> bytes:
        """This GPU's part of a checkpoint: the state before the next step.

        It works during a step too, such as from a signal handler or another
        thread. `merge` joins the parts of all GPUs. A `KeyboardInterrupt`
        can drop a batch that this already counts as taken, as with any
        iterator: save between training steps, not in an `except` around
        the loop.
        """
    def resume(self, checkpoint: bytes | bytearray) -> None: ...
    @property
    def epoch(self) -> int:
        """The epoch of the next step, as in `checkpoint()`."""
    @property
    def lanes(self) -> range: ...

def merge(parts: Sequence[bytes | bytearray]) -> bytes:
    """One global checkpoint from the parts of all GPUs, in any order."""
