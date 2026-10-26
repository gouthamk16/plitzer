"""A PyTorch ``IterableDataset`` over a plitzer ``Loader``.

Each item is one step of this GPU: a ``TokenBatch`` of int32 tensors that
share memory with the loader's export. Each item is already a batch::

    dataset = plitzer.torch.Dataset("s3://bucket/data", seed=0, lanes=512, ...)
    loader = DataLoader(dataset, batch_size=None, num_workers=0)

The reads run in Rust threads, so ``num_workers`` must be 0. ``state_dict()``
gathers the parts of all ranks of the data-parallel group into one global
checkpoint, so every rank must call it, as Lightning, Accelerate and
``torch.distributed.checkpoint`` do. Each rank then holds the same bytes,
and they resume on any number of GPUs.
"""

import os
from collections.abc import Iterator, Mapping
from typing import NamedTuple, NotRequired, SupportsIndex, TypedDict, Unpack

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info
from torch.utils.dlpack import from_dlpack

import plitzer

_ONE_PROCESS = (
    "plitzer reads in Rust threads, which a DataLoader worker process "
    "cannot use; use DataLoader(num_workers=0)"
)


class TokenBatch(NamedTuple):
    """One step of one GPU. A NamedTuple, so ``DataLoader`` passes it on as
    it is, also with ``pin_memory=True``."""

    tokens: torch.Tensor
    """Shape (lanes, seq_len), int32: one sequence per lane, padded with ``pad``."""
    lengths: torch.Tensor
    """Shape (lanes,), int32: the real (not pad) tokens of each sequence."""


class LoaderSettings(TypedDict):
    """The keywords of ``plitzer.Loader`` but ``rank`` and ``world``, which
    come from the process group."""

    seed: SupportsIndex
    lanes: SupportsIndex
    window: SupportsIndex
    seq_len: SupportsIndex
    docs: SupportsIndex
    cap: SupportsIndex
    pad: SupportsIndex
    budget: SupportsIndex
    column: NotRequired[str]
    cache: NotRequired[str | os.PathLike[str] | None]
    storage_options: NotRequired[Mapping[str, str | bool] | None]


class State(TypedDict):
    """The state of a ``Dataset``: a global checkpoint, as bytes, so
    ``torch.load(weights_only=True)`` reads it back."""

    plitzer: bytes


class Dataset(IterableDataset[TokenBatch]):
    """The lanes of this rank of the data-parallel ``group`` (the default
    process group if ``None``; rank 0 of 1 without ``torch.distributed``).

    Iterate it for the rest of the current epoch; the next iteration gives
    the next epoch.
    """

    def __init__(
        self,
        location: str | os.PathLike[str],
        *,
        group: dist.ProcessGroup | None = None,
        **settings: Unpack[LoaderSettings],
    ) -> None:
        super().__init__()
        self._group = group
        self._world = 1
        rank = 0
        if dist.is_available() and dist.is_initialized():
            rank, self._world = dist.get_rank(group), dist.get_world_size(group)
        elif group is not None:
            raise plitzer.ConfigError("group needs torch.distributed to be initialized")
        world = self._world
        self._loader = plitzer.Loader(location, rank=rank, world=world, **settings)

    def __iter__(self) -> Iterator[TokenBatch]:
        if get_worker_info() is not None:
            raise RuntimeError(_ONE_PROCESS)
        for batch in self._loader:
            yield TokenBatch(from_dlpack(batch.tokens), from_dlpack(batch.lengths))

    def __getstate__(self) -> object:
        # A DataLoader worker made by spawn gets a pickled copy; refuse it as
        # clearly as a forked one.
        raise RuntimeError(_ONE_PROCESS)

    def set_epoch(self, epoch: SupportsIndex) -> None:
        """Go to the start of ``epoch`` if it is later than the current one."""
        self._loader.set_epoch(epoch)

    def state_dict(self) -> State:
        """The global checkpoint before the next step, the same on every rank.
        A collective over the group: every rank must call it."""
        part = self._loader.checkpoint()
        if self._world == 1:
            return {"plitzer": plitzer.merge([part])}
        gathered: list[object] = [None] * self._world
        dist.all_gather_object(gathered, part, group=self._group)
        parts = [p for p in gathered if isinstance(p, bytes)]
        if len(parts) != self._world:
            raise plitzer.CheckpointError("a rank gave no checkpoint part")
        return {"plitzer": plitzer.merge(parts)}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Stand where ``state`` was taken, on any number of GPUs."""
        checkpoint = state.get("plitzer")
        if not isinstance(checkpoint, bytes | bytearray):
            raise plitzer.CheckpointError('a plitzer state is {"plitzer": bytes}')
        self._loader.resume(checkpoint)

    def part(self) -> bytes:
        """This rank's part of a checkpoint, not a collective. ``plitzer.merge``
        joins the parts of all ranks, for checkpoint code that gathers them."""
        return self._loader.checkpoint()

    @property
    def epoch(self) -> int:
        """The epoch of the next step."""
        return self._loader.epoch

    @property
    def lanes(self) -> range:
        """This rank's lanes of the global batch."""
        return self._loader.lanes
