"""A PyTorch ``IterableDataset`` over a plitzer ``Loader``.

Each item is one step of this GPU: a ``TokenBatch`` of int32 tensors that
share memory with the loader's export. Each item is already a batch::

    dataset = plitzer.torch.Dataset("s3://bucket/data", seed=0, lanes=512, ...)
    loader = DataLoader(dataset, batch_size=None, num_workers=0)

The reads run in Rust threads, so ``num_workers`` must be 0. The dataset
splits the lanes by rank itself, so do not let a framework shard it again
(such as ``accelerator.prepare`` on its DataLoader).

``state_dict()`` gathers the parts of all ranks of the data-parallel group
into one global checkpoint, so every rank must call it, as Lightning and
``torch.distributed.checkpoint`` do. Each rank then holds the same bytes,
and they resume on any number of GPUs. With NCCL, call
``torch.cuda.set_device`` first, as torch needs for object collectives. If
the training loop fetches batches before it trains them, pass ``ahead``:
the state then names the first batch not yet trained.
"""

import operator
import os
from collections import deque
from collections.abc import Generator, Mapping
from typing import NamedTuple, NotRequired, SupportsIndex, TypedDict, Unpack

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info
from torch.utils.dlpack import from_dlpack

import plitzer

_ONE_PROCESS = (
    "plitzer reads in Rust threads: a DataLoader worker process cannot use "
    "them, and a Dataset cannot be pickled or copied; use DataLoader(num_workers=0)"
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
    the next epoch. ``ahead`` is the number of batches the training loop
    takes before it trains them, such as 1 for a fetcher that prefetches one.
    """

    def __init__(
        self,
        location: str | os.PathLike[str],
        *,
        group: dist.ProcessGroup | None = None,
        ahead: int = 0,
        **settings: Unpack[LoaderSettings],
    ) -> None:
        super().__init__()
        if isinstance(ahead, bool) or not isinstance(ahead, int) or ahead < 0:
            raise plitzer.ConfigError(
                f"ahead must be an int of 0 or more, not {ahead!r}"
            )
        self._group, self._ahead = group, ahead
        rank, self._world = _placement(group)
        self._loader = plitzer.Loader(
            location, rank=rank, world=self._world, **settings
        )
        # With `ahead`: (epoch, checkpoint) before each of the last `ahead`
        # batches, and now. The first is the state of the first batch that
        # the loop has not trained.
        self._seen: deque[tuple[int, bytes]] = deque(maxlen=ahead + 1)
        # Set when an iterator ends early: the loop dropped the batches it
        # took but did not train, so the next iterator gives them again.
        self._rewind = False
        # Bumped by each iterator and each jump: an older iterator then
        # leaves the bookkeeping alone.
        self._generation = 0
        self._restart()

    def __iter__(self) -> Generator[TokenBatch, None, None]:
        if get_worker_info() is not None:
            raise RuntimeError(_ONE_PROCESS)
        self._check_world()
        generation = self._generation = self._generation + 1
        if self._rewind:
            self._loader.resume(self._seen[0][1])
            self._rewind = False
            self._restart()
        ended = False
        try:
            for batch in self._loader:
                if self._ahead and generation == self._generation:
                    self._remember()
                yield TokenBatch(from_dlpack(batch.tokens), from_dlpack(batch.lengths))
            ended = True
        finally:
            # No I/O here: this can run at teardown or while an error unwinds.
            if generation == self._generation:
                if ended:  # the loop trains each batch it took
                    self._restart()
                elif len(self._seen) > 1:
                    self._rewind = True

    def __getstate__(self) -> object:
        # A DataLoader worker made by spawn gets a pickled copy; refuse it as
        # clearly as a forked one.
        raise RuntimeError(_ONE_PROCESS)

    def set_epoch(self, epoch: SupportsIndex) -> None:
        """Go to the start of ``epoch`` if it is later than the current one."""
        before = self._loader.checkpoint()
        self._loader.set_epoch(epoch)
        moved = self._loader.checkpoint() != before
        # Past the epoch of the untrained batches, the loop leaves them, also
        # when the loader already stands there after the last step.
        if moved or operator.index(epoch) > self._seen[0][0]:
            self._jump()

    def state_dict(self) -> State:
        """The global checkpoint before the first batch not yet trained, the
        same on every rank. A collective over the group: every rank must
        call it."""
        self._check_world()
        part = self.part()
        if self._world == 1:
            return {"plitzer": plitzer.merge([part])}
        gathered: list[object] = [None] * self._world
        dist.all_gather_object(gathered, part, group=self._group)
        # Each slot holds a rank's part; merge refuses missing lanes.
        parts = [p for p in gathered if isinstance(p, bytes)]
        return {"plitzer": plitzer.merge(parts)}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Stand where ``state`` was taken, on any number of GPUs."""
        checkpoint = state.get("plitzer")
        if not isinstance(checkpoint, bytes | bytearray):
            raise plitzer.CheckpointError('a plitzer state is {"plitzer": bytes}')
        self._loader.resume(checkpoint)
        self._jump()

    def part(self) -> bytes:
        """This rank's part of ``state_dict()``, not a collective.
        ``plitzer.merge`` joins the parts of all ranks, for checkpoint code
        that gathers them itself."""
        return self._seen[0][1] if self._ahead else self._loader.checkpoint()

    @property
    def epoch(self) -> int:
        """The epoch of the next step."""
        return self._loader.epoch

    @property
    def lanes(self) -> range:
        """This rank's lanes of the global batch."""
        return self._loader.lanes

    def _restart(self) -> None:
        """Forget the batches before now: none of them waits to be trained."""
        self._seen.clear()
        self._remember()

    def _remember(self) -> None:
        self._seen.append((self._loader.epoch, self._loader.checkpoint()))

    def _jump(self) -> None:
        """After a move to another position: older iterators and their
        untrained batches no longer count."""
        self._generation += 1
        self._rewind = False
        self._restart()

    def _check_world(self) -> None:
        # A Dataset made before init_process_group (with no WORLD_SIZE to
        # tell) took rank 0 of 1, and every rank would read all lanes.
        if dist.is_available() and dist.is_initialized():
            if dist.get_world_size(self._group) != self._world:
                raise plitzer.ConfigError(
                    "the Dataset was made before torch.distributed was "
                    "initialized; make it after init_process_group"
                )


def _placement(group: dist.ProcessGroup | None) -> tuple[int, int]:
    """This process's rank in ``group``, and the group's size."""
    if dist.is_available() and dist.is_initialized():
        rank, world = dist.get_rank(group), dist.get_world_size(group)
        if rank < 0:
            raise plitzer.ConfigError("this process is not in the group")
        return rank, world
    if group is not None:
        raise plitzer.ConfigError("group needs torch.distributed to be initialized")
    # Under torchrun, before init_process_group: every rank would read all
    # lanes and train on the same data, with no error.
    size = os.environ.get("WORLD_SIZE", "")
    if size.isdigit() and int(size) > 1:
        raise plitzer.ConfigError(
            f"WORLD_SIZE is {size}, but torch.distributed is not initialized; "
            "call init_process_group before making the Dataset, or unset "
            "WORLD_SIZE for one process"
        )
    return 0, 1
