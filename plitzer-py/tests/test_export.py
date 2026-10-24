import gc
import sys
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pytest

import plitzer
from conftest import MakeLoader

if sys.platform != "win32":
    import resource

if TYPE_CHECKING:
    from typing_extensions import CapsuleType


def test_batch_shapes_and_padding(make: MakeLoader) -> None:
    batch = next(make())
    assert batch.tokens is batch.tokens
    assert batch.tokens.shape == (4, 8)
    assert batch.lengths.shape == (4,)
    tokens = np.from_dlpack(batch.tokens)
    lengths = np.from_dlpack(batch.lengths)
    assert tokens.dtype == np.int32 and lengths.dtype == np.int32
    for row, n in zip(tokens, lengths, strict=True):
        assert (row[n:] == -1).all(), "pad 2**32 - 1 reads as -1"
        assert (row[:n] >= 0).all()


class Legacy:
    """A producer as old consumers call it: `__dlpack__()` with no arguments."""

    def __init__(self, array: plitzer.Array) -> None:
        self.array = array

    def __dlpack__(self, *, stream: None = None, **_: object) -> "CapsuleType":
        return self.array.__dlpack__()

    def __dlpack_device__(self) -> tuple[int, int]:
        return self.array.__dlpack_device__()


def test_negative_pad_is_the_same_token(make: MakeLoader) -> None:
    a = np.from_dlpack(next(make(pad=-1)).tokens)
    b = np.from_dlpack(next(make(pad=2**32 - 1)).tokens)
    assert np.array_equal(a, b) and (a == -1).any()


def test_dlpack_versioned_and_legacy(make: MakeLoader) -> None:
    batch = next(make())
    assert '"dltensor_versioned"' in repr(batch.tokens.__dlpack__(max_version=(1, 0)))
    assert '"dltensor"' in repr(batch.tokens.__dlpack__())
    versioned = np.from_dlpack(batch.tokens)
    legacy = np.from_dlpack(Legacy(batch.tokens))
    assert np.array_equal(versioned, legacy)
    assert batch.tokens.__dlpack_device__() == (1, 0)


def max_rss_kib() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


@pytest.mark.skipif(sys.platform != "linux", reason="ru_maxrss is in KiB on Linux")
def test_unused_capsules_are_freed(make: MakeLoader) -> None:
    batch = next(make())
    for _ in range(10_000):  # reach the peak RSS before measuring
        batch.tokens.__dlpack__()
    before = max_rss_kib()
    # Each leaked capsule would keep about 200 bytes: about 60 MiB in all.
    for _ in range(150_000):
        batch.tokens.__dlpack__()
        batch.tokens.__dlpack__(max_version=(1, 0))
    gc.collect()
    assert max_rss_kib() - before < 16 * 1024


def test_data_outlives_the_batch_and_the_loader(make: MakeLoader) -> None:
    loader = make()
    batch = next(loader)
    tokens = np.from_dlpack(batch.tokens)
    table = pa.array(batch.tokens)
    want = tokens.copy()
    del batch, loader
    gc.collect()
    # Fill the heap, so a freed buffer would likely be reused and differ.
    noise = [np.full(32, 7, np.int32) for _ in range(10_000)]
    assert np.array_equal(tokens, want)
    assert np.array_equal(table.to_numpy_ndarray(), want)
    del noise


def test_copy_and_device(make: MakeLoader) -> None:
    batch = next(make())
    shared = np.from_dlpack(batch.tokens)
    copied = np.from_dlpack(batch.tokens, copy=True)
    copied[0, 0] = 12345
    assert shared[0, 0] != 12345
    with pytest.raises(BufferError):
        batch.tokens.__dlpack__(dl_device=(2, 0))


def test_arrow_export(make: MakeLoader) -> None:
    batch = next(make())
    tokens = pa.array(batch.tokens)
    assert isinstance(tokens.type, pa.FixedShapeTensorType)
    assert tokens.type.shape == [8]
    assert np.array_equal(tokens.to_numpy_ndarray(), np.from_dlpack(batch.tokens))
    lengths = pa.array(batch.lengths)
    assert lengths.type == pa.int32()
    assert np.array_equal(lengths.to_numpy(), np.from_dlpack(batch.lengths))
