"""The stream side of the end-to-end harness: the settings of every run, the
digest that a rank logs for each lane and step, the logs read back as one
stream, and the run that never stops, to compare with."""

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import numpy.typing as npt

import plitzer

if TYPE_CHECKING:
    from conftest import Settings

# 12 lanes: the world can be 1, 2, 3, 4 or 6.
SETTINGS: "Settings" = {
    "seed": 5,
    "lanes": 12,
    "window": 4,
    "seq_len": 16,
    "docs": 2,
    "cap": 24,
    "pad": 2**32 - 1,
    "budget": 1 << 30,
}

Stream = dict[tuple[int, int], str]


def digest(tokens: npt.NDArray[np.generic], length: int) -> str:
    data = tokens.tobytes() + length.to_bytes(4, "little")
    return hashlib.sha256(data).hexdigest()[:16]


def read(path: Path) -> Stream:
    """The complete lines of a log; a kill can leave the last one cut."""
    if not path.exists():
        return {}
    text = path.read_text()
    stream: Stream = {}
    for line in text[: text.rfind("\n") + 1].splitlines():
        step, lane, digest = line.split()
        key = (int(step), int(lane))
        assert key not in stream, f"{path.name} logs {key} twice"
        stream[key] = digest
    return stream


def stream(out: Path, world: int) -> Stream:
    joined: Stream = {}
    for r in range(world):
        part = read(out / f"log-{r}")
        twice = joined.keys() & part.keys()
        assert not twice, f"two ranks log {min(twice)}"
        joined.update(part)
    return joined


class Reference(NamedTuple):
    """One GPU, no stop."""

    ends: list[int]  # the step count at the end of each epoch
    steps: int  # the steps of the run
    want: Stream


def reference(location: Path, cache: Path, epochs: float) -> Reference:
    """One GPU, no stop, in this process, for `epochs` epochs."""
    loader = plitzer.Loader(location, **SETTINGS, cache=str(cache))
    want: list[dict[int, str]] = []
    ends = []
    for _ in range(int(epochs) + 1):
        for b in loader:
            tokens, lengths = np.from_dlpack(b.tokens), np.from_dlpack(b.lengths)
            lanes = range(SETTINGS["lanes"])
            want.append({i: digest(tokens[i], int(lengths[i])) for i in lanes})
        ends.append(len(want))
    whole = int(epochs)
    last = ends[whole] - (ends[whole - 1] if whole else 0)
    steps = (ends[whole - 1] if whole else 0) + int(last * (epochs - whole))
    stream = {(s, i): d for s in range(steps) for i, d in want[s].items()}
    return Reference(ends, steps, stream)


def check(want: Stream, got: Stream, steps: int, case: str) -> None:
    """The streams hold the same batch for each (step, lane) of the run."""
    keys = [(s, lane) for s in range(steps) for lane in range(SETTINGS["lanes"])]
    for key in keys:
        assert got.get(key) == want.get(key), f"{case}: first difference at {key}"
    assert set(got) == set(keys), f"{case}: steps or lanes outside the run"
