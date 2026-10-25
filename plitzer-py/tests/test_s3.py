"""The Loader over S3. CI's `s3` job runs this against versitygw; locally,
set PLITZER_S3_ENDPOINT (and the PLITZER_S3_* keys) to run it."""

import os
import time
from pathlib import Path

import numpy as np
import pyarrow.fs as pafs
import pytest

import plitzer
from conftest import SETTINGS

ENDPOINT = os.environ.get("PLITZER_S3_ENDPOINT")
pytestmark = pytest.mark.skipif(ENDPOINT is None, reason="PLITZER_S3_ENDPOINT not set")


BUCKET = os.environ.get("PLITZER_S3_BUCKET", "plitzer-test")
KEY = os.environ.get("PLITZER_S3_ACCESS_KEY", "test")
SECRET = os.environ.get("PLITZER_S3_SECRET_KEY", "testsecret")


def upload(dataset: Path) -> str:
    """The files of `dataset` under a new S3 prefix, byte for byte, so with
    the same row groups and the same order. Returns the prefix."""
    assert ENDPOINT is not None
    scheme, host = ENDPOINT.split("://")
    fs = pafs.S3FileSystem(
        access_key=KEY,
        secret_key=SECRET,
        endpoint_override=host,
        scheme=scheme,
        region="us-east-1",
    )
    prefix = f"py-{time.time_ns()}"
    for path in sorted(dataset.iterdir()):
        with fs.open_output_stream(f"{BUCKET}/{prefix}/{path.name}") as out:
            out.write(path.read_bytes())
    return prefix


def test_s3_matches_local(dataset: Path, tmp_path: Path) -> None:
    assert ENDPOINT is not None
    options: dict[str, str | bool] = {
        "endpoint": ENDPOINT,
        "allow_http": True,  # a bool becomes "true"
        "region": "us-east-1",
        "access_key_id": KEY,
        "secret_access_key": SECRET,
    }
    remote = plitzer.Loader(
        f"s3://{BUCKET}/{upload(dataset)}",
        cache=str(tmp_path / "s3.idx"),
        storage_options=options,
        **SETTINGS,
    )
    local = plitzer.Loader(str(dataset), cache=str(tmp_path / "local.idx"), **SETTINGS)
    pairs = list(zip(remote, local, strict=True))
    assert pairs
    for a, b in pairs:
        assert np.array_equal(np.from_dlpack(a.tokens), np.from_dlpack(b.tokens))
