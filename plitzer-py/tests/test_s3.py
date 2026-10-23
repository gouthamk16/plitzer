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


def test_s3_matches_local(dataset: Path, tmp_path: Path) -> None:
    assert ENDPOINT is not None
    bucket = os.environ.get("PLITZER_S3_BUCKET", "plitzer-test")
    key = os.environ.get("PLITZER_S3_ACCESS_KEY", "test")
    secret = os.environ.get("PLITZER_S3_SECRET_KEY", "testsecret")
    scheme, host = ENDPOINT.split("://")
    fs = pafs.S3FileSystem(
        access_key=key,
        secret_key=secret,
        endpoint_override=host,
        scheme=scheme,
        region="us-east-1",
    )
    prefix = f"py-{time.time_ns()}"
    # The same bytes, so the same row groups and the same order.
    for path in sorted(dataset.iterdir()):
        with fs.open_output_stream(f"{bucket}/{prefix}/{path.name}") as out:
            out.write(path.read_bytes())
    options = {
        "endpoint": ENDPOINT,
        "allow_http": "true",
        "region": "us-east-1",
        "access_key_id": key,
        "secret_access_key": secret,
    }
    remote = plitzer.Loader(
        f"s3://{bucket}/{prefix}",
        cache=str(tmp_path / "s3.idx"),
        storage_options=options,
        **SETTINGS,  # type: ignore[arg-type]
    )
    local = plitzer.Loader(str(dataset), cache=str(tmp_path / "local.idx"), **SETTINGS)  # type: ignore[arg-type]
    pairs = list(zip(remote, local, strict=True))
    assert pairs
    for a, b in pairs:
        assert np.array_equal(np.from_dlpack(a.tokens), np.from_dlpack(b.tokens))
