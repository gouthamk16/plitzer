"""The PyPI description repeats the README's quick start: they must agree."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CODE = re.compile(r"```python\n(.*?)```", re.S)


@pytest.mark.skipif(not (ROOT / "DESCRIPTION.md").exists(), reason="no source tree")
def test_description_examples_are_the_readme_examples() -> None:
    description = CODE.findall((ROOT / "DESCRIPTION.md").read_text(encoding="utf-8"))
    readme = CODE.findall((ROOT / "README.md").read_text(encoding="utf-8"))
    assert description, "DESCRIPTION.md has its quick start"
    assert description == readme
