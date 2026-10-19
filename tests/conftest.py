import shutil
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CASE_FOLDERS = ["CASE_1_Acme_Bank", "CASE_2_Northstar_Telecom"]
ACME = "case_1_acme_bank"
NORTHSTAR = "case_2_northstar_telecom"


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def cases_copy(tmp_path: Path) -> Path:
    """A disposable copy of both original case folders, so tests never touch the originals."""
    root = tmp_path / "cases"
    root.mkdir()
    for folder in CASE_FOLDERS:
        shutil.copytree(REPO_ROOT / folder, root / folder)
    return root
