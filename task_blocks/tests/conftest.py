from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data_dir = tmp_path / "data"
    monkeypatch.setenv("DATA_DIR", str(data_dir))

    from task_blocks.config import get_settings

    get_settings.cache_clear()
    yield data_dir
    get_settings.cache_clear()


@pytest.fixture
def clear_asset_cache():
    from task_blocks import assets

    assets.load_prompt.cache_clear()
    assets.load_soul.cache_clear()
    assets.load_skills.cache_clear()
    yield
    assets.load_prompt.cache_clear()
    assets.load_soul.cache_clear()
    assets.load_skills.cache_clear()
