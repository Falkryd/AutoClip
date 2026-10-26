from __future__ import annotations

from pathlib import Path

import pytest

from src.common.config import AppConfig
from src.common.state import State


@pytest.fixture
def cfg(tmp_path: Path) -> AppConfig:
    return AppConfig.model_validate(
        {
            "channel": {"id": "UC" + "a" * 22},
            "paths": {
                "work_dir": str(tmp_path / "work"),
                "log_dir": str(tmp_path / "logs"),
                "state_dir": str(tmp_path / "state"),
            },
        }
    )


@pytest.fixture
def state(tmp_path: Path) -> State:
    return State(tmp_path / "state.sqlite3")


FIXTURES = Path(__file__).parent / "fixtures"
