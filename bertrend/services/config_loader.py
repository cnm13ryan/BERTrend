from __future__ import annotations

from pathlib import Path
from typing import Any

from bertrend import load_toml_config


class ConfigLoader:
    """Tiny wrapper around `load_toml_config` to enable DI & mocking."""

    def __call__(self, path: str | Path) -> dict[str, Any]:
        return load_toml_config(path)
