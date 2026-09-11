"""Configuration discovery and loading."""

from __future__ import annotations

import json
from pathlib import Path

from .simulator import load_config


CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def config_path(name: str) -> Path:
    """Return the path of a bundled JSON configuration."""
    filename = name if name.endswith(".json") else f"{name}.json"
    path = CONFIG_DIR / filename
    if not path.is_file():
        available = ", ".join(sorted(item.stem for item in CONFIG_DIR.glob("*.json")))
        raise ValueError(f"unknown bundled config {name!r}; available: {available}")
    return path


def load_json(path: str | Path) -> dict:
    """Load one JSON object and reject non-object roots."""
    result = json.loads(Path(path).read_text())
    if not isinstance(result, dict):
        raise ValueError(f"configuration root must be an object: {path}")
    return result


def load_default_simulator_config(profile: str = "qp_ode_generic") -> dict:
    """Load the bundled base simulator config plus a bundled profile."""
    return load_config(config_path("simulator_base"), config_path(profile))
