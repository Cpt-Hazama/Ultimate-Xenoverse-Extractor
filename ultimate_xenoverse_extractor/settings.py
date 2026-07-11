from __future__ import annotations

import json
from pathlib import Path

APP_DIR = Path.home() / ".ultimate_xenoverse_extractor"
CONFIG_PATH = APP_DIR / "config.json"


def load_settings() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_settings(data: dict) -> None:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
