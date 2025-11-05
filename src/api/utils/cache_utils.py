import os
import json
from typing import Any

def load_cache(file_path: str, default: Any = None) -> Any:
    """Loading cache from JSON file; returns default if file missing or invalid."""
    if not os.path.exists(file_path):
        return default
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception:
        return default

def save_cache(file_path: str, data: Any):
    """Saving cache to JSON file; creates parent directories if missing."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)