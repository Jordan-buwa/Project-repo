# src/models/utils/cache_utils.py
import json
import os
from typing import Any

def load_cache(file_path: str, default: Any = None) -> Any:
    """
    Load cache from a JSON file.
    Returns `default` if file doesn't exist or cannot be read.
    """
    if not os.path.exists(file_path):
        return default
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception:
        return default


def save_cache(file_path: str, data: Any):
    """
    Save data to a JSON cache file.
    Creates parent directories if missing.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)
