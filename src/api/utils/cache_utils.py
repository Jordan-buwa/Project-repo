import json
import os

def load_cache(file_path, default=None):
    if not os.path.exists(file_path):
        return default if default is not None else []
    with open(file_path, "r") as f:
        return json.load(f)

def save_cache(file_path, data):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)
