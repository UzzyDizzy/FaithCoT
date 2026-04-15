#src/utils/cache.py
import os
import json
import hashlib

CACHE_DIR = "cache/results"

def _hash(text: str):
    return hashlib.md5(text.encode()).hexdigest()

def get_cache(key: str):
    path = os.path.join(CACHE_DIR, key + ".json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def save_cache(key: str, value):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, key + ".json")
    with open(path, "w") as f:
        json.dump(value, f)