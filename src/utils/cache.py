#src/utils/cache.py
import os
import json
import hashlib

CACHE_DIR = "cache/results"

def _hash(text: str):
    return hashlib.md5(text.encode()).hexdigest()

def get_cache(key):
    import os, json

    path = os.path.join(CACHE_DIR, key + ".json")

    if not os.path.exists(path):
        return None

    try:
        with open(path) as f:
            return json.load(f)

    except Exception:
        # ❌ corrupted cache → delete it
        try:
            os.remove(path)
        except:
            pass
        return None

def save_cache(key, value):
    import os, json, tempfile

    os.makedirs(CACHE_DIR, exist_ok=True)

    final_path = os.path.join(CACHE_DIR, key + ".json")

    safe_value = {
        "raw_output": value.get("raw_output", ""),
        "num_steps": value.get("num_steps", 0),
        "num_generated_tokens": value.get("num_generated_tokens", 0),
    }

    # ✅ write to temp file first
    with tempfile.NamedTemporaryFile("w", delete=False, dir=CACHE_DIR) as tmp:
        json.dump(safe_value, tmp)
        temp_path = tmp.name

    # ✅ atomic replace
    os.replace(temp_path, final_path)