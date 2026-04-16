#src/data/download_datasets.py
import os
from datasets import load_dataset

DATASET_MAP = {
    "gsm8k": ("openai/gsm8k", "main"),
    "math": ("qwedsacf/competition_math", None),
    "strategyqa": ("ChilleD/StrategyQA", None),
    "arc-challenge": ("ai2_arc", "ARC-Challenge"),
    "folio": ("tasksource/folio", None)
}

def download_all_datasets(output_dir: str, cache_dir: str = None):
    os.makedirs(output_dir, exist_ok=True)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    download_info = {}

    for bench, (dataset_id, config) in DATASET_MAP.items():
        print(f"\nDownloading {bench}...")

        if dataset_id is None:
            print(f"  ❌ {bench} not available.")
            download_info[bench] = {"error": "not available"}
            continue

        try:
            # FIX: NO use_auth_token ANYMORE
            if config:
                ds = load_dataset(dataset_id, config, cache_dir=cache_dir)
            else:
                ds = load_dataset(dataset_id, cache_dir=cache_dir)

            splits = list(ds.keys())
            sizes = {s: len(ds[s]) for s in splits}

            download_info[bench] = {
                "splits": splits,
                "sizes": sizes
            }

            print(f"  ✅ Done: {splits}")
            for s in splits:
                print(f"    {s}: {sizes[s]}")

        except Exception as e:
            print(f"  ❌ Failed: {e}")
            download_info[bench] = {"error": str(e)}

    print("\n=== Summary ===")
    for k, v in download_info.items():
        if "error" in v:
            print(f"{k}: ❌ {v['error']}")
        else:
            print(f"{k}: {v['splits']}")

    return download_info