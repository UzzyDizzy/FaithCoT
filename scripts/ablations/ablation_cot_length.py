#scripts/ablations/ablation_cot_length.py
"""
Ablation 2: CoT Length Effect on Faithfulness.

Analyzes how the length of CoT (number of steps) correlates with
faithfulness metrics. Bins: short (<=5), medium (5-15), long (>15).
"""

import os, sys, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from configs.model_config import MODEL_REGISTRY
from configs.experiment_config import EXPERIMENT_CONFIG, PATHS, ensure_dirs
from src.utils.logger import setup_logger

logger = setup_logger("ablation_cot_length")

COT_LENGTH_BINS = EXPERIMENT_CONFIG.get("cot_length_bins", {
    "short": (0, 5),
    "medium": (6, 15),
    "long": (16, float("inf"))
})


def run_cot_length_ablation():
    """Analyze faithfulness by CoT length bins."""
    ensure_dirs()
    output_dir = os.path.join(PATHS["raw_results_dir"], "ablations", "cot_length")
    os.makedirs(output_dir, exist_ok=True)

    # Load predictions from baseline
    baseline_dir = os.path.join(PATHS["raw_results_dir"], "baseline")
    all_results = {}

    for model_key in MODEL_REGISTRY:
        config = MODEL_REGISTRY[model_key]
        pred_path = os.path.join(baseline_dir, f"predictions_{model_key}.json")
        if not os.path.exists(pred_path):
            logger.warning(f"No predictions for {model_key}")
            continue

        with open(pred_path) as f:
            pred_data = json.load(f)

        model_results = {}
        for bench_key, predictions in pred_data.get("predictions", {}).items():
            if not predictions:
                continue
            bin_results = {bin_name: {"correct": 0, "total": 0, "steps": []}
                          for bin_name in COT_LENGTH_BINS}

            for pred in predictions:
                num_steps = int(pred.get("num_steps", 0) or 0)
                is_correct = pred.get("is_correct", False)

                for bin_name, (lo, hi) in COT_LENGTH_BINS.items():
                    if lo <= num_steps < hi:
                        bin_results[bin_name]["total"] += 1
                        bin_results[bin_name]["steps"].append(num_steps)
                        if is_correct:
                            bin_results[bin_name]["correct"] += 1
                        break

            # Compute stats per bin
            for bin_name in bin_results:
                br = bin_results[bin_name]
                br["accuracy"] = br["correct"] / max(1, br["total"]) * 100
                br["mean_steps"] = float(np.mean(br["steps"])) if br["steps"] else 0
                br["std_steps"] = float(np.std(br["steps"])) if br["steps"] else 0
                del br["steps"]

            model_results[bench_key] = bin_results

        save_path = os.path.join(output_dir, f"cot_length_{model_key}.json")
        with open(save_path, "w") as f:
            json.dump({"model": config.short_name, "results": model_results}, f, indent=2)

        all_results[model_key] = model_results

    agg_path = os.path.join(output_dir, "cot_length_all.json")
    with open(agg_path, "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results


if __name__ == "__main__":
    run_cot_length_ablation()
