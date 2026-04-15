#scripts/experiments/exp_cross_model_analysis.py
"""
Experiment 5: Cross-Model Analysis.

Compares faithfulness metrics across model families and sizes.
Tests the inverse scaling hypothesis (larger models = less faithful).
"""

import os, sys, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from configs.model_config import MODEL_REGISTRY
from configs.experiment_config import PATHS, ensure_dirs
from src.utils.logger import setup_logger

logger = setup_logger("exp_cross_model")


def run_cross_model_analysis():
    """Analyze faithfulness trends across model sizes and families."""
    ensure_dirs()
    output_dir = os.path.join(PATHS["raw_results_dir"], "cross_model")
    os.makedirs(output_dir, exist_ok=True)

    # Load previous experiment results
    faithfulness_dir = os.path.join(PATHS["raw_results_dir"], "faithfulness")
    baseline_dir = os.path.join(PATHS["raw_results_dir"], "baseline")
    perturbation_dir = os.path.join(PATHS["raw_results_dir"], "perturbation")

    # Collect data points for analysis
    model_data = []
    for model_key, config in MODEL_REGISTRY.items():
        entry = {
            "model_key": model_key,
            "model_name": config.short_name,
            "params_b": config.params_b,
            "family": config.family,
        }

        # Load baseline accuracy
        base_path = os.path.join(baseline_dir, f"baseline_{model_key}.json")
        if os.path.exists(base_path):
            with open(base_path) as f:
                base_data = json.load(f)
            entry["baseline"] = base_data.get("results", {})

        # Load faithfulness metrics
        faith_path = os.path.join(faithfulness_dir, f"faithfulness_{model_key}.json")
        if os.path.exists(faith_path):
            with open(faith_path) as f:
                faith_data = json.load(f)
            entry["faithfulness"] = faith_data.get("results", {})

        # Load perturbation results
        pert_path = os.path.join(perturbation_dir, f"perturbation_{model_key}.json")
        if os.path.exists(pert_path):
            with open(pert_path) as f:
                pert_data = json.load(f)
            entry["perturbation"] = pert_data.get("results", {})

        model_data.append(entry)

    # Sort by parameter count
    model_data.sort(key=lambda x: x["params_b"])

    # Analysis 1: Accuracy vs. Faithfulness scaling
    scaling_analysis = []
    for entry in model_data:
        accuracies = []
        rfi_values = []
        for bench_key, bench_data in entry.get("baseline", {}).items():
            acc = bench_data.get("accuracy", 0)
            accuracies.append(acc)
        for bench_key, bench_data in entry.get("faithfulness", {}).items():
            rfi = bench_data.get("rfi_aggregate", {}).get("mean_rfi", 0)
            rfi_values.append(rfi)

        scaling_analysis.append({
            "model": entry["model_name"],
            "params_b": entry["params_b"],
            "mean_accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
            "mean_rfi": float(np.mean(rfi_values)) if rfi_values else 0.0,
        })

    # Analysis 2: Inverse scaling test
    params = [s["params_b"] for s in scaling_analysis]
    rfis = [s["mean_rfi"] for s in scaling_analysis]
    if len(params) >= 3:
        if len(params) >= 3 and np.std(rfis) > 1e-6:
            correlation = float(np.corrcoef(params, rfis)[0, 1])
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0
        inverse_scaling = correlation < -0.3
    else:
        correlation = 0.0
        inverse_scaling = False

    # Analysis 3: Family comparison
    family_groups = {}
    for entry in model_data:
        family = entry["family"]
        if family not in family_groups:
            family_groups[family] = []
        family_groups[family].append(entry)

    results = {
        "scaling_analysis": scaling_analysis,
        "inverse_scaling_test": {
            "correlation": correlation,
            "inverse_scaling_detected": inverse_scaling,
            "interpretation": (
                "Larger models show LESS faithful reasoning"
                if inverse_scaling
                else "No clear inverse scaling pattern detected"
            ),
        },
        "family_comparison": {
            family: {
                "num_models": len(models),
                "model_names": [m["model_name"] for m in models],
            }
            for family, models in family_groups.items()
        },
        "raw_data": model_data,
    }

    save_path = os.path.join(output_dir, "cross_model_analysis.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Cross-model analysis saved to {save_path}")

    return results


if __name__ == "__main__":
    run_cross_model_analysis()
