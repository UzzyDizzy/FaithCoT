#scripts/ablations/ablation_model_scaling.py
"""
Ablation 5: Model Size Scaling Analysis.

Studies how faithfulness metrics scale with model size (7B -> 8B -> 14B -> 32B).
Tests the inverse scaling hypothesis with regression analysis.
"""

import os, sys, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from configs.model_config import MODEL_REGISTRY
from configs.experiment_config import PATHS, ensure_dirs
from src.utils.logger import setup_logger

logger = setup_logger("ablation_scaling")


def run_model_scaling_analysis():
    """Analyze faithfulness as a function of model parameter count."""
    ensure_dirs()
    output_dir = os.path.join(PATHS["raw_results_dir"], "ablations", "model_scaling")
    os.makedirs(output_dir, exist_ok=True)

    # Collect data from previous experiments
    baseline_dir = os.path.join(PATHS["raw_results_dir"], "baseline")
    faithfulness_dir = os.path.join(PATHS["raw_results_dir"], "faithfulness")
    perturbation_dir = os.path.join(PATHS["raw_results_dir"], "perturbation")

    scaling_data = []

    for model_key, config in sorted(
        MODEL_REGISTRY.items(), key=lambda x: x[1].params_b
    ):
        entry = {
            "model_key": model_key,
            "model_name": config.short_name,
            "params_b": config.params_b,
            "family": config.family,
        }

        # Load baseline
        base_path = os.path.join(baseline_dir, f"baseline_{model_key}.json")
        if os.path.exists(base_path):
            with open(base_path) as f:
                base = json.load(f)
            accs = [
                v.get("accuracy", 0)
                for v in base.get("results", {}).values()
            ]
            entry["mean_accuracy"] = float(np.mean(accs)) if accs else 0.0

        # Load faithfulness
        faith_path = os.path.join(faithfulness_dir, f"faithfulness_{model_key}.json")
        if os.path.exists(faith_path):
            with open(faith_path) as f:
                faith = json.load(f)
            rfis = []
            for bench_data in faith.get("results", {}).values():
                rfi = bench_data.get("rfi_aggregate", {}).get("mean_rfi", 0)
                rfis.append(rfi)
            entry["mean_rfi"] = float(np.mean(rfis)) if rfis else 0.0

        # Load perturbation
        pert_path = os.path.join(perturbation_dir, f"perturbation_{model_key}.json")
        if os.path.exists(pert_path):
            with open(pert_path) as f:
                pert = json.load(f)
            post_hoc_rates = []
            for bench_data in pert.get("results", {}).values():
                ea = bench_data.get("early_answering", {})
                post_hoc_rates.append(ea.get("post_hoc_ratio", 0))
            entry["mean_post_hoc_rate"] = float(np.mean(post_hoc_rates)) if post_hoc_rates else 0.0

        scaling_data.append(entry)

    # Regression analysis
    params = np.array([d["params_b"] for d in scaling_data if d.get("params_b", 0) > 0])

    if len(params) == 0:
        logger.warning("No valid params for scaling analysis")
        return {}

    log_params = np.log(params)

    regressions = {}

    # Accuracy vs. parameters
    acc_pairs = [(d["params_b"], d["mean_accuracy"]) 
             for d in scaling_data if "mean_accuracy" in d]

    if len(acc_pairs) >= 3:
        params_acc = np.array([p for p, _ in acc_pairs])
        log_params_acc = np.log(params_acc)
        accs = np.array([a for _, a in acc_pairs])

        if np.std(accs) > 1e-6:
            corr = float(np.corrcoef(log_params_acc, accs)[0, 1])
            if np.isnan(corr):
                corr = 0.0

            regressions["accuracy_vs_logparams"] = {
                "correlation": corr,
                "trend": "positive" if corr > 0.3 else "negative" if corr < -0.3 else "neutral",
            }

    # RFI vs. parameters (inverse scaling test)
    rfi_pairs = [(d["params_b"], d["mean_rfi"]) 
             for d in scaling_data if "mean_rfi" in d]

    if len(rfi_pairs) >= 3:
        params_rfi = np.array([p for p, _ in rfi_pairs])
        log_params_rfi = np.log(params_rfi)
        rfis = np.array([r for _, r in rfi_pairs])

        if np.std(rfis) > 1e-6:
            corr = float(np.corrcoef(log_params_rfi, rfis)[0, 1])
            if np.isnan(corr):
                corr = 0.0

            regressions["rfi_vs_logparams"] = {
                "correlation": corr,
                "trend": "positive" if corr > 0.3 else "negative" if corr < -0.3 else "neutral",
                "inverse_scaling": corr < -0.3,
            }

    results = {
        "scaling_data": scaling_data,
        "regressions": regressions,
        "summary": {
            "num_models": len(scaling_data),
            "param_range": f"{min(params):.0f}B - {max(params):.0f}B" if len(params) > 0 else "N/A",
        },
    }

    save_path = os.path.join(output_dir, "model_scaling_analysis.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Scaling analysis saved to {save_path}")

    return results


if __name__ == "__main__":
    run_model_scaling_analysis()
