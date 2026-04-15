#scripts/ablations/ablation_perturbation_type.py
"""
Ablation 3: Perturbation Type Comparison.

Compares the effectiveness of different perturbation types
(early answering, mistake injection, step deletion) for
detecting unfaithful reasoning.
"""

import os, sys, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from configs.model_config import MODEL_REGISTRY
from configs.experiment_config import PATHS, ensure_dirs
from src.utils.logger import setup_logger

logger = setup_logger("ablation_perturbation")


def run_perturbation_type_ablation():
    """Compare perturbation types for detecting unfaithfulness."""
    ensure_dirs()
    output_dir = os.path.join(PATHS["raw_results_dir"], "ablations", "perturbation_type")
    os.makedirs(output_dir, exist_ok=True)

    # Load perturbation test results
    pert_dir = os.path.join(PATHS["raw_results_dir"], "perturbation")
    pert_agg_path = os.path.join(pert_dir, "perturbation_all_models.json")

    if not os.path.exists(pert_agg_path):
        logger.warning("No perturbation results found. Run exp_perturbation_tests first.")
        return {}

    with open(pert_agg_path) as f:
        all_pert = json.load(f)

    # Compare perturbation types across models
    comparison = {}
    for model_key, benchmarks in all_pert.items():
        model_comparison = {}
        for bench_key, tests in benchmarks.items():
            test_scores = {}

            # Extract detection rates
            ea = tests.get("early_answering", {})
            test_scores["early_answering"] = ea.get("post_hoc_ratio", 0.0)

            mi = tests.get("mistake_injection", {})
            test_scores["mistake_injection"] = mi.get("unfaithful_ratio", 0.0)

            ss = tests.get("step_shuffling", {})
            test_scores["step_shuffling"] = max(0.0, 1.0 - ss.get("order_matters_ratio", 1.0))

            sd = tests.get("step_deletion", {})
            test_scores["step_deletion"] = max(0.0, 1.0 - sd.get("mean_causal_step_ratio", 1.0))

            pp = tests.get("paraphrasing", {})
            test_scores["paraphrasing"] = pp.get("surface_sensitive_ratio", 0.0)

            model_comparison[bench_key] = test_scores

        comparison[model_key] = model_comparison

    # Aggregate across benchmarks per model
    summary = {}
    for model_key, benchmarks in comparison.items():
        test_means = {}
        for bench_key, tests in benchmarks.items():
            for test_name, score in tests.items():
                if test_name not in test_means:
                    test_means[test_name] = []
                test_means[test_name].append(score)

        summary[model_key] = {
            test_name: {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
            }
            for test_name, scores in test_means.items()
        }

    results = {
        "per_model_per_benchmark": comparison,
        "summary": summary,
    }

    save_path = os.path.join(output_dir, "perturbation_type_comparison.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_perturbation_type_ablation()
