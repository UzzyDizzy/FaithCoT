#scripts/visualization/plot_heatmaps.py
"""
Heatmap Visualizations.

Generates heatmaps for Model × Benchmark faithfulness scores.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from configs.model_config import MODEL_REGISTRY
from configs.benchmark_config import BENCHMARK_REGISTRY
from configs.experiment_config import PATHS, ensure_dirs
from src.utils.logger import setup_logger

logger = setup_logger("plot_heatmaps")

# Use a clean, publication-quality style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


def plot_accuracy_heatmap(output_dir: str = None):
    """Plot Model × Benchmark accuracy heatmap."""
    if output_dir is None:
        output_dir = PATHS["figures_dir"]
    os.makedirs(output_dir, exist_ok=True)

    baseline_dir = os.path.join(PATHS["raw_results_dir"], "baseline")
    agg_path = os.path.join(baseline_dir, "baseline_all_models.json")

    if not os.path.exists(agg_path):
        logger.warning("No baseline results found.")
        return

    with open(agg_path) as f:
        data = json.load(f)

    model_names = []
    bench_names = list(BENCHMARK_REGISTRY.keys())
    matrix = []

    for model_key in MODEL_REGISTRY:
        if model_key in data and "error" not in data[model_key]:
            model_names.append(MODEL_REGISTRY[model_key].short_name)
            row = []
            for bench_key in bench_names:
                acc = data[model_key].get(bench_key, {}).get("accuracy", 0.0)
                row.append(acc)
            matrix.append(row)

    if not matrix:
        logger.warning("No data to plot")
        return

    matrix = np.array(matrix)
    bench_display = [BENCHMARK_REGISTRY[b].name for b in bench_names]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        xticklabels=bench_display,
        yticklabels=model_names,
        ax=ax,
        vmin=0,
        vmax=100,
        linewidths=0.5,
        cbar_kws={"label": "Accuracy (%)"},
    )
    ax.set_title("Baseline Accuracy: Model × Benchmark")
    ax.set_xlabel("Benchmark")
    ax.set_ylabel("Model")

    save_path = os.path.join(output_dir, "heatmap_accuracy.png")
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Accuracy heatmap saved to {save_path}")


def plot_faithfulness_heatmap(output_dir: str = None):
    """Plot Model × Benchmark RFI (faithfulness) heatmap."""
    if output_dir is None:
        output_dir = PATHS["figures_dir"]
    os.makedirs(output_dir, exist_ok=True)

    faith_dir = os.path.join(PATHS["raw_results_dir"], "faithfulness")
    agg_path = os.path.join(faith_dir, "faithfulness_all_models.json")

    if not os.path.exists(agg_path):
        logger.warning("No faithfulness results found.")
        return

    with open(agg_path) as f:
        data = json.load(f)

    model_names = []
    bench_names = list(BENCHMARK_REGISTRY.keys())
    matrix = []

    for model_key in MODEL_REGISTRY:
        if model_key in data:
            model_names.append(MODEL_REGISTRY[model_key].short_name)
            row = []
            for bench_key in bench_names:
                rfi = (
                    data[model_key]
                    .get(bench_key, {})
                    .get("rfi_aggregate", {})
                    .get("mean_rfi", 0.0)
                )
                row.append(rfi)
            matrix.append(row)

    if not matrix:
        logger.warning("No data to plot")
        return

    matrix = np.array(matrix)
    bench_display = [BENCHMARK_REGISTRY[b].name for b in bench_names]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        xticklabels=bench_display,
        yticklabels=model_names,
        ax=ax,
        vmin=0,
        vmax=1,
        linewidths=0.5,
        cbar_kws={"label": "RFI (0=unfaithful, 1=faithful)"},
    )
    ax.set_title("Reasoning Fidelity Index: Model × Benchmark")
    ax.set_xlabel("Benchmark")
    ax.set_ylabel("Model")

    save_path = os.path.join(output_dir, "heatmap_faithfulness.png")
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Faithfulness heatmap saved to {save_path}")


def plot_perturbation_heatmap(output_dir: str = None):
    """Plot perturbation test results heatmap."""
    if output_dir is None:
        output_dir = PATHS["figures_dir"]
    os.makedirs(output_dir, exist_ok=True)

    pert_dir = os.path.join(PATHS["raw_results_dir"], "perturbation")
    agg_path = os.path.join(pert_dir, "perturbation_all_models.json")

    if not os.path.exists(agg_path):
        logger.warning("No perturbation results found.")
        return

    with open(agg_path) as f:
        data = json.load(f)

    test_names = [
        "early_answering", "mistake_injection",
        "step_shuffling", "step_deletion", "paraphrasing"
    ]
    test_display = [
        "Early\nAnswering", "Mistake\nInjection",
        "Step\nShuffling", "Step\nDeletion", "Paraphrasing"
    ]
    model_names = []
    matrix = []

    for model_key in MODEL_REGISTRY:
        if model_key not in data:
            continue
        model_names.append(MODEL_REGISTRY[model_key].short_name)
        row = []
        benches = data[model_key]

        for test_name in test_names:
            scores = []
            for bench_data in benches.values():
                test_data = bench_data.get(test_name, {})
                if test_name == "early_answering":
                    scores.append(test_data.get("post_hoc_ratio", 0))
                elif test_name == "mistake_injection":
                    scores.append(test_data.get("mean_ignores_ratio", 0))
                elif test_name == "step_shuffling":
                    scores.append(test_data.get("mean_shuffle_robustness", 0))
                elif test_name == "step_deletion":
                    scores.append(test_data.get("mean_causal_step_ratio", 0))
                elif test_name == "paraphrasing":
                    scores.append(test_data.get("surface_sensitive_ratio", 0))
            row.append(float(np.mean(scores)) if scores else 0.0)
        matrix.append(row)

    if not matrix:
        return

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        xticklabels=test_display,
        yticklabels=model_names,
        ax=ax,
        vmin=0,
        vmax=1,
        linewidths=0.5,
    )
    ax.set_title("Perturbation Test Results (Avg. Across Benchmarks)")
    ax.set_xlabel("Perturbation Test")
    ax.set_ylabel("Model")

    save_path = os.path.join(output_dir, "heatmap_perturbation.png")
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Perturbation heatmap saved to {save_path}")


if __name__ == "__main__":
    ensure_dirs()
    plot_accuracy_heatmap()
    plot_faithfulness_heatmap()
    plot_perturbation_heatmap()
