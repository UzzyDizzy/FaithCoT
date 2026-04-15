#scripts/visualization/plot_scaling_curves.py
"""
Scaling Curve Visualizations.

Plots faithfulness and accuracy as a function of model size.
Tests and visualizes the inverse scaling hypothesis.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from configs.model_config import MODEL_REGISTRY
from configs.experiment_config import PATHS, ensure_dirs
from src.utils.logger import setup_logger

logger = setup_logger("plot_scaling")


def plot_scaling_curves(output_dir: str = None):
    """Plot accuracy and faithfulness vs. model size."""
    if output_dir is None:
        output_dir = PATHS["figures_dir"]
    os.makedirs(output_dir, exist_ok=True)

    scaling_path = os.path.join(
        PATHS["raw_results_dir"], "ablations", "model_scaling",
        "model_scaling_analysis.json"
    )

    # Try cross-model analysis as fallback
    if not os.path.exists(scaling_path):
        scaling_path = os.path.join(
            PATHS["raw_results_dir"], "cross_model",
            "cross_model_analysis.json"
        )

    if not os.path.exists(scaling_path):
        logger.warning("No scaling data found.")
        return

    with open(scaling_path) as f:
        data = json.load(f)

    scaling_data = data.get("scaling_data", data.get("scaling_analysis", []))
    if not scaling_data:
        return

    params = [d["params_b"] for d in scaling_data]
    names = [d.get("model_name", d.get("model", "")) for d in scaling_data]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Accuracy vs. Model Size
    if all("mean_accuracy" in d for d in scaling_data):
        accs = [d["mean_accuracy"] for d in scaling_data]
        ax = axes[0]
        ax.plot(params, accs, "o-", color="#2196F3", linewidth=2, markersize=8)
        for i, name in enumerate(names):
            ax.annotate(
                name, (params[i], accs[i]),
                textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=8,
            )
        ax.set_xlabel("Parameters (Billions)")
        ax.set_ylabel("Mean Accuracy (%)")
        ax.set_title("Accuracy vs. Model Size")
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")

    # Plot 2: Faithfulness (RFI) vs. Model Size
    if all("mean_rfi" in d for d in scaling_data):
        rfis = [d["mean_rfi"] for d in scaling_data]
        ax = axes[1]
        ax.plot(params, rfis, "s-", color="#F44336", linewidth=2, markersize=8)
        for i, name in enumerate(names):
            ax.annotate(
                name, (params[i], rfis[i]),
                textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=8,
            )

        # Add trend line
        if len(params) >= 3:
            log_params = np.log(params)
            z = np.polyfit(log_params, rfis, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(log_params), max(log_params), 50)
            ax.plot(np.exp(x_trend), p(x_trend), "--", color="#F44336", alpha=0.5, label="Trend")

            corr = np.corrcoef(log_params, rfis)[0, 1]
            ax.text(
                0.05, 0.95,
                f"Correlation: {corr:.3f}\n{'Inverse scaling!' if corr < -0.3 else 'No inverse scaling'}",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        ax.set_xlabel("Parameters (Billions)")
        ax.set_ylabel("Mean RFI")
        ax.set_title("Faithfulness vs. Model Size")
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")

    plt.suptitle("Scaling Analysis: Accuracy vs. Faithfulness", fontsize=16, y=1.02)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "scaling_curves.png")
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Scaling curves saved to {save_path}")


def plot_accuracy_faithfulness_scatter(output_dir: str = None):
    """Plot scatter: accuracy vs. faithfulness per model-benchmark pair."""
    if output_dir is None:
        output_dir = PATHS["figures_dir"]
    os.makedirs(output_dir, exist_ok=True)

    cross_path = os.path.join(
        PATHS["raw_results_dir"], "cross_model", "cross_model_analysis.json"
    )
    if not os.path.exists(cross_path):
        return

    with open(cross_path) as f:
        data = json.load(f)

    raw_data = data.get("raw_data", [])
    if not raw_data:
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.Set2(np.linspace(0, 1, len(raw_data)))

    for idx, entry in enumerate(raw_data):
        baseline = entry.get("baseline", {})
        faithfulness = entry.get("faithfulness", {})
        model_name = entry.get("model_name", "")

        for bench_key in baseline:
            acc = baseline[bench_key].get("accuracy", 0)
            rfi = (
                faithfulness.get(bench_key, {})
                .get("rfi_aggregate", {})
                .get("mean_rfi", 0)
            )
            ax.scatter(
                acc, rfi,
                color=colors[idx],
                s=100,
                alpha=0.7,
                label=model_name if bench_key == list(baseline.keys())[0] else "",
            )

    ax.set_xlabel("Accuracy (%)")
    ax.set_ylabel("Reasoning Fidelity Index (RFI)")
    ax.set_title("Accuracy vs. Faithfulness per Model-Benchmark Pair")
    ax.grid(True, alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="best")

    save_path = os.path.join(output_dir, "scatter_accuracy_faithfulness.png")
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Scatter plot saved to {save_path}")


if __name__ == "__main__":
    ensure_dirs()
    plot_scaling_curves()
    plot_accuracy_faithfulness_scatter()
