#scripts/visualization/plot_radar_charts.py
"""
Radar Chart Visualizations.

Generates radar/spider charts showing failure mode distribution per model.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from configs.model_config import MODEL_REGISTRY
from configs.experiment_config import PATHS, ensure_dirs
from src.utils.logger import setup_logger

logger = setup_logger("plot_radar")

# Failure category display names
FAILURE_DISPLAY = {
    "F1_post_hoc_rationalization": "Post-hoc\nRationalization",
    "F2_invalid_reasoning": "Invalid\nReasoning",
    "F3_redundant_exploration": "Redundant\nExploration",
    "F4_incorrect_backtracking": "Incorrect\nBacktracking",
    "F5_distribution_brittleness": "Distribution\nBrittleness",
    "F6_hallucinated_conclusions": "Hallucinated\nConclusions",
}


def plot_failure_radar(output_dir: str = None):
    """Plot radar chart of failure mode distribution per model."""
    if output_dir is None:
        output_dir = PATHS["figures_dir"]
    os.makedirs(output_dir, exist_ok=True)

    failure_dir = os.path.join(PATHS["raw_results_dir"], "failure_taxonomy")
    agg_path = os.path.join(failure_dir, "failure_all_models.json")

    if not os.path.exists(agg_path):
        logger.warning("No failure taxonomy results found.")
        return

    with open(agg_path) as f:
        data = json.load(f)

    categories = list(FAILURE_DISPLAY.keys())
    cat_labels = list(FAILURE_DISPLAY.values())
    num_vars = len(categories)

    # Angles for radar chart
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    # Colors for each model
    colors = plt.cm.Set2(np.linspace(0, 1, len(MODEL_REGISTRY)))

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

    for idx, (model_key, model_config) in enumerate(MODEL_REGISTRY.items()):
        if model_key not in data:
            continue

        model_data = data[model_key]

        # Average failure rates across benchmarks
        avg_rates = []
        for cat in categories:
            rates = []
            for bench_data in model_data.values():
                fr = bench_data.get("failure_rates", {})
                rates.append(fr.get(cat, 0.0))
            avg_rates.append(float(np.mean(rates)) if rates else 0.0)

        values = avg_rates + avg_rates[:1]

        ax.plot(
            angles, values,
            linewidth=2,
            linestyle="solid",
            label=model_config.short_name,
            color=colors[idx],
        )
        ax.fill(angles, values, alpha=0.1, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cat_labels, size=9)
    ax.set_ylim(0, 1)
    ax.set_title(
        "Failure Mode Distribution Across Models\n(Avg. Across Benchmarks)",
        size=14,
        pad=20,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)

    save_path = os.path.join(output_dir, "radar_failure_modes.png")
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Failure radar chart saved to {save_path}")


def plot_step_type_radar(output_dir: str = None):
    """Plot radar chart of step type distributions (faithful/decorative/shortcut/irrelevant)."""
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

    step_categories = ["faithful", "decorative", "shortcut", "irrelevant"]
    num_vars = len(step_categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    colors = plt.cm.tab10(np.linspace(0, 1, len(MODEL_REGISTRY)))

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))

    for idx, (model_key, model_config) in enumerate(MODEL_REGISTRY.items()):
        if model_key not in data:
            continue

        model_data = data[model_key]
        avg_ratios = []
        for cat in step_categories:
            ratios = []
            for bench_data in model_data.values():
                r = (
                    bench_data
                    .get("rfi_aggregate", {})
                    .get("mean_category_ratios", {})
                    .get(cat, 0.0)
                )
                ratios.append(r)
            avg_ratios.append(float(np.mean(ratios)) if ratios else 0.0)

        values = avg_ratios + avg_ratios[:1]

        ax.plot(angles, values, linewidth=2, label=model_config.short_name, color=colors[idx])
        ax.fill(angles, values, alpha=0.1, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([c.capitalize() for c in step_categories], size=11)
    ax.set_ylim(0, 1)
    ax.set_title("Step Type Distribution per Model", size=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)

    save_path = os.path.join(output_dir, "radar_step_types.png")
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Step type radar chart saved to {save_path}")


if __name__ == "__main__":
    ensure_dirs()
    plot_failure_radar()
    plot_step_type_radar()
