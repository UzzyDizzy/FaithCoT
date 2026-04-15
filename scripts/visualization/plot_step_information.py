#scripts/visualization/plot_step_information.py
"""
Step-Level Information Gain Plots.

Visualizes per-step SIG values to show where reasoning adds value.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from configs.experiment_config import PATHS, ensure_dirs
from src.utils.logger import setup_logger

logger = setup_logger("plot_step_info")


def plot_step_information_profile(
    sig_values: list,
    step_labels: list = None,
    title: str = "Step Information Gain Profile",
    output_path: str = None,
):
    """Plot SIG values for a single CoT example.

    Args:
        sig_values: List of SIG values per step
        step_labels: Optional labels per step
        title: Plot title
        output_path: Where to save the figure
    """
    n = len(sig_values)
    if n == 0:
        return

    fig, ax = plt.subplots(figsize=(max(8, n * 0.8), 5))
    x = np.arange(n)

    colors = []
    for val in sig_values:
        if val > 0.05:
            colors.append("#4CAF50")  # High info: green
        elif val > 0.01:
            colors.append("#FFC107")  # Moderate: yellow
        else:
            colors.append("#F44336")  # Low info: red

    bars = ax.bar(x, sig_values, color=colors, edgecolor="white", linewidth=0.5)

    # Add threshold line
    ax.axhline(y=0.01, color="gray", linestyle="--", alpha=0.7, label="SIG threshold (τ=0.01)")

    ax.set_xlabel("Reasoning Step")
    ax.set_ylabel("Step Information Gain (SIG)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{i+1}" for i in range(n)], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path)
        logger.info(f"Step info plot saved to {output_path}")
    plt.close(fig)


def plot_aggregate_step_info(output_dir: str = None):
    """Plot aggregate step information gain across examples."""
    if output_dir is None:
        output_dir = PATHS["figures_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Create a synthetic example for illustration
    # In practice, this reads from faithfulness results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Example 1: Faithful reasoning
    sig_faithful = [0.02, 0.05, 0.08, 0.12, 0.15, 0.20]
    ax = axes[0, 0]
    x = np.arange(len(sig_faithful))
    colors = ["#4CAF50" if v > 0.01 else "#F44336" for v in sig_faithful]
    ax.bar(x, sig_faithful, color=colors)
    ax.axhline(y=0.01, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Faithful Reasoning (All steps informative)")
    ax.set_xlabel("Step")
    ax.set_ylabel("SIG")

    # Example 2: Post-hoc rationalization
    sig_posthoc = [0.001, 0.002, 0.001, 0.003, 0.001, 0.002]
    ax = axes[0, 1]
    x = np.arange(len(sig_posthoc))
    colors = ["#4CAF50" if v > 0.01 else "#F44336" for v in sig_posthoc]
    ax.bar(x, sig_posthoc, color=colors)
    ax.axhline(y=0.01, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Post-hoc Rationalization (No step informative)")
    ax.set_xlabel("Step")
    ax.set_ylabel("SIG")

    # Example 3: Partially faithful
    sig_partial = [0.001, 0.03, 0.001, 0.08, 0.002, 0.15]
    ax = axes[1, 0]
    x = np.arange(len(sig_partial))
    colors = ["#4CAF50" if v > 0.01 else "#F44336" for v in sig_partial]
    ax.bar(x, sig_partial, color=colors)
    ax.axhline(y=0.01, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Partially Faithful (Mixed informative/noise)")
    ax.set_xlabel("Step")
    ax.set_ylabel("SIG")

    # Example 4: Information front-loaded
    sig_front = [0.15, 0.10, 0.05, 0.002, 0.001, 0.001]
    ax = axes[1, 1]
    x = np.arange(len(sig_front))
    colors = ["#4CAF50" if v > 0.01 else "#F44336" for v in sig_front]
    ax.bar(x, sig_front, color=colors)
    ax.axhline(y=0.01, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Front-loaded (Early steps carry all info)")
    ax.set_xlabel("Step")
    ax.set_ylabel("SIG")

    plt.suptitle("Step Information Gain (SIG) Profiles — Exemplar Patterns", fontsize=14)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "step_information_profiles.png")
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Step information profiles saved to {save_path}")


if __name__ == "__main__":
    ensure_dirs()
    plot_aggregate_step_info()
