#configs/experiment_config.py
"""
Experiment-level configuration for FaithCoT.

Global settings: paths, seeds, hardware config, precision, metric thresholds.
"""

import os
from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================

# Auto-detect project root (directory containing this file's parent)
_CONFIG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _CONFIG_DIR.parent

PATHS = {
    "project_root": str(PROJECT_ROOT),
    "data_dir": str(PROJECT_ROOT / "data"),
    "raw_data_dir": str(PROJECT_ROOT / "data" / "raw"),
    "processed_data_dir": str(PROJECT_ROOT / "data" / "processed"),
    "results_dir": str(PROJECT_ROOT / "results"),
    "tables_dir": str(PROJECT_ROOT / "results" / "tables"),
    "figures_dir": str(PROJECT_ROOT / "results" / "figures"),
    "raw_results_dir": str(PROJECT_ROOT / "results" / "raw"),
    "cache_dir": str(PROJECT_ROOT / "cache"),
    "model_cache_dir": str(PROJECT_ROOT / "cache" / "models"),
    "logs_dir": str(PROJECT_ROOT / "logs"),
}


def ensure_dirs():
    """Create all required directories if they don't exist."""
    for path in PATHS.values():
        os.makedirs(path, exist_ok=True)


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

EXPERIMENT_CONFIG = {
    # Reproducibility
    "seed": 42,
    "deterministic": True,

    # Hardware
    "device": "cuda",  # "cuda" or "cpu"
    "num_workers": 32,  # CPU workers for data loading (out of 46 CPUs)
    "pin_memory": True,

    # Precision (Global)
    "dtype": "bfloat16",  # float16 for all models on 96GB VRAM
    "use_amp": True,     # Automatic Mixed Precision

    # Inference
    "max_reasoning_tokens": 2048,
    "max_answer_tokens": 256,

    # Subsampling
    "subsample_size": 200,       # Samples per benchmark for main experiments
    "full_eval_size": None,       # None = full test set

    # Metric thresholds
    "sig_threshold": 0.01,        # Minimum SIG to count as informative step
    "cns_threshold": 0.05,        # Minimum CNS to count as causally necessary
    "rfi_threshold": 0.3,         # Below this = unfaithful reasoning

    # Perturbation settings
    "num_perturbation_samples": 10,  # Samples for perturbation tests
    "mistake_types": ["arithmetic", "logical", "factual"],

    # Ablation settings
    "temperature_values": [0.0, 0.3, 0.6, 1.0],
    "cot_length_bins": {
        "short": (0, 5),
        "medium": (5, 15),
        "long": (15, float("inf")),
    },
    "prompt_formats": ["zero_shot_cot", "few_shot_cot", "explicit_steps"],

    # Logging
    "log_level": "INFO",
    "save_intermediate": True,
    "verbose": True,
}


# ============================================================
# FAILURE TAXONOMY CATEGORIES
# ============================================================

FAILURE_CATEGORIES = {
    "F1_post_hoc_rationalization": {
        "name": "Post-hoc Rationalization",
        "description": "CoT doesn't causally drive answer; model produces same answer regardless",
        "detection": "early_answering",
    },
    "F2_invalid_reasoning": {
        "name": "Invalid Reasoning Steps",
        "description": "Logical errors in reasoning chain",
        "detection": "step_validation",
    },
    "F3_redundant_exploration": {
        "name": "Redundant Exploration",
        "description": "Repeating same ineffective reasoning paths",
        "detection": "repetition_detection",
    },
    "F4_incorrect_backtracking": {
        "name": "Incorrect Backtracking",
        "description": "Restoring inconsistent state during revision",
        "detection": "state_consistency",
    },
    "F5_distribution_brittleness": {
        "name": "Distribution-Dependent Brittleness",
        "description": "CoT breaks under OOD inputs",
        "detection": "ood_accuracy_delta",
    },
    "F6_hallucinated_conclusions": {
        "name": "Hallucinated Conclusions",
        "description": "Conclusions unsupported by reasoning steps",
        "detection": "entailment_verification",
    },
}
