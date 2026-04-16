#model_config.py
"""
Model configuration for FaithCoT experiments.

Defines all 5 open-source reasoning models (ascending parameter count),
generation hyperparameters, and hardware-optimized batch sizes for
NVIDIA RTX 6000 Pro (96GB VRAM, 46 CPUs, 500 TFLOPS).
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    model_id: str
    short_name: str
    family: str
    params_b: float  # Billions of parameters
    batch_size: int  # Optimized for 96GB VRAM in fp16
    max_new_tokens: int = 2048
    dtype: str = "bfloat16"  # Global precision config
    trust_remote_code: bool = True
    device_map: str = "auto"
    attn_implementation: Optional[str] = None  # "flash_attention_2" if available


# ============================================================
# MODEL REGISTRY: 5 Open-Source Reasoning Models (Ascending Size)
# ============================================================
# Uncomment/comment models as needed for individual runs.
# For ablations/experiments using multiple models, all are used.

# --- Model 1: DeepSeek-R1-Distill-Qwen-7B (7B) ---
MODEL_1 = ModelConfig(
    model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    short_name="DS-R1-Qwen-7B",
    family="deepseek-r1",
    params_b=7.0,
    batch_size=32,
    device_map="auto",
    attn_implementation="flash_attention_2",
)

# --- Model 2: DeepSeek-R1-Distill-Llama-8B (8B) ---
MODEL_2 = ModelConfig(
    model_id="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    short_name="DS-R1-Llama-8B",
    family="llama",
    params_b=8.0,
    batch_size=32,
    device_map="auto",
    attn_implementation="flash_attention_2",
)

# --- Model 3: DeepSeek-R1-Distill-Qwen-14B (14B) ---
MODEL_3 = ModelConfig(
    model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    short_name="DS-R1-Qwen-14B",
    family="deepseek-r1",
    params_b=14.0,
    batch_size=16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)

# --- Model 4: Qwen-QwQ-32B (32B) ---
MODEL_4 = ModelConfig(
    model_id="Qwen/QwQ-32B",
    short_name="QwQ-32B",
    family="qwen-qwq",
    params_b=32.0,
    batch_size=8,
    device_map="auto",
    attn_implementation="flash_attention_2",
)

# --- Model 5: DeepSeek-R1-Distill-Qwen-32B (32B) ---
MODEL_5 = ModelConfig(
    model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    short_name="DS-R1-Qwen-32B",
    family="deepseek-r1",
    params_b=32.0,
    batch_size=8,
    device_map="auto",
    attn_implementation="flash_attention_2",
)

# All models in ascending parameter order
MODEL_REGISTRY: Dict[str, ModelConfig] = {
    "ds-r1-qwen-7b": MODEL_1,
    "ds-r1-llama-8b": MODEL_2,
    "ds-r1-qwen-14b": MODEL_3,
    "qwq-32b": MODEL_4,
    "ds-r1-qwen-32b": MODEL_5,
}

# Current active model (change for single-model runs)
MODEL = MODEL_1  # Default: smallest model for testing

# ============================================================
# GENERATION CONFIGURATION (Shared across all models)
# ============================================================

GENERATION_CONFIG = {
    "max_new_tokens": 256,       # Max reasoning tokens
    "temperature": 0.0,          # Deterministic greedy decoding
    "top_p": 1.0,                # No nucleus sampling
    "top_k": 1,                  # Greedy
    "do_sample": False,          # Greedy decoding
    "repetition_penalty": 1.0,   # No repetition penalty
    "num_return_sequences": 1,
    "pad_token_id": None,        # Set dynamically per model
    "eos_token_id": None,        # Set dynamically per model
}

# Answer extraction generation config (shorter, still deterministic)
ANSWER_EXTRACTION_CONFIG = {
    "max_new_tokens": 256,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 1,
    "do_sample": False,
    "num_return_sequences": 1,
}

# ============================================================
# API MODEL CONFIGURATION (Optional - for comparison)
# ============================================================

API_MODELS = {
    "deepseek-chat": {
        "provider": "deepseek",
        "model_name": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
    },
    "deepseek-reasoner": {
        "provider": "deepseek",
        "model_name": "deepseek-reasoner",
        "base_url": "https://api.deepseek.com",
    },
    "gpt-4o": {
        "provider": "openai",
        "model_name": "gpt-4o",
        "base_url": None,  # Uses default OpenAI base URL
    },
    "o1-mini": {
        "provider": "openai",
        "model_name": "o1-mini",
        "base_url": None,
    },
}


def get_model_config(model_key: str) -> ModelConfig:
    """Get model configuration by key.

    Args:
        model_key: Key from MODEL_REGISTRY (e.g., 'ds-r1-qwen-7b')

    Returns:
        ModelConfig instance

    Raises:
        KeyError: If model_key not found in registry
    """
    if model_key not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise KeyError(
            f"Model '{model_key}' not found. Available: {available}"
        )
    return MODEL_REGISTRY[model_key]
