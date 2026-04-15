#src/models/model_loader.py
"""
Model Loader.

Loads HuggingFace models with fp16 precision and AMP support.
Optimized for NVIDIA RTX 6000 Pro (96GB VRAM).
"""

import gc
import torch
from typing import Optional, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer

from configs.model_config import ModelConfig, GENERATION_CONFIG
from src.utils.logger import get_logger

logger = get_logger("model_loader")


class ModelManager:
    """Manages loading and unloading of HuggingFace models."""

    def __init__(self, cache_dir: Optional[str] = None, hf_token: Optional[str] = None):
        """Initialize model manager.

        Args:
            cache_dir: Model cache directory
            hf_token: HuggingFace token for gated models
        """
        self.cache_dir = cache_dir
        self.hf_token = hf_token
        self.current_model = None
        self.current_tokenizer = None
        self.current_config = None

    def load_model(
        self, config: ModelConfig
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load a model and tokenizer.

        Args:
            config: ModelConfig instance

        Returns:
            Tuple of (model, tokenizer)
        """
        # Unload current model if different
        if self.current_config and self.current_config.model_id != config.model_id:
            self.unload_model()

        if self.current_model is not None:
            logger.info(f"Model {config.short_name} already loaded")
            return self.current_model, self.current_tokenizer

        logger.info(
            f"Loading model {config.short_name} ({config.model_id}) "
            f"in {config.dtype}..."
        )

        # Determine torch dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(config.dtype, torch.float16)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id,
            trust_remote_code=config.trust_remote_code,
            cache_dir=self.cache_dir,
            token=self.hf_token,
        )

        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Load model
        model_kwargs = {
            "pretrained_model_name_or_path": config.model_id,
            "torch_dtype": torch_dtype,
            "device_map": config.device_map,
            "trust_remote_code": config.trust_remote_code,
            "cache_dir": self.cache_dir,
            "token": self.hf_token,
        }

        # Add attention implementation if specified
        if config.attn_implementation:
            model_kwargs["attn_implementation"] = config.attn_implementation

        model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        model.eval()

        # Log memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(
                f"GPU Memory: {allocated:.1f}GB allocated, "
                f"{reserved:.1f}GB reserved"
            )

        self.current_model = model
        self.current_tokenizer = tokenizer
        self.current_config = config

        logger.info(f"Successfully loaded {config.short_name}")
        return model, tokenizer

    def unload_model(self) -> None:
        """Unload the current model and free GPU memory."""
        if self.current_model is not None:
            model_name = self.current_config.short_name if self.current_config else "unknown"
            logger.info(f"Unloading model {model_name}...")

            del self.current_model
            del self.current_tokenizer
            self.current_model = None
            self.current_tokenizer = None
            self.current_config = None

            # Force garbage collection and CUDA memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            logger.info("Model unloaded and memory freed")

    def get_generation_config(self) -> dict:
        """Get generation config with model-specific token IDs.

        Returns:
            Generation config dict with pad/eos token IDs set
        """
        config = dict(GENERATION_CONFIG)
        if self.current_tokenizer:
            config["pad_token_id"] = self.current_tokenizer.pad_token_id
            config["eos_token_id"] = self.current_tokenizer.eos_token_id
        return config

    @property
    def is_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self.current_model is not None
