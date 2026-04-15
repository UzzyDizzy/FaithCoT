#scripts/ablations/ablation_prompt_format.py
"""
Ablation 4: Prompt Format Effect on Faithfulness.

Tests how different prompting formats affect CoT faithfulness:
- Zero-shot CoT ("Let's think step by step")
- Few-shot CoT (with examples)
- Explicit step format ("Step 1: ...")
"""

import os, sys, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from configs.model_config import MODEL_REGISTRY
from configs.benchmark_config import BENCHMARK_REGISTRY
from configs.experiment_config import EXPERIMENT_CONFIG, PATHS, ensure_dirs
from src.data.dataset_loader import DatasetLoader
from src.data.preprocessing import preprocess_dataset
from src.models.model_loader import ModelManager
from src.models.inference import InferenceEngine
from src.utils.answer_extractor import AnswerExtractor
from src.utils.logger import setup_logger

logger = setup_logger("ablation_prompt_format")

PROMPT_FORMATS = EXPERIMENT_CONFIG.get(
    "prompt_formats", ["zero_shot_cot", "few_shot_cot", "explicit_steps"]
)
ABLATION_MODELS = ["ds-r1-qwen-7b", "ds-r1-qwen-14b", "qwq-32b"]


def run_prompt_format_ablation():
    """Run prompt format ablation study."""
    ensure_dirs()
    output_dir = os.path.join(PATHS["raw_results_dir"], "ablations", "prompt_format")
    os.makedirs(output_dir, exist_ok=True)

    model_manager = ModelManager(cache_dir=PATHS.get("model_cache_dir"))
    answer_extractor = AnswerExtractor()
    num_samples = 50

    all_results = {}

    for model_key in ABLATION_MODELS:
        if model_key not in MODEL_REGISTRY:
            continue
        config = MODEL_REGISTRY[model_key]
        logger.info(f"=== Prompt Format Ablation: {config.short_name} ===")

        model, tokenizer = model_manager.load_model(config)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        gen_config = model_manager.get_generation_config()
        engine = InferenceEngine(
            model, tokenizer, gen_config,
            device=EXPERIMENT_CONFIG["device"],
            use_amp=EXPERIMENT_CONFIG["use_amp"],
        )

        model_results = {}
        for fmt in PROMPT_FORMATS:
            logger.info(f"  Format: {fmt}")
            fmt_results = {}

            for bench_key in ["gsm8k", "strategyqa"]:
                bench_config = BENCHMARK_REGISTRY[bench_key]
                loader = DatasetLoader(bench_config)
                data = loader.load()
                data = data.select(range(min(num_samples, len(data))))
                data = preprocess_dataset(data, bench_key, fmt)

                correct = 0
                total_steps = []

                prompts = [d["prompt"] for d in data]

                outputs = engine.generate_batch(
                    prompts,
                    batch_size=min(config.batch_size, len(prompts)),
                    max_new_tokens=config.max_new_tokens
                )

                for d, output in zip(data, outputs):
                    predicted = engine.extract_answer(
                        output["raw_output"], d["answer_type"], bench_key
                    )
                    if answer_extractor.check_answer(
                        predicted, d["gold_answer"], d["answer_type"]
                    ):
                        correct += 1
                    total_steps.append(output["num_steps"])

                fmt_results[bench_key] = {
                    "accuracy": correct / max(1, len(data)) * 100,
                    "mean_steps": float(np.mean(total_steps)),
                    "std_steps": float(np.std(total_steps)),
                }

            model_results[fmt] = fmt_results

        save_path = os.path.join(output_dir, f"prompt_format_{model_key}.json")
        with open(save_path, "w") as f:
            json.dump({"model": config.short_name, "results": model_results}, f, indent=2)

        all_results[model_key] = model_results
        model_manager.unload_model()

    agg_path = os.path.join(output_dir, "prompt_format_all.json")
    with open(agg_path, "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results


if __name__ == "__main__":
    run_prompt_format_ablation()
