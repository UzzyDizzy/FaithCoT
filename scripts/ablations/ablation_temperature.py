#scripst/ablations/ablation_temperature.py
"""
Ablation 1: Temperature Effect on Faithfulness.

Tests how sampling temperature affects CoT faithfulness metrics.
Temperatures: {0.0, 0.3, 0.6, 1.0}
"""

import os, sys, json, copy
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from configs.model_config import MODEL_REGISTRY, MODEL_1, GENERATION_CONFIG
from configs.benchmark_config import BENCHMARK_REGISTRY
from configs.experiment_config import EXPERIMENT_CONFIG, PATHS, ensure_dirs
from src.data.dataset_loader import DatasetLoader
from src.data.preprocessing import preprocess_dataset
from src.models.model_loader import ModelManager
from src.models.inference import InferenceEngine
from src.metrics.step_information_gain import StepInformationGain
from src.utils.answer_extractor import AnswerExtractor
from src.utils.logger import setup_logger

logger = setup_logger("ablation_temp")

# Temperature values to test
TEMPERATURES = EXPERIMENT_CONFIG.get("temperature_values", [0.0, 0.3, 0.6, 1.0])

# Use a representative subset of models for ablation
ABLATION_MODELS = ["ds-r1-qwen-7b", "ds-r1-qwen-14b", "qwq-32b"]


def run_temperature_ablation():
    """Run temperature ablation study."""
    ensure_dirs()
    output_dir = os.path.join(PATHS["raw_results_dir"], "ablations", "temperature")
    os.makedirs(output_dir, exist_ok=True)

    model_manager = ModelManager(cache_dir=PATHS.get("model_cache_dir"))
    sig_metric = StepInformationGain()
    answer_extractor = AnswerExtractor()
    num_samples = 50

    all_results = {}

    for model_key in ABLATION_MODELS:
        if model_key not in MODEL_REGISTRY:
            continue
        config = MODEL_REGISTRY[model_key]
        logger.info(f"=== Temperature Ablation: {config.short_name} ===")

        model, tokenizer = model_manager.load_model(config)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        base_gen_config = model_manager.get_generation_config()

        model_results = {}

        for temp in TEMPERATURES:
            logger.info(f"  Temperature: {temp}")

            # Modify generation config for this temperature
            gen_config = copy.deepcopy(base_gen_config)
            gen_config["temperature"] = temp
            gen_config["do_sample"] = temp > 0.0
            if temp > 0:
                gen_config["top_p"] = 0.95
                gen_config["top_k"] = 50

            engine = InferenceEngine(
                model, tokenizer, gen_config,
                device=EXPERIMENT_CONFIG["device"],
                use_amp=EXPERIMENT_CONFIG["use_amp"],
            )

            temp_results = {}
            for bench_key in ["gsm8k", "strategyqa"]:  # Representative benchmarks
                bench_config = BENCHMARK_REGISTRY[bench_key]
                loader = DatasetLoader(bench_config)
                data = loader.load()
                data = data.select(range(min(num_samples, len(data))))
                data = preprocess_dataset(data, bench_key, "zero_shot_cot")

                correct = 0
                total_steps = []
                sig_means = []

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
                    is_correct = answer_extractor.check_answer(
                        predicted, d["gold_answer"], d["answer_type"]
                    )
                    if is_correct:
                        correct += 1
                    total_steps.append(output["num_steps"])

                accuracy = correct / max(1, len(data)) * 100
                temp_results[bench_key] = {
                    "accuracy": accuracy,
                    "mean_steps": float(np.mean(total_steps)) if total_steps else 0,
                    "std_steps": float(np.std(total_steps)) if total_steps else 0,
                }

            model_results[str(temp)] = temp_results

        save_path = os.path.join(output_dir, f"temp_ablation_{model_key}.json")
        with open(save_path, "w") as f:
            json.dump({"model": config.short_name, "results": model_results}, f, indent=2)

        all_results[model_key] = model_results
        model_manager.unload_model()

    agg_path = os.path.join(output_dir, "temp_ablation_all.json")
    with open(agg_path, "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results


if __name__ == "__main__":
    run_temperature_ablation()
