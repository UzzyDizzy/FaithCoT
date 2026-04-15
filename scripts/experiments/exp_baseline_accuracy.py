#scripts/experiments/exp_baseline_accuracy.py
"""
Experiment 1: Baseline Accuracy.

Runs all 5 models on all 5 benchmarks with standard CoT prompting.
Produces accuracy results as the baseline for faithfulness analysis.
"""

import os
import sys
import json
import time
from typing import Any, Dict, List

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from configs.model_config import MODEL_REGISTRY, GENERATION_CONFIG
from configs.benchmark_config import BENCHMARK_REGISTRY
from configs.experiment_config import EXPERIMENT_CONFIG, PATHS, ensure_dirs
from src.data.dataset_loader import DatasetLoader
from src.data.preprocessing import preprocess_dataset
from src.models.model_loader import ModelManager
from src.models.inference import InferenceEngine
from src.utils.answer_extractor import AnswerExtractor
from src.utils.logger import setup_logger

logger = setup_logger("exp_baseline")


def run_baseline_for_model(
    model_key: str,
    benchmarks: Dict[str, List[Dict]],
    model_manager: ModelManager,
    output_dir: str,
) -> Dict[str, Any]:
    """Run baseline accuracy for one model on all benchmarks.

    Args:
        model_key: Model registry key
        benchmarks: Dict of benchmark_key -> parsed examples
        model_manager: ModelManager instance
        output_dir: Directory to save results

    Returns:
        Dict with accuracy per benchmark
    """
    config = MODEL_REGISTRY[model_key]
    logger.info(f"=== Running baseline for {config.short_name} ===")

    model, tokenizer = model_manager.load_model(config)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    gen_config = model_manager.get_generation_config()

    engine = InferenceEngine(
        model, tokenizer, gen_config,
        device=EXPERIMENT_CONFIG["device"],
        use_amp=EXPERIMENT_CONFIG["use_amp"],
    )

    answer_extractor = AnswerExtractor()
    results = {}

    for bench_key, examples in benchmarks.items():
        logger.info(f"  Benchmark: {bench_key} ({len(examples)} examples)")

        correct = 0
        predictions = []

        # BATCH
        batch_size = min(config.batch_size, len(examples))
        prompts = [ex["prompt"] for ex in examples]

        outputs = engine.generate_batch(
            prompts,
            batch_size=batch_size,
            max_new_tokens=config.max_new_tokens
        )

        for i, (ex, output) in enumerate(zip(examples, outputs)):
            predicted = engine.extract_answer(
                output["raw_output"],
                ex["answer_type"],
                bench_key
            )

            is_correct = answer_extractor.check_answer(
                predicted,
                ex["gold_answer"],
                ex["answer_type"]
            )

            if is_correct:
                correct += 1

            predictions.append({
                "id": ex.get("id", f"{bench_key}_{i}"),
                "question": ex["question"][:200],
                "gold_answer": ex["gold_answer"],
                "predicted_answer": predicted,
                "is_correct": is_correct,
                "raw_output": output["raw_output"][:500],
                "num_steps": output["num_steps"],
                "num_tokens": output["num_generated_tokens"],
            })

            if i < 3:
                print("\n--- DEBUG ---")
                print("OUTPUT:", output["raw_output"][:200])
                print("PRED:", predicted)
                print("GOLD:", ex["gold_answer"])

            if (i + 1) % 50 == 0:
                acc = correct / (i + 1) * 100
                logger.info(f"    {bench_key}: {i+1}/{len(examples)}, Acc: {acc:.1f}%")

        accuracy = correct / max(1, len(examples)) * 100

        results[bench_key] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(examples),
            "predictions": predictions,
        }

        logger.info(f"  {bench_key}: {accuracy:.2f}%")

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    save_path = os.path.join(output_dir, f"baseline_{model_key}.json")
    summary = {
        k: {kk: vv for kk, vv in v.items() if kk != "predictions"}
        for k, v in results.items()
    }

    with open(save_path, "w") as f:
        json.dump({"model": config.short_name, "results": summary}, f, indent=2)

    pred_path = os.path.join(output_dir, f"predictions_{model_key}.json")
    with open(pred_path, "w") as f:
        json.dump({
            "model": config.short_name,
            "predictions": {k: v["predictions"] for k, v in results.items()}
        }, f, indent=2)

    model_manager.unload_model()
    return results


def run_all_baselines():
    """Run baseline accuracy for all models on all benchmarks."""
    ensure_dirs()

    output_dir = os.path.join(PATHS["raw_results_dir"], "baseline")
    os.makedirs(output_dir, exist_ok=True)

    benchmarks = {}

    # CORRECT LOADING
    for bench_key, bench_config in BENCHMARK_REGISTRY.items():
        loader = DatasetLoader(bench_config)
        data = loader.load()
        data = data.select(range(min(EXPERIMENT_CONFIG["subsample_size"],len(data))))
        data = preprocess_dataset(data, bench_key, "zero_shot_cot")
        benchmarks[bench_key] = data

    model_manager = ModelManager(cache_dir=PATHS.get("model_cache_dir"))
    all_results = {}

    for model_key in MODEL_REGISTRY:
        try:
            results = run_baseline_for_model(
                model_key,
                benchmarks,
                model_manager,
                output_dir
            )
            all_results[model_key] = results

        except Exception as e:
            logger.error(f"Failed for model {model_key}: {e}")
            all_results[model_key] = {"error": str(e)}

    # Save aggregate
    agg = {
        mk: {
            bk: {"accuracy": bv["accuracy"]}
            for bk, bv in mv.items()
        }
        for mk, mv in all_results.items()
        if "error" not in mv
    }

    with open(os.path.join(output_dir, "baseline_all_models.json"), "w") as f:
        json.dump(agg, f, indent=2)

    logger.info("=== Baseline Complete ===")
    return all_results


if __name__ == "__main__":
    run_all_baselines()
