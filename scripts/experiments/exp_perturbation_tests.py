#scripts/experiments/exp_perturbation_tests.py
"""
Experiment 3: Perturbation Tests.

Runs all 5 perturbation tests (early answering, mistake injection,
step shuffling, step deletion, paraphrasing) across models and benchmarks.
"""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from configs.model_config import MODEL_REGISTRY
from configs.benchmark_config import BENCHMARK_REGISTRY
from configs.experiment_config import EXPERIMENT_CONFIG, PATHS, ensure_dirs
from src.data.dataset_loader import DatasetLoader
from src.data.preprocessing import preprocess_dataset
from src.models.model_loader import ModelManager
from src.models.inference import InferenceEngine
from src.perturbation.early_answering import EarlyAnsweringTest
from src.perturbation.mistake_injection import MistakeInjectionTest
from src.perturbation.step_shuffling import StepShufflingTest
from src.perturbation.step_deletion import StepDeletionTest
from src.perturbation.paraphrasing import ParaphrasingTest
from src.utils.logger import setup_logger

from src.utils.cot_parser import CoTParser
from src.utils.cache import get_cache, save_cache

logger = setup_logger("exp_perturbation")


def run_perturbation_tests():
    """Run all perturbation tests."""
    ensure_dirs()
    output_dir = os.path.join(PATHS["raw_results_dir"], "perturbation")
    baseline_dir = os.path.join(PATHS["raw_results_dir"], "baseline")
    os.makedirs(output_dir, exist_ok=True)

    # Initialize perturbation tests
    seed = EXPERIMENT_CONFIG["seed"]
    tests = {
        "early_answering": EarlyAnsweringTest(),
        "mistake_injection": MistakeInjectionTest(seed=seed),
        "step_shuffling": StepShufflingTest(num_shuffles=3, seed=seed),
        "step_deletion": StepDeletionTest(),
        "paraphrasing": ParaphrasingTest(seed=seed),
    }

    model_manager = ModelManager(cache_dir=PATHS.get("model_cache_dir"))
    all_results = {}

    cot_parser = CoTParser()

    for model_key in MODEL_REGISTRY:
        config = MODEL_REGISTRY[model_key]
        logger.info(f"=== Perturbation tests: {config.short_name} ===")

        try:
            model, tokenizer = model_manager.load_model(config)
            gen_config = model_manager.get_generation_config()
            engine = InferenceEngine(
                model, tokenizer, gen_config,
                device=EXPERIMENT_CONFIG["device"],
                use_amp=EXPERIMENT_CONFIG["use_amp"],
            )
        except Exception as e:
            logger.error(f"Failed to load {config.short_name}: {e}")
            continue

        model_results = {}

        pred_path = os.path.join(baseline_dir, f"predictions_{model_key}.json")

        if not os.path.exists(pred_path):
            logger.warning(f"No baseline predictions for {model_key}")
            model_manager.unload_model()
            continue

        with open(pred_path) as f:
            pred_data = json.load(f)

        num_samples = min(50, EXPERIMENT_CONFIG.get("num_perturbation_samples", 50))

        for bench_key in BENCHMARK_REGISTRY:
            predictions = pred_data.get("predictions", {}).get(bench_key, [])

            if not predictions:
                continue

            predictions = predictions[:num_samples]

            cache_key = f"faithfulness_{model_key}_{bench_key}"

            cached = get_cache(cache_key)
            if cached:
                logger.info(f"⚡ Using cache for {bench_key}")
                model_results[bench_key] = cached
                continue

            logger.info(f"  Benchmark: {bench_key}")

            # bench_config = BENCHMARK_REGISTRY[bench_key]
            # loader = DatasetLoader(bench_config)
            # data = loader.load()
            # data = data.select(range(min(EXPERIMENT_CONFIG["subsample_size"],len(data))))
            # data = preprocess_dataset(data, bench_key, "zero_shot_cot")

            # # Generate CoTs first
            examples = []
            
            # prompts = [d["prompt"] for d in data]

            # outputs = engine.generate_batch(
            #     prompts,
            #     batch_size=config.batch_size
            # )

            #for d, output in zip(data, outputs):
            for p in predictions:
                raw_output = p.get("raw_output", "")
                parsed_cot = cot_parser.parse(raw_output)

                examples.append({
                    "parsed_cot": parsed_cot,
                    "predicted_answer": p.get("predicted_answer", ""),
                    "raw_output": raw_output,
                })

            # Run each perturbation test
            bench_results = {}
            for test_name, test in tests.items():
                logger.info(f"    Running {test_name}...")
                try:
                    test_results = test.run_batch(engine, examples)

                    # Aggregate
                    if test_name == "early_answering":
                        post_hoc_count = sum(1 for r in test_results if r.get("is_post_hoc", False))
                        bench_results[test_name] = {
                            "post_hoc_ratio": post_hoc_count / max(1, len(test_results)),
                            "mean_early_answering_ratio": sum(r.get("early_answering_ratio", 0) for r in test_results) / max(1, len(test_results)),
                        }
                    elif test_name == "mistake_injection":
                        unfaithful_count = sum(1 for r in test_results if r.get("is_unfaithful", False))
                        bench_results[test_name] = {
                            "unfaithful_ratio": unfaithful_count / max(1, len(test_results)),
                            "mean_ignores_ratio": sum(r.get("ignores_mistakes_ratio", 0) for r in test_results) / max(1, len(test_results)),
                        }
                    elif test_name == "step_shuffling":
                        order_matters_count = sum(1 for r in test_results if r.get("order_matters", True))
                        bench_results[test_name] = {
                            "order_matters_ratio": order_matters_count / max(1, len(test_results)),
                            "mean_shuffle_robustness": sum(r.get("robust_to_shuffling_ratio", 0) for r in test_results) / max(1, len(test_results)),
                        }
                    elif test_name == "step_deletion":
                        bench_results[test_name] = {
                            "mean_causal_step_ratio": sum(r.get("causal_step_ratio", 0) for r in test_results) / max(1, len(test_results)),
                        }
                    elif test_name == "paraphrasing":
                        sensitive_count = sum(1 for r in test_results if r.get("is_surface_sensitive", False))
                        bench_results[test_name] = {
                            "surface_sensitive_ratio": sensitive_count / max(1, len(test_results)),
                        }

                    logger.info(f"      {test_name}: {bench_results[test_name]}")
                except Exception as e:
                    logger.error(f"    {test_name} failed: {e}")
                    bench_results[test_name] = {"error": str(e)}

            model_results[bench_key] = bench_results
            save_cache(cache_key, bench_results)

        save_path = os.path.join(output_dir, f"perturbation_{model_key}.json")
        with open(save_path, "w") as f:
            json.dump({"model": config.short_name, "results": model_results}, f, indent=2)

        all_results[model_key] = model_results
        model_manager.unload_model()

    agg_path = os.path.join(output_dir, "perturbation_all_models.json")
    with open(agg_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"All perturbation results saved to {agg_path}")

    return all_results


if __name__ == "__main__":
    run_perturbation_tests()
