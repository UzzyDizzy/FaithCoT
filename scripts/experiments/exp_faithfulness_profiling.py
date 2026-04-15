#scripts/experiments/exp_faithfulness_profiling.py
"""
Experiment 2: Faithfulness Profiling.

Computes SIG, CNS, and RFI metrics for all model × benchmark combinations.
"""

import os
import sys
import json
from typing import Any, Dict, List


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from configs.model_config import MODEL_REGISTRY
from configs.benchmark_config import BENCHMARK_REGISTRY
from configs.experiment_config import EXPERIMENT_CONFIG, PATHS, ensure_dirs
from src.data.dataset_loader import DatasetLoader
from src.data.preprocessing import preprocess_dataset, load_processed_data
from src.models.model_loader import ModelManager
from src.models.inference import InferenceEngine
from src.metrics.step_information_gain import StepInformationGain
from src.metrics.causal_necessity_score import CausalNecessityScore
from src.metrics.reasoning_fidelity_index import ReasoningFidelityIndex
from src.utils.logger import setup_logger

logger = setup_logger("exp_faithfulness")

from src.utils.cache import get_cache, save_cache
from src.utils.cot_parser import CoTParser


def run_faithfulness_profiling():
    """Run faithfulness profiling for all models × benchmarks."""
    ensure_dirs()
    output_dir = os.path.join(PATHS["raw_results_dir"], "faithfulness")
    os.makedirs(output_dir, exist_ok=True)

    # Load predictions from baseline experiment
    baseline_dir = os.path.join(PATHS["raw_results_dir"], "baseline")

    sig_metric = StepInformationGain(threshold=EXPERIMENT_CONFIG["sig_threshold"])
    cns_metric = CausalNecessityScore(threshold=EXPERIMENT_CONFIG["cns_threshold"])
    rfi_metric = ReasoningFidelityIndex(
        sig_threshold=EXPERIMENT_CONFIG["sig_threshold"],
        rfi_threshold=EXPERIMENT_CONFIG["rfi_threshold"],
    )

    model_manager = ModelManager(cache_dir=PATHS.get("model_cache_dir"))
    all_results = {}

    for model_key in MODEL_REGISTRY:
        config = MODEL_REGISTRY[model_key]
        logger.info(f"=== Faithfulness profiling: {config.short_name} ===")

        # Load model
        try:
            model, tokenizer = model_manager.load_model(config)
            gen_config = model_manager.get_generation_config()
            engine = InferenceEngine(
                model, tokenizer, gen_config,
                device=EXPERIMENT_CONFIG["device"],
                use_amp=EXPERIMENT_CONFIG["use_amp"],
            )
            cot_parser = CoTParser()
        except Exception as e:
            logger.error(f"Failed to load {config.short_name}: {e}")
            continue

        # Load predictions
        pred_path = os.path.join(baseline_dir, f"predictions_{model_key}.json")
        if not os.path.exists(pred_path):
            logger.warning(f"No predictions found for {model_key}, skipping")
            model_manager.unload_model()
            continue

        with open(pred_path, "r") as f:
            pred_data = json.load(f)

        model_results = {}

        for bench_key in BENCHMARK_REGISTRY:
            cache_key = f"faithfulness_{model_key}_{bench_key}"
            cached = get_cache(cache_key)
            if cached:
                logger.info(f"  ⚡ Using cache for {bench_key}")
                model_results[bench_key] = cached
                continue

            predictions = pred_data.get("predictions", {}).get(bench_key, [])
            if not predictions:
                continue

            logger.info(f"  Processing {bench_key} ({len(predictions)} examples)")

            # Limit to fewer samples for metrics (computationally expensive)
            num_samples = min(
                EXPERIMENT_CONFIG.get("num_perturbation_samples", 50),
                len(predictions),
            )
            predictions = predictions[:num_samples]

            # Load dataset via config
            #bench_config = BENCHMARK_REGISTRY[bench_key]
            #loader = DatasetLoader(bench_config)
            #data = loader.load()
            #data = data.select(range(num_samples))
            #data = preprocess_dataset(data, bench_key)

            # Re-generate with parsed CoTs
            examples_with_cot = []

            #for d, p in zip(data, predictions):
            for p in predictions:
                raw_output = p.get("raw_output", "")
                parsed_cot = cot_parser.parse(raw_output)

                examples_with_cot.append({
                    "parsed_cot": parsed_cot,
                    "predicted_answer": p.get("predicted_answer", ""),
                    "is_correct": p.get("is_correct", False),
                })

            # Compute SIG
            logger.info("    Computing SIG...")
            sig_results = sig_metric.compute_batch(engine, examples_with_cot)

            # Compute CNS
            logger.info("    Computing CNS...")
            cns_results = cns_metric.compute_batch(engine, examples_with_cot)

            # Compute RFI
            logger.info("    Computing RFI...")
            rfi_results = rfi_metric.compute_batch(sig_results, cns_results)
            rfi_aggregate = rfi_metric.aggregate(rfi_results)

            model_results[bench_key] = {
                "sig_summary": {
                    "mean_sig": float(sum(r["mean_sig"] for r in sig_results) / max(1, len(sig_results))),
                    "mean_informative_ratio": float(sum(r["informative_ratio"] for r in sig_results) / max(1, len(sig_results))),
                },
                "cns_summary": {
                    "mean_cns": float(sum(r["mean_cns"] for r in cns_results) / max(1, len(cns_results))),
                    "mean_causal_ratio": float(sum(r["causal_ratio"] for r in cns_results) / max(1, len(cns_results))),
                },
                "rfi_aggregate": rfi_aggregate,
                "num_examples": len(examples_with_cot),
            }
            save_cache(cache_key, model_results[bench_key])

            logger.info(
                f"    RFI: {rfi_aggregate['mean_rfi']:.4f}, "
                f"Faithful: {rfi_aggregate['faithful_ratio']:.2%}"
            )

        # Save model results
        save_path = os.path.join(output_dir, f"faithfulness_{model_key}.json")
        with open(save_path, "w") as f:
            json.dump({"model": config.short_name, "results": model_results}, f, indent=2)

        all_results[model_key] = model_results
        model_manager.unload_model()

    # Save aggregate
    agg_path = os.path.join(output_dir, "faithfulness_all_models.json")
    with open(agg_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"All faithfulness results saved to {agg_path}")

    return all_results


if __name__ == "__main__":
    run_faithfulness_profiling()
