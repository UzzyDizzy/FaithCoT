#scripts/experiments/exp_failure_classification.py
"""
Experiment 4: Failure Classification.

Classifies CoT outputs into the 6 failure categories across all models.
"""

import os, sys, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from configs.model_config import MODEL_REGISTRY
from configs.benchmark_config import BENCHMARK_REGISTRY
from configs.experiment_config import EXPERIMENT_CONFIG, PATHS, ensure_dirs
from src.data.dataset_loader import DatasetLoader
from src.data.preprocessing import preprocess_dataset
from src.models.model_loader import ModelManager
from src.models.inference import InferenceEngine
from src.metrics.failure_taxonomy import FailureTaxonomyClassifier
from src.utils.logger import setup_logger

from src.utils.cot_parser import CoTParser
cot_parser = CoTParser()

logger = setup_logger("exp_failure")


def run_failure_classification():
    """Classify CoT failure modes across all models and benchmarks."""
    ensure_dirs()
    output_dir = os.path.join(PATHS["raw_results_dir"], "failure_taxonomy")
    baseline_dir = os.path.join(PATHS["raw_results_dir"], "baseline")
    os.makedirs(output_dir, exist_ok=True)

    classifier = FailureTaxonomyClassifier()
    model_manager = ModelManager(cache_dir=PATHS.get("model_cache_dir"))
    all_results = {}
    num_samples = min(100, EXPERIMENT_CONFIG.get("subsample_size", 200))

    for model_key in MODEL_REGISTRY:
        config = MODEL_REGISTRY[model_key]
        logger.info(f"=== Failure classification: {config.short_name} ===")

        # try:
        #     model, tokenizer = model_manager.load_model(config)
        #     gen_config = model_manager.get_generation_config()
        #     engine = InferenceEngine(
        #         model, tokenizer, gen_config,
        #         device=EXPERIMENT_CONFIG["device"],
        #         use_amp=EXPERIMENT_CONFIG["use_amp"],
        #     )
        # except Exception as e:
        #     logger.error(f"Failed to load {config.short_name}: {e}")
        #     continue

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
            logger.info(f"  Benchmark: {bench_key}")

            predictions = pred_data.get("predictions", {}).get(bench_key, [])

            if not predictions:
                continue

            predictions = predictions[:num_samples]

            # bench_config = BENCHMARK_REGISTRY[bench_key]
            # loader = DatasetLoader(bench_config)
            # data = loader.load()
            # data = data.select(range(min(EXPERIMENT_CONFIG["subsample_size"],len(data))))
            # data = preprocess_dataset(data, bench_key, "zero_shot_cot")

            examples_for_classification = []

            for p in predictions:
                raw_output = p.get("raw_output", "")
                parsed_cot = cot_parser.parse(raw_output)

                examples_for_classification.append({
                    "parsed_cot": parsed_cot,
                    "predicted_answer": p.get("predicted_answer", ""),
                    "raw_output": raw_output,
                })
            
            # prompts = [d["prompt"] for d in data]

            # outputs = engine.generate_batch(
            #     prompts,
            #     batch_size=config.batch_size
            # )

            # for d, output in zip(data, outputs):
            #     predicted = engine.extract_answer(
            #         output["raw_output"], d["answer_type"], bench_key
            #     )
            #     from src.utils.answer_extractor import AnswerExtractor
            #     ae = AnswerExtractor()
            #     is_correct = ae.check_answer(predicted, d["gold_answer"], d["answer_type"])

            #     examples_for_classification.append({
            #         "parsed_cot": output["parsed_cot"],
            #         "is_correct": is_correct,
            #         "sig_result": None,
            #         "cns_result": None,
            #         "rfi_result": None,
            #         "early_answering_same": False,
            #     })

            classifications = classifier.classify_batch(examples_for_classification)
            taxonomy_summary = classifier.aggregate_taxonomy(classifications)

            model_results[bench_key] = taxonomy_summary
            logger.info(f"    Failure rates: {taxonomy_summary.get('failure_rates', {})}")

        save_path = os.path.join(output_dir, f"failure_{model_key}.json")
        with open(save_path, "w") as f:
            json.dump({"model": config.short_name, "results": model_results}, f, indent=2)

        all_results[model_key] = model_results
        model_manager.unload_model()

    agg_path = os.path.join(output_dir, "failure_all_models.json")
    with open(agg_path, "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results


if __name__ == "__main__":
    run_failure_classification()
