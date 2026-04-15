# Implementation Plan: Mapping the Faithfulness of Chain-of-Thought Reasoning

## Directory Structure

```
FaithCoT/
├── ResearchPlan.md
├── ImplementationPlan.md
├── requirements.txt
├── .env.local                    # API keys (user fills in)
├── main.ipynb                    # Master notebook - entire pipeline
│
├── configs/
│   ├── __init__.py
│   ├── model_config.py           # Model IDs, hyperparams, batch sizes
│   ├── benchmark_config.py       # Dataset names, splits, subsample sizes
│   └── experiment_config.py      # Experiment-level settings
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download_datasets.py  # Download all 5 benchmarks
│   │   ├── dataset_loader.py     # Unified loader interface
│   │   ├── parsers/
│   │   │   ├── __init__.py
│   │   │   ├── gsm8k_parser.py
│   │   │   ├── math_parser.py
│   │   │   ├── strategyqa_parser.py
│   │   │   ├── arc_parser.py
│   │   │   └── folio_parser.py
│   │   └── preprocessing.py      # Train/val/test splits, subsampling
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model_loader.py       # Load HF models with AMP/fp16
│   │   ├── inference.py          # Generate CoT + extract answers
│   │   └── api_models.py         # OpenAI / DeepSeek API wrappers
│   │
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── step_information_gain.py    # SIG metric
│   │   ├── causal_necessity_score.py   # CNS metric
│   │   ├── reasoning_fidelity_index.py # RFI composite metric
│   │   └── failure_taxonomy.py         # 6-category failure classifier
│   │
│   ├── perturbation/
│   │   ├── __init__.py
│   │   ├── early_answering.py
│   │   ├── mistake_injection.py
│   │   ├── step_shuffling.py
│   │   ├── step_deletion.py
│   │   └── paraphrasing.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── cot_parser.py          # Parse CoT into individual steps
│       ├── answer_extractor.py    # Extract final answer from CoT
│       └── logger.py             # Logging utilities
│
├── scripts/
│   ├── run_baseline.py           # Run baseline accuracy (all models x benchmarks)
│   ├── run_faithfulness.py       # Compute faithfulness metrics
│   ├── run_failure_taxonomy.py   # Classify failure modes
│   │
│   ├── ablations/
│   │   ├── ablation_temperature.py
│   │   ├── ablation_cot_length.py
│   │   ├── ablation_perturbation_type.py
│   │   ├── ablation_prompt_format.py
│   │   └── ablation_model_scaling.py
│   │
│   ├── experiments/
│   │   ├── exp_baseline_accuracy.py
│   │   ├── exp_faithfulness_profiling.py
│   │   ├── exp_perturbation_tests.py
│   │   ├── exp_failure_classification.py
│   │   └── exp_cross_model_analysis.py
│   │
│   └── visualization/
│       ├── plot_heatmaps.py
│       ├── plot_radar_charts.py
│       ├── plot_scaling_curves.py
│       ├── plot_step_information.py
│       └── generate_tables.py
│
├── results/
│   ├── tables/
│   ├── figures/
│   └── raw/
│
└── data/
    └── raw/
```

## Proposed Changes

### Phase 1: Configuration & Utilities
- Model config with all 5 models, hyperparameters, batch sizes
- Benchmark config with all 5 datasets
- Global experiment config (precision, seeds, paths)

### Phase 2: Data Pipeline
- Per-benchmark parsers that normalize to unified format: {question, answer, gold_cot (if available), metadata}
- Download script using HuggingFace datasets
- Preprocessing: train/val/test splits, subsampling

### Phase 3: Model Pipeline
- Unified model loader for all HF models (fp16, AMP)
- CoT inference engine with step-by-step generation
- API wrappers for DeepSeek/OpenAI

### Phase 4: Metrics & Perturbation
- Three information-theoretic metrics (SIG, CNS, RFI)
- Five perturbation tests (early answering, mistake injection, shuffling, deletion, paraphrasing)
- Six-category failure taxonomy classifier

### Phase 5: Experiments & Ablations
- 5 experiment scripts (baseline, faithfulness, perturbation, failure classification, cross-model)
- 5 ablation scripts (temperature, length, perturbation type, prompt format, scaling)

### Phase 6: Visualization & Results
- Heatmaps, radar charts, scaling curves, step-level plots
- Table generators (min 3 result tables)

### Phase 7: Master Notebook
- main.ipynb orchestrating entire pipeline end-to-end

## Verification Plan
- Static code verification agent checks all imports, API signatures, tensor shapes, file paths
- No actual execution or dependency installation required
