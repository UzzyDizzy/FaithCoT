# Walkthrough: FaithCoT — Mapping the Faithfulness of Chain-of-Thought Reasoning

## Research Goal

Develop a comprehensive taxonomy of CoT failure modes and information-theoretic metrics (SIG, CNS, RFI) to predict when reasoning steps add genuine signal vs. noise, tested across 5 models × 5 benchmarks.

---

## Complete File Structure

```
FaithCoT/
├── ResearchPlan.md                          # Research plan document
├── ImplementationPlan.md                    # Implementation plan
├── requirements.txt                         # Python dependencies
├── .env.local                               # API keys template
├── .gitignore                               # Git ignore rules
├── main.ipynb                               # Master notebook (full pipeline)
├── validate_pipeline.py                     # Validation script (all tests pass)
│
├── configs/
│   ├── __init__.py                          # Config package exports
│   ├── model_config.py                      # 5 models: DS-R1-Qwen-7B/14B/32B, DS-R1-Llama-8B, QwQ-32B
│   ├── benchmark_config.py                  # 5 benchmarks + 3 prompt formats (15 templates)
│   └── experiment_config.py                 # Paths, seeds, hardware, thresholds, ablation params
│
├── src/
│   ├── __init__.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── cot_parser.py                    # Multi-format CoT parser (think tags, numbered, NL)
│   │   ├── answer_extractor.py              # Numeric/yes-no/MC/NLI answer extraction + comparison
│   │   └── logger.py                        # Timestamped logging utility
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── parsers/
│   │   │   ├── __init__.py
│   │   │   ├── gsm8k_parser.py              # GSM8K: #### format extraction
│   │   │   ├── math_parser.py               # MATH: \boxed{} extraction
│   │   │   ├── strategyqa_parser.py         # StrategyQA: bool → yes/no
│   │   │   ├── arc_parser.py                # ARC: choices dict parsing
│   │   │   └── folio_parser.py              # FOLIO: NLI label normalization
│   │   ├── dataset_loader.py                # Unified loader (local JSONL + HuggingFace fallback)
│   │   ├── download_datasets.py             # Bulk downloader for all 5 datasets
│   │   └── preprocessing.py                 # Splits, prompt formatting, serialization
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model_loader.py                  # fp16 model loading with 96GB VRAM optimization
│   │   ├── inference.py                     # CoT gen, log-prob extraction, batch inference + AMP
│   │   └── api_models.py                    # OpenAI/DeepSeek API wrappers
│   │
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── step_information_gain.py         # SIG: H(A|S<i) − H(A|S≤i) per step
│   │   ├── causal_necessity_score.py        # CNS: step deletion → answer change detection
│   │   ├── reasoning_fidelity_index.py      # RFI: composite SIG×CNS with 4-category classification
│   │   └── failure_taxonomy.py              # 6-category failure classifier (F1-F6)
│   │
│   └── perturbation/
│       ├── __init__.py
│       ├── early_answering.py               # Truncate CoT at each step → detect post-hoc reasoning
│       ├── mistake_injection.py             # Corrupt steps → detect if model ignores its own CoT
│       ├── step_shuffling.py                # Randomize step order → test logical dependency
│       ├── step_deletion.py                 # Remove steps → compute CNS
│       └── paraphrasing.py                  # Surface-form rewrites → test sensitivity
│
├── scripts/
│   ├── experiments/
│   │   ├── exp_baseline_accuracy.py         # Exp 1: All models × all benchmarks accuracy
│   │   ├── exp_faithfulness_profiling.py    # Exp 2: SIG/CNS/RFI computation
│   │   ├── exp_perturbation_tests.py        # Exp 3: All 5 perturbation tests
│   │   ├── exp_failure_classification.py    # Exp 4: 6-category failure taxonomy
│   │   └── exp_cross_model_analysis.py      # Exp 5: Inverse scaling hypothesis test
│   │
│   ├── ablations/
│   │   ├── ablation_temperature.py          # Abl 1: Temperature {0.0, 0.3, 0.6, 1.0}
│   │   ├── ablation_cot_length.py           # Abl 2: Short/Medium/Long CoT bins
│   │   ├── ablation_perturbation_type.py    # Abl 3: Compare perturbation effectiveness
│   │   ├── ablation_prompt_format.py        # Abl 4: Zero-shot/Few-shot/Explicit
│   │   └── ablation_model_scaling.py        # Abl 5: 7B→32B scaling regression
│   │
│   └── visualization/
│       ├── plot_heatmaps.py                 # Accuracy, RFI, perturbation heatmaps
│       ├── plot_radar_charts.py             # Failure mode + step type radar charts
│       ├── plot_scaling_curves.py           # Accuracy vs faithfulness scaling + scatter
│       ├── plot_step_information.py          # Per-step SIG bar charts (4 exemplar patterns)
│       └── generate_tables.py               # 4 markdown tables (accuracy, faithfulness, perturbation, ablation)
```

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| **Precision** | float16 globally | 96GB VRAM fits all models up to 32B in fp16 without quantization |
| **Decoding** | Greedy (temp=0.0) | Deterministic for reproducibility; temperature ablation varies this |
| **Batch sizes** | 16 (7-8B), 8 (14B), 4 (32B) | Maximizes GPU utilization per model size |
| **Subsample** | 200 per benchmark | Balances statistical significance with compute budget |
| **AMP** | Enabled | Reduces inference time with no accuracy loss |
| **SIG threshold** | τ = 0.01 | Below this entropy reduction, step is classified as noise |
| **RFI threshold** | 0.3 | Below this, overall reasoning is classified as unfaithful |

---

## Novel Contributions

### 3 Information-Theoretic Metrics

1. **Step Information Gain (SIG)**: `SIG(sᵢ) = H(A|S<ᵢ) − H(A|S≤ᵢ)` — measures entropy reduction per step
2. **Causal Necessity Score (CNS)**: Binary indicator of whether removing a step changes the answer
3. **Reasoning Fidelity Index (RFI)**: `RFI = (1/N) Σ 1[SIG(sᵢ) > τ] · CNS(sᵢ)` — composite faithfulness score

### 4-Category Step Classification

| SIG > τ | CNS > 0 | Category | Meaning |
|---|---|---|---|
| ✓ | ✓ | **Faithful** | Step is both informative AND causally necessary |
| ✓ | ✗ | **Decorative** | Provides info but removing it doesn't change answer |
| ✗ | ✓ | **Shortcut** | Not informative but affects answer (surface sensitivity) |
| ✗ | ✗ | **Irrelevant** | Neither informative nor causal |

### 6-Category Failure Taxonomy (F1-F6)

- **F1**: Post-hoc Rationalization (detected by early answering)
- **F2**: Invalid Reasoning Steps (detected by arithmetic/logic checks)
- **F3**: Redundant Exploration (detected by n-gram overlap)
- **F4**: Incorrect Backtracking (detected by consecutive backtrack markers)
- **F5**: Distribution-Dependent Brittleness (detected by OOD comparison)
- **F6**: Hallucinated Conclusions (detected by unsupported assertion markers + low final-step SIG)

---

## Validation Results

```
VALIDATION COMPLETE — All checks passed!

Pipeline Summary:
  Models: 5 (ds-r1-qwen-7b, ds-r1-llama-8b, ds-r1-qwen-14b, qwq-32b, ds-r1-qwen-32b)
  Benchmarks: 5 (gsm8k, math, strategyqa, arc_challenge, folio)
  Prompt formats: 3
  Metrics: SIG, CNS, RFI, Failure Taxonomy (4)
  Perturbation tests: 5
  Experiments: 5 scripts
  Ablations: 5 scripts
  Visualizations: 4 scripts + table generator
  Master notebook: main.ipynb
```

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run validation
python validate_pipeline.py

# 3. Run the full pipeline via notebook
jupyter notebook main.ipynb

# 4. Or run individual experiments
python scripts/experiments/exp_baseline_accuracy.py
python scripts/experiments/exp_faithfulness_profiling.py
python scripts/experiments/exp_perturbation_tests.py
python scripts/experiments/exp_failure_classification.py
python scripts/experiments/exp_cross_model_analysis.py

# 5. Run ablations
python scripts/ablations/ablation_temperature.py
python scripts/ablations/ablation_cot_length.py
python scripts/ablations/ablation_perturbation_type.py
python scripts/ablations/ablation_prompt_format.py
python scripts/ablations/ablation_model_scaling.py

# 6. Generate tables and figures
python scripts/visualization/generate_tables.py
python scripts/visualization/plot_heatmaps.py
python scripts/visualization/plot_radar_charts.py
python scripts/visualization/plot_scaling_curves.py
python scripts/visualization/plot_step_information.py
```
