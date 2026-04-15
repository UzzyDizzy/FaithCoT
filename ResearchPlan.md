# Research Plan: Mapping the Faithfulness of Chain-of-Thought Reasoning

## 1. Research Question

**Can we build a comprehensive taxonomy of CoT failure modes and develop information-theoretic metrics that predict when reasoning steps add genuine signal versus noise?**

Recent evidence shows CoT reasoning is often a "mirage" — models produce fluent reasoning that doesn't causally drive answers and breaks under distribution shift. This project develops a unified framework to quantify when each reasoning step contributes causally to the answer across model families (DeepSeek-R1 distilled, Qwen-QwQ, Llama-based reasoners).

---

## 2. Key References & Related Work

### 2.1 Core Papers
| Paper | Key Contribution | Gap We Address |
|-------|-----------------|----------------|
| Zhao et al., "Is CoT Reasoning a Mirage?" (arXiv:2508.01191, Aug 2025) | CoT is a "structured inductive bias" from in-distribution data; breaks under distribution shift via DataAlchemy framework | No cross-model comparison; no per-step causal metrics |
| Lu et al., "Reasoning LLMs are Wandering in a Maze" (arXiv:2505.20296, May 2025) | Categorizes failures: invalid explorations, redundant paths, incorrect backtracking, hallucinated conclusions | No information-theoretic quantification; no unified taxonomy |
| "What Makes a Good Reasoning Chain?" (EMNLP 2025 Main) | LCoT2Tree: structural predictors (exploration, backtracking, verification) outperform surface metrics | Focuses on structural tree analysis; lacks causal intervention tests |
| Barez & Wu (Oxford, Jul 2025) | CoT operates as post-hoc rationalization, not faithful reasoning | Observational analysis; no perturbation-based causal validation |
| FACT-E (Apr 2026) | Causality-inspired perturbation to separate genuine step-dependence from bias-driven artifacts | Very recent; focuses on trajectory selection, not taxonomy |

### 2.2 Emerging Methods (2025-2026)
- **Causal Weaving Score (CWS)**: Combines output correctness + causal robustness via structured perturbations
- **Faithfulness by Unlearning Reasoning (FUR)**: Tests parametric faithfulness by unlearning information from model parameters
- **Project Ariadne**: SCM-based auditing for "Reasoning Theater" detection via counterfactual interventions
- **Pivotal Token Search**: Mechanistic interpretability identifying critical reasoning tokens
- **Conditional Mutual Information (CMI)**: Information-theoretic step-level contribution without human annotations

### 2.3 Key Finding: The "Faithfulness Gap"
Multiple 2025 studies confirm:
1. Models frequently engage in "Reasoning Theater" — plausible explanations that don't drive outputs
2. **Inverse scaling of faithfulness**: Larger models may produce *less* faithful reasoning on certain tasks
3. Models fail to correctly update decisions when CoT logic is externally edited/flipped
4. CoT utility as a "scratchpad" remains even when not perfectly faithful

---

## 3. Proposed Methodology

### 3.1 Unified CoT Failure Taxonomy
We propose a **6-category failure taxonomy** synthesized from all prior work:

| Failure Mode | Source | Description | Detection Method |
|-------------|--------|-------------|-----------------|
| **F1: Post-hoc Rationalization** | Barez & Wu | CoT doesn't causally drive answer; model produces same answer regardless | Early Answering Test |
| **F2: Invalid Reasoning Steps** | Lu et al. | Logical errors in reasoning chain | Step-level logical validation |
| **F3: Redundant Exploration** | Lu et al. | Repeating same ineffective reasoning paths | N-gram repetition + semantic similarity |
| **F4: Incorrect Backtracking** | Lu et al. | Restoring inconsistent state during revision | State consistency tracking |
| **F5: Distribution-Dependent Brittleness** | Zhao et al. | CoT breaks under OOD inputs | ID vs OOD accuracy delta |
| **F6: Hallucinated Conclusions** | Lu et al. + EMNLP 2025 | Conclusions unsupported by reasoning steps | Entailment verification |

### 3.2 Information-Theoretic Metrics (Novel Contribution)

We propose **three novel metrics** to quantify reasoning step contribution:

#### Metric 1: Step Information Gain (SIG)
```
SIG(s_i) = I(A; S_i | S_{<i}) = H(A | S_{<i}) - H(A | S_{<=i})
```
Measures the conditional mutual information between step s_i and the final answer A, given all previous steps. Computed via token-level log-probability shifts when each step is appended.

#### Metric 2: Causal Necessity Score (CNS)
```
CNS(s_i) = P(A != a* | do(remove s_i)) - P(A != a* | original)
```
Measures how much removing/corrupting step s_i changes the model's answer. Higher CNS = more causally necessary.

#### Metric 3: Reasoning Fidelity Index (RFI)
```
RFI = (1/N) * sum_{i=1}^{N} 1[SIG(s_i) > tau] * CNS(s_i)
```
Composite metric combining information gain and causal necessity. Ranges 0-1; higher = more faithful.

### 3.3 Perturbation Tests (Causal Validation)
Following Anthropic's protocol with extensions:

1. **Early Answering**: Truncate CoT at each step, measure answer change
2. **Mistake Injection**: Insert logical/factual errors, measure if answer still correct
3. **Step Shuffling**: Randomize step order, measure coherence and accuracy impact
4. **Step Deletion**: Remove individual steps, measure accuracy delta (for CNS)
5. **Paraphrasing**: Semantically equivalent rewrites, measure answer stability

---

## 4. Benchmarks (5 Benchmarks)

| # | Benchmark | Domain | Size (test) | Task Type | Why Selected |
|---|-----------|--------|-------------|-----------|-------------|
| 1 | **GSM8K** | Math (Grade School) | 1,319 | Multi-step arithmetic | Standard CoT benchmark; tests step-by-step math reasoning |
| 2 | **MATH** | Math (Competition) | 5,000 | Advanced math problem solving | Higher difficulty; tests deep reasoning chains |
| 3 | **StrategyQA** | Commonsense/Multi-hop | 2,290 | Yes/No implicit reasoning | Tests implicit multi-hop; reveals if CoT adds value for commonsense |
| 4 | **ARC-Challenge** | Science Reasoning | 1,172 | Multiple choice science | Tests scientific reasoning; moderate difficulty |
| 5 | **FOLIO** | Formal Logic (FOL) | 204 | NLI (Entailment/Contradiction/Unknown) | First-order logic; tests rigorous logical reasoning faithfulness |

### Dataset Loading Strategy
- All loaded via HuggingFace `datasets` library
- Standard splits used; validation created from train (10%) where no val split exists
- Subsample for efficiency: 200 samples per benchmark for main experiments, full test set for final evaluation

---

## 5. Models (5 Open-Source Reasoning Models, Ascending Parameters)

| # | Model | Parameters | HuggingFace ID | Family | Notes |
|---|-------|-----------|----------------|--------|-------|
| 1 | **DeepSeek-R1-Distill-Qwen-7B** | 7B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | DeepSeek-R1 distilled | Smallest reasoning model |
| 2 | **DeepSeek-R1-Distill-Llama-8B** | 8B | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | Llama-based reasoner | Different architecture, similar size |
| 3 | **DeepSeek-R1-Distill-Qwen-14B** | 14B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | DeepSeek-R1 distilled | Mid-range |
| 4 | **Qwen-QwQ-32B** | 32B | `Qwen/QwQ-32B` | Qwen reasoning | Dedicated reasoning model |
| 5 | **DeepSeek-R1-Distill-Qwen-32B** | 32B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | DeepSeek-R1 distilled | Largest distilled model |

### API Models (for comparison, optional)
- `deepseek-chat` / `deepseek-reasoner` via DeepSeek API
- `gpt-4o` / `o1-mini` via OpenAI API

### Model Configuration
- **Precision**: `float16` (AMP enabled) — optimal for 96GB VRAM
- **Max new tokens**: 2048 (reasoning) + 256 (answer extraction)
- **Temperature**: 0.0 (deterministic greedy decoding for reproducibility)
- **Top-p**: 1.0 (no nucleus sampling — ensures determinism)
- **Batch size**: Adaptive per model (7B->16, 8B->16, 14B->8, 32B->4)

---

## 6. Novelty Analysis

### What's New (vs. Prior Work)
1. **Unified Taxonomy**: First 6-category failure taxonomy synthesizing Zhao et al., Lu et al., Barez & Wu, and EMNLP 2025
2. **Information-Theoretic Metrics**: SIG, CNS, and RFI — novel metrics combining conditional mutual information with causal intervention scores
3. **Cross-Model Analysis**: Systematic comparison across 5 model families (7B-32B), revealing how faithfulness scales
4. **Cross-Domain Analysis**: 5 diverse benchmarks (math, logic, commonsense, science) — no prior work spans all these domains
5. **Inverse Scaling Hypothesis Testing**: Explicitly test whether larger models are less faithful

### Deterministic Guarantees (No Worse Than Existing)
- All perturbation tests are established (early answering, mistake injection per Anthropic)
- Information-theoretic metrics are additive/complementary to existing accuracy metrics
- We report standard accuracy alongside faithfulness metrics — faithfulness analysis NEVER modifies model outputs
- Our framework is purely analytical (inference-only); cannot degrade model performance

---

## 7. Experimental Design

### 7.1 Main Experiments
1. **Baseline Accuracy**: All 5 models x 5 benchmarks (standard CoT prompting)
2. **Faithfulness Profiling**: SIG, CNS, RFI computed per model x benchmark
3. **Failure Taxonomy Distribution**: Classify each CoT into failure categories

### 7.2 Ablation Studies (5 Scripts)
1. **Temperature ablation**: {0.0, 0.3, 0.6, 1.0} — effect on faithfulness
2. **CoT length ablation**: Short (<=5 steps), medium (5-15), long (>15)
3. **Perturbation type ablation**: Early answering vs. mistake injection vs. step deletion
4. **Prompt format ablation**: Zero-shot CoT vs. few-shot CoT vs. explicit step format
5. **Model size scaling analysis**: Faithfulness vs. parameter count regression

### 7.3 Visualization
- Heatmaps: Model x Benchmark faithfulness scores
- Radar charts: Failure mode distribution per model
- Scaling curves: Faithfulness vs. model size
- Step-level information gain plots
- Confusion matrices for failure classification

---

## 8. Hardware Utilization Strategy

**Target**: NVIDIA RTX 6000 Pro, 96GB VRAM, 500 TFLOPS, 46 CPUs

| Model | VRAM (est. fp16) | Batch Size | CPU Workers | Throughput |
|-------|-----------------|------------|-------------|------------|
| 7B | ~14GB | 16 | 8 | High |
| 8B | ~16GB | 16 | 8 | High |
| 14B | ~28GB | 8 | 8 | Medium |
| 32B (QwQ) | ~64GB | 4 | 8 | Low-Medium |
| 32B (DS-R1) | ~64GB | 4 | 8 | Low-Medium |

- **AMP**: Enabled via `torch.cuda.amp.autocast(dtype=torch.float16)`
- **CPU optimization**: `num_workers=8` for data loading; `pin_memory=True`
- **No quantization needed**: 96GB VRAM fits all models in fp16
- **Gradient checkpointing**: Not needed (inference-only)
