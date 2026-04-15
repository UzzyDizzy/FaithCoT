#configs/benchmark_config.py
"""
Benchmark configuration for FaithCoT experiments.

Defines all 5 reasoning benchmarks with their HuggingFace dataset IDs,
field mappings, task types, and evaluation strategies.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark dataset."""
    name: str                          # Human-readable name
    hf_dataset_id: str                # HuggingFace dataset identifier
    hf_subset: Optional[str]          # HuggingFace config/subset name
    task_type: str                     # "math", "logic", "commonsense", "science"
    answer_type: str                   # "numeric", "yes_no", "multiple_choice", "nli"
    question_field: str                # Field name for question text
    answer_field: str                  # Field name for answer text
    train_split: str = "train"
    test_split: str = "test"
    val_split: Optional[str] = None   # None = create from train
    val_fraction: float = 0.1          # Fraction of train for validation
    subsample_size: int = 200          # Samples for main experiments
    full_eval: bool = False            # Use full test set for final eval
    extra_fields: Dict[str, str] = field(default_factory=dict)


# ============================================================
# BENCHMARK REGISTRY: 5 Diverse Reasoning Benchmarks
# ============================================================

# --- Benchmark 1: GSM8K (Grade School Math) ---
GSM8K = BenchmarkConfig(
    name="GSM8K",
    hf_dataset_id="openai/gsm8k",
    hf_subset="main",
    task_type="math",
    answer_type="numeric",
    question_field="question",
    answer_field="answer",
    train_split="train",
    test_split="test",
    val_split=None,
    subsample_size=200,
)

# --- Benchmark 2: MATH (Competition Mathematics) ---
MATH = BenchmarkConfig(
    name="MATH",
    hf_dataset_id="qwedsacf/competition_math",
    hf_subset=None,
    task_type="math",
    answer_type="numeric",
    question_field="problem",
    answer_field="solution",
    train_split="train",
    test_split="train",
    val_split=None,
    subsample_size=200,
    extra_fields={"level": "level", "type": "type"},
)

# --- Benchmark 3: StrategyQA (Multi-hop Commonsense) ---
STRATEGYQA = BenchmarkConfig(
    name="StrategyQA",
    hf_dataset_id="ChilleD/StrategyQA",
    hf_subset=None,
    task_type="commonsense",
    answer_type="yes_no",
    question_field="question",
    answer_field="answer",
    train_split="train",
    test_split="test",
    val_split=None,
    subsample_size=200,
)

# --- Benchmark 4: ARC-Challenge (Science Reasoning) ---
ARC_CHALLENGE = BenchmarkConfig(
    name="ARC-Challenge",
    hf_dataset_id="allenai/ai2_arc",
    hf_subset="ARC-Challenge",
    task_type="science",
    answer_type="multiple_choice",
    question_field="question",
    answer_field="answerKey",
    train_split="train",
    test_split="test",
    val_split="validation",
    subsample_size=200,
    extra_fields={"choices": "choices"},
)

# --- Benchmark 5: FOLIO (First-Order Logic) ---
FOLIO = BenchmarkConfig(
    name="FOLIO",
    hf_dataset_id="tasksource/folio",
    hf_subset=None,
    task_type="logic",
    answer_type="nli",
    question_field="premises",
    answer_field="label",
    train_split="train",
    test_split="validation",  # FOLIO uses validation as test
    val_split=None,
    subsample_size=200,
    extra_fields={"conclusion": "conclusion"},
)

# All benchmarks
BENCHMARK_REGISTRY: Dict[str, BenchmarkConfig] = {
    "gsm8k": GSM8K,
    "math": MATH,
    "strategyqa": STRATEGYQA,
    "arc_challenge": ARC_CHALLENGE,
    "folio": FOLIO,
}


# ============================================================
# PROMPT TEMPLATES (Per benchmark, per format)
# ============================================================

PROMPT_TEMPLATES = {
    "zero_shot_cot": {
        "gsm8k": (
            "Solve the following math problem step by step. "
            "Show your reasoning clearly, then provide the final numeric answer ONLY.\n\n"
            "Problem: {question}\n\n"
            "Solution: Let me think step by step.\n"
            "IMPORTANT:\n"
            "- Output ONLY the final answer.\n"
            "- Do NOT write anything after final answer.\n\n"
            "Final Answer: <answer>\n\n"
        ),
        "math": (
            "Solve the following competition math problem step by step. "
            "Show your detailed reasoning, then provide the final answer ONLY.\n\n"
            "Problem: {question}\n\n"
            "Solution: Let me think step by step.\n"
            "IMPORTANT:\n"
            "- Output ONLY the final answer.\n"
            "- Do NOT write anything after final answer.\n\n"
            "Final Answer: <answer>\n\n"
        ),
        "strategyqa": (
            "Answer the following question with 'Yes' or 'No' ONLY. "
            "Show your reasoning step by step before giving the final answer.\n\n"
            "Question: {question}\n\n"
            "Reasoning: Let me think step by step.\n"
            "IMPORTANT:\n"
            "- Output ONLY the final answer.\n"
            "- Do NOT write anything after final answer.\n\n"
            "Final Answer: <answer>\n\n"
        ),
        "arc_challenge": (
            "Answer the following science question by selecting the correct option ONLY. "
            "Show your reasoning step by step.\n\n"
            "Question: {question}\n"
            "Options:\n{choices}\n\n"
            "Reasoning: Let me think step by step.\n"
            "IMPORTANT:\n"
            "- Output ONLY the final answer.\n"
            "- Do NOT write anything after final answer.\n\n"
            "Final Answer: <answer>\n\n"
        ),
        "folio": (
            "Given the following premises and conclusion, determine if the conclusion "
            "is 'True', 'False', or 'Unknown' ONLY based on the premises. "
            "Show your logical reasoning step by step.\n\n"
            "Premises: {question}\n"
            "Conclusion: {conclusion}\n\n"
            "Reasoning: Let me think step by step.\n"
            "IMPORTANT:\n"
            "- Output ONLY the final answer.\n"
            "- Do NOT write anything after final answer.\n\n"
            "Final Answer: <answer>\n\n"
        ),
    },
    "few_shot_cot": {
        "gsm8k": (
            "Solve math problems step by step. Here are examples:\n\n"
            "Problem: Janet's ducks lay 16 eggs per day. She eats three for breakfast "
            "every morning and bakes muffins for her friends every day with four. She "
            "sells the remainder for $2 each. How much does she make per day?\n"
            "Solution: Step 1: Janet has 16 eggs per day.\n"
            "Step 2: She uses 3 + 4 = 7 eggs.\n"
            "Step 3: She sells 16 - 7 = 9 eggs.\n"
            "Step 4: She makes 9 * $2 = $18 per day.\n"
            "Final Answer: 18\n\n"
            "Now solve this problem:\n"
            "Problem: {question}\n\n"
            "Solution:\n"
        ),
        "math": (
            "Solve competition math problems step by step.\n\n"
            "Problem: What is the value of $\\sqrt{{16}}$?\n"
            "Solution: Step 1: Recall that $\\sqrt{{16}}$ means finding a number that "
            "when multiplied by itself gives 16.\n"
            "Step 2: $4 \\times 4 = 16$.\n"
            "Final Answer: 4\n\n"
            "Now solve:\n"
            "Problem: {question}\n\n"
            "Solution:\n"
        ),
        "strategyqa": (
            "Answer yes/no questions with step-by-step reasoning.\n\n"
            "Question: Would a pear sink in water?\n"
            "Reasoning: Step 1: The density of a pear is about 0.6 g/cm³.\n"
            "Step 2: Water has a density of 1.0 g/cm³.\n"
            "Step 3: Objects less dense than water float.\n"
            "Final Answer: No\n\n"
            "Now answer:\n"
            "Question: {question}\n\n"
            "Reasoning:\n"
        ),
        "arc_challenge": (
            "Answer science questions with reasoning.\n\n"
            "Question: Which property of a mineral can be determined just by looking at it?\n"
            "Options: (A) luster (B) mass (C) weight (D) hardness\n"
            "Reasoning: Step 1: Luster is the way light reflects off a mineral's surface.\n"
            "Step 2: You can see how shiny or dull a mineral is just by looking.\n"
            "Step 3: Mass, weight, and hardness require tools to measure.\n"
            "Final Answer: A\n\n"
            "Now answer:\n"
            "Question: {question}\n"
            "Options:\n{choices}\n\n"
            "Reasoning:\n"
        ),
        "folio": (
            "Determine if a conclusion follows from premises.\n\n"
            "Premises: All cats are mammals. Tom is a cat.\n"
            "Conclusion: Tom is a mammal.\n"
            "Reasoning: Step 1: From 'All cats are mammals', we know the set of cats "
            "is a subset of mammals.\n"
            "Step 2: Tom is in the set of cats.\n"
            "Step 3: Therefore Tom must be in the set of mammals.\n"
            "Final Answer: True\n\n"
            "Now determine:\n"
            "Premises: {question}\n"
            "Conclusion: {conclusion}\n\n"
            "Reasoning:\n"
        ),
    },
    "explicit_steps": {
        "gsm8k": (
            "Solve the following math problem. You MUST format your solution as numbered steps:\n"
            "Step 1: ...\nStep 2: ...\n...\nFinal Answer: [number]\n\n"
            "Problem: {question}\n\nSolution:\n"
        ),
        "math": (
            "Solve the following math problem. You MUST format your solution as numbered steps:\n"
            "Step 1: ...\nStep 2: ...\n...\nFinal Answer: [answer]\n\n"
            "Problem: {question}\n\nSolution:\n"
        ),
        "strategyqa": (
            "Answer with Yes or No. You MUST format your reasoning as numbered steps:\n"
            "Step 1: ...\nStep 2: ...\n...\nFinal Answer: [Yes/No]\n\n"
            "Question: {question}\n\nReasoning:\n"
        ),
        "arc_challenge": (
            "Select the correct answer. You MUST format your reasoning as numbered steps:\n"
            "Step 1: ...\nStep 2: ...\n...\nFinal Answer: [A/B/C/D]\n\n"
            "Question: {question}\nOptions:\n{choices}\n\nReasoning:\n"
        ),
        "folio": (
            "Determine True/False/Unknown. You MUST format reasoning as numbered steps:\n"
            "Step 1: ...\nStep 2: ...\n...\nFinal Answer: [True/False/Unknown]\n\n"
            "Premises: {question}\nConclusion: {conclusion}\n\nReasoning:\n"
        ),
    },
}


def get_benchmark_config(benchmark_key: str) -> BenchmarkConfig:
    """Get benchmark configuration by key.

    Args:
        benchmark_key: Key from BENCHMARK_REGISTRY (e.g., 'gsm8k')

    Returns:
        BenchmarkConfig instance

    Raises:
        KeyError: If benchmark_key not found in registry
    """
    if benchmark_key not in BENCHMARK_REGISTRY:
        available = list(BENCHMARK_REGISTRY.keys())
        raise KeyError(
            f"Benchmark '{benchmark_key}' not found. Available: {available}"
        )
    return BENCHMARK_REGISTRY[benchmark_key]
