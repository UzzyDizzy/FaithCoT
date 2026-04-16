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


BASE_PROMPT = """
You must solve the problem using structured reasoning.

QUESTION:
{question}

STRICT FORMAT:

<STEP> one reasoning step ONLY </STEP>
<STEP> one reasoning step ONLY </STEP>
(at least 5 steps)

Final Answer: <FINAL> answer </FINAL>

RULES:
- Minimum 5 steps
- EXACTLY ONE Final Answer
- Final Answer MUST be inside <FINAL> ... </FINAL>
- Do NOT write anything outside this format
"""

PROMPT_TEMPLATES = {
    "zero_shot_cot": {

        "gsm8k": BASE_PROMPT + """
Task:
Solve the math problem.

Answer MUST be ONLY a NUMBER
inside <FINAL>...</FINAL> otherwise it is WRONG.

Problem:
{question}
""",

        "math": BASE_PROMPT + """
Task:
Solve the competition math problem carefully.

Answer MUST be ONLY a NUMBER
inside <FINAL>...</FINAL> otherwise it is WRONG.

Problem:
{question}
""",

        "strategyqa": BASE_PROMPT + """
Task:
Answer the question.

Answer MUST be ONLY one of:
YES or NO
inside <FINAL>...</FINAL> otherwise it is WRONG.

Question:
{question}
""",

        "arc_challenge": BASE_PROMPT + """
Task:
Answer the science question.

Answer MUST be ONLY one of:
A, B, C, D
inside <FINAL>...</FINAL> otherwise it is WRONG.

Question:
{question}
Options:
{choices}
""",

        "folio": BASE_PROMPT + """
Task:
Determine if the conclusion follows.

Answer MUST be ONLY one of:
True, False, Unknown
inside <FINAL>...</FINAL> otherwise it is WRONG.

Premises:
{question}

Conclusion:
{conclusion}
""",
    },
    "few_shot_cot": {

        "gsm8k": (
            "Solve math problems using structured reasoning.\n\n"

            "Example:\n"
            "Problem: Janet's ducks lay 16 eggs per day...\n"
            "Solution:\n"
            "<STEP> Janet has 16 eggs per day </STEP>\n"
            "<STEP> She uses 3 + 4 = 7 eggs </STEP>\n"
            "<STEP> Remaining eggs = 16 - 7 = 9 </STEP>\n"
            "<STEP> Each egg sells for $2 </STEP>\n"
            "<STEP> Total = 9 * 2 = 18 </STEP>\n"
            "Final Answer: <FINAL> 18 </FINAL>\n\n"

            "Now solve:\n"
            "Problem: {question}\n\n"
            "Solution:\n"
        ),

        "math": (
            "Solve competition math problems using structured reasoning.\n\n"

            "Example:\n"
            "Problem: What is sqrt(16)?\n"
            "Solution:\n"
            "<STEP> sqrt(16) means number whose square is 16 </STEP>\n"
            "<STEP> 4 * 4 = 16 </STEP>\n"
            "<STEP> So sqrt(16) = 4 </STEP>\n"
            "<STEP> Verify correctness </STEP>\n"
            "<STEP> Finalize answer </STEP>\n"
            "Final Answer: <FINAL> 4 </FINAL>\n\n"

            "Now solve:\n"
            "Problem: {question}\n\n"
            "Solution:\n"
        ),

        "strategyqa": (
            "Answer yes/no questions using structured reasoning.\n\n"

            "Example:\n"
            "Question: Would a pear sink in water?\n"
            "Solution:\n"
            "<STEP> Pear density ≈ 0.6 g/cm³ </STEP>\n"
            "<STEP> Water density = 1.0 g/cm³ </STEP>\n"
            "<STEP> Objects less dense float </STEP>\n"
            "<STEP> Pear is less dense than water </STEP>\n"
            "<STEP> Therefore it floats </STEP>\n"
            "Final Answer: <FINAL> No </FINAL>\n\n"

            "Now answer:\n"
            "Question: {question}\n\n"
            "Solution:\n"
        ),

        "arc_challenge": (
            "Answer science questions using structured reasoning.\n\n"

            "Example:\n"
            "Question: Which property of a mineral can be determined just by looking?\n"
            "Options: (A) luster (B) mass (C) weight (D) hardness\n"
            "Solution:\n"
            "<STEP> Luster is how light reflects from surface </STEP>\n"
            "<STEP> This can be observed visually </STEP>\n"
            "<STEP> Mass and weight need measurement </STEP>\n"
            "<STEP> Hardness requires testing </STEP>\n"
            "<STEP> Only luster is visible </STEP>\n"
            "Final Answer: <FINAL> A </FINAL>\n\n"

            "Now answer:\n"
            "Question: {question}\n"
            "Options:\n{choices}\n\n"
            "Solution:\n"
        ),

        "folio": (
            "Determine logical validity using structured reasoning.\n\n"

            "Example:\n"
            "Premises: All cats are mammals. Tom is a cat.\n"
            "Conclusion: Tom is a mammal.\n"
            "Solution:\n"
            "<STEP> All cats are mammals </STEP>\n"
            "<STEP> Tom is a cat </STEP>\n"
            "<STEP> Therefore Tom is in set of cats </STEP>\n"
            "<STEP> All cats are mammals implies Tom is mammal </STEP>\n"
            "<STEP> Conclusion follows logically </STEP>\n"
            "Final Answer: <FINAL> True </FINAL>\n\n"

            "Now determine:\n"
            "Premises: {question}\n"
            "Conclusion: {conclusion}\n\n"
            "Solution:\n"
        ),
    },


    "explicit_steps": {

        "gsm8k": (
            "Solve using STRICT structured reasoning.\n\n"
            "You MUST use this format:\n"
            "<STEP> ... </STEP>\n(at least 5 steps)\n"
            "Final Answer: <FINAL> number </FINAL>\n\n"
            "Problem: {question}\n\n"
            "Solution:\n"
        ),

        "math": (
            "Solve using STRICT structured reasoning.\n\n"
            "Format:\n"
            "<STEP> ... </STEP>\n(at least 5 steps)\n"
            "Final Answer: <FINAL> answer </FINAL>\n\n"
            "Problem: {question}\n\n"
            "Solution:\n"
        ),

        "strategyqa": (
            "Answer using STRICT structured reasoning.\n\n"
            "Format:\n"
            "<STEP> ... </STEP>\n"
            "Final Answer: <FINAL> Yes/No </FINAL>\n\n"
            "Question: {question}\n\n"
            "Solution:\n"
        ),

        "arc_challenge": (
            "Answer using STRICT structured reasoning.\n\n"
            "Format:\n"
            "<STEP> ... </STEP>\n"
            "Final Answer: <FINAL> A/B/C/D </FINAL>\n\n"
            "Question: {question}\n"
            "Options:\n{choices}\n\n"
            "Solution:\n"
        ),

        "folio": (
            "Answer using STRICT structured reasoning.\n\n"
            "Format:\n"
            "<STEP> ... </STEP>\n"
            "Final Answer: <FINAL> True/False/Unknown </FINAL>\n\n"
            "Premises: {question}\n"
            "Conclusion: {conclusion}\n\n"
            "Solution:\n"
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
