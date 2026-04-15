"""
GSM8K Parser.

Parses the GSM8K (Grade School Math 8K) dataset into unified format.
Dataset: openai/gsm8k (main configuration)
Fields: question (str), answer (str with reasoning + #### final_answer)
Splits: train (7,473), test (1,319)
"""

import re
from typing import Any, Dict, List


def parse_gsm8k(example: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a single GSM8K example into unified format.

    The GSM8K answer field contains the full solution with reasoning steps,
    followed by '#### <numeric_answer>' as the final answer marker.

    Args:
        example: Raw example dict with 'question' and 'answer' fields

    Returns:
        Unified format dict with keys:
        - question: str (problem text)
        - gold_answer: str (numeric answer only)
        - gold_cot: str (reasoning steps from the solution)
        - answer_type: str ("numeric")
        - benchmark: str ("gsm8k")
        - metadata: dict (additional info)
    """
    question = example["question"].strip()
    raw_answer = example["answer"].strip()

    # Split answer into reasoning and final numeric answer
    # GSM8K format: "reasoning text\n#### <number>"
    gold_cot = ""
    gold_answer = ""

    if "####" in raw_answer:
        parts = raw_answer.split("####")
        gold_cot = parts[0].strip()
        gold_answer = parts[1].strip()
    else:
        # Fallback: try to find the last number
        gold_cot = raw_answer
        numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", raw_answer)
        gold_answer = numbers[-1] if numbers else raw_answer

    # Normalize the numeric answer
    gold_answer = gold_answer.replace(",", "").strip()

    return {
        "question": question,
        "gold_answer": gold_answer,
        "gold_cot": gold_cot,
        "answer_type": "numeric",
        "benchmark": "gsm8k",
        "metadata": {
            "raw_answer": raw_answer,
            "num_cot_steps": len(gold_cot.split("\n")),
        },
    }


def parse_gsm8k_batch(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Parse a batch of GSM8K examples.

    Args:
        examples: List of raw example dicts

    Returns:
        List of unified format dicts
    """
    return [parse_gsm8k(ex) for ex in examples]
