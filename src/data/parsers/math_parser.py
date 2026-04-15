"""
MATH Parser.

Parses the MATH (Competition Mathematics) dataset into unified format.
Dataset: hendrycks/competition_math
Fields: problem (str), solution (str with LaTeX), level (str), type (str)
Splits: train (7,500), test (5,000)
"""

import re
from typing import Any, Dict, List


def parse_math(example: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a single MATH example into unified format.

    The MATH solution field contains LaTeX-formatted solutions.
    The final answer is typically in \\boxed{...}.

    Args:
        example: Raw example dict with 'problem', 'solution', 'level', 'type' fields

    Returns:
        Unified format dict
    """
    question = example["problem"].strip()
    raw_solution = example["solution"].strip()

    # Extract final answer from \boxed{...}
    gold_answer = ""
    boxed_matches = re.findall(r"\\boxed\{([^}]+)\}", raw_solution)
    if boxed_matches:
        gold_answer = boxed_matches[-1].strip()
    else:
        # Fallback: try "answer is" pattern
        ans_match = re.search(
            r"(?:answer is|=)\s*(.+?)(?:\.|$)", raw_solution, re.IGNORECASE
        )
        if ans_match:
            gold_answer = ans_match.group(1).strip()
        else:
            # Last resort: final line
            lines = [l for l in raw_solution.split("\n") if l.strip()]
            gold_answer = lines[-1].strip() if lines else ""

    # The CoT is everything in the solution
    gold_cot = raw_solution

    # Get difficulty level and problem type
    level = example.get("level", "Unknown")
    prob_type = example.get("type", "Unknown")

    return {
        "question": question,
        "gold_answer": gold_answer,
        "gold_cot": gold_cot,
        "answer_type": "numeric",
        "benchmark": "math",
        "metadata": {
            "level": level,
            "type": prob_type,
            "raw_solution": raw_solution,
        },
    }


def parse_math_batch(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Parse a batch of MATH examples."""
    return [parse_math(ex) for ex in examples]
