"""
StrategyQA Parser.

Parses the StrategyQA (Multi-hop Commonsense) dataset into unified format.
Dataset: ChilleD/StrategyQA
Fields: question (str), answer (bool), facts (list[str])
Splits: train (2,290), test (490)
"""

from typing import Any, Dict, List


def parse_strategyqa(example: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a single StrategyQA example into unified format.

    StrategyQA has boolean answers and optional decomposition facts.

    Args:
        example: Raw example dict with 'question', 'answer', and optional 'facts'

    Returns:
        Unified format dict
    """
    question = example["question"].strip()

    # StrategyQA answer is boolean (True/False)
    raw_answer = example["answer"]
    if isinstance(raw_answer, bool):
        gold_answer = "yes" if raw_answer else "no"
    elif isinstance(raw_answer, str):
        gold_answer = raw_answer.strip().lower()
        if gold_answer in ("true", "1"):
            gold_answer = "yes"
        elif gold_answer in ("false", "0"):
            gold_answer = "no"
    else:
        gold_answer = "yes" if raw_answer else "no"

    # Extract facts/decomposition if available
    facts = example.get("facts", [])
    if facts is None:
        facts = []
    decomposition = example.get("decomposition", [])
    if decomposition is None:
        decomposition = []

    # Build gold CoT from decomposition/facts
    gold_cot_parts = []
    if decomposition:
        for i, step in enumerate(decomposition):
            gold_cot_parts.append(f"Step {i + 1}: {step}")
    elif facts:
        for i, fact in enumerate(facts):
            gold_cot_parts.append(f"Fact {i + 1}: {fact}")
    gold_cot = "\n".join(gold_cot_parts)

    return {
        "question": question,
        "gold_answer": gold_answer,
        "gold_cot": gold_cot,
        "answer_type": "yes_no",
        "benchmark": "strategyqa",
        "metadata": {
            "facts": facts,
            "decomposition": decomposition,
            "raw_answer": raw_answer,
        },
    }


def parse_strategyqa_batch(
    examples: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Parse a batch of StrategyQA examples."""
    return [parse_strategyqa(ex) for ex in examples]
