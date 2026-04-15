"""
ARC-Challenge Parser.

Parses the ARC-Challenge (Science Reasoning) dataset into unified format.
Dataset: allenai/ai2_arc (ARC-Challenge subset)
Fields: question (str), choices (dict with text/label lists), answerKey (str)
Splits: train (1,119), validation (299), test (1,172)
"""

from typing import Any, Dict, List


def parse_arc_challenge(example: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a single ARC-Challenge example into unified format.

    ARC-Challenge has multiple choice questions with choices dict containing
    'text' and 'label' lists.

    Args:
        example: Raw example dict with 'question', 'choices', 'answerKey'

    Returns:
        Unified format dict
    """
    question = example["question"].strip()
    answer_key = example["answerKey"].strip().upper()

    # Parse choices
    choices_data = example["choices"]
    # choices_data is a dict: {"text": [...], "label": [...]}
    choice_texts = choices_data["text"]
    choice_labels = choices_data["label"]

    # Build formatted choices string
    choices_formatted = []
    gold_answer_text = ""
    for label, text in zip(choice_labels, choice_texts):
        choices_formatted.append(f"({label}) {text}")
        if label.upper() == answer_key:
            gold_answer_text = text

    choices_str = "\n".join(choices_formatted)

    return {
        "question": question,
        "gold_answer": answer_key,
        "gold_cot": "",  # ARC doesn't come with gold CoT
        "answer_type": "multiple_choice",
        "benchmark": "arc_challenge",
        "metadata": {
            "choices_text": choice_texts,
            "choices_labels": choice_labels,
            "choices_formatted": choices_str,
            "gold_answer_text": gold_answer_text,
        },
    }


def parse_arc_challenge_batch(
    examples: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Parse a batch of ARC-Challenge examples."""
    return [parse_arc_challenge(ex) for ex in examples]
