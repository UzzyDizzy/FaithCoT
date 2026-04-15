"""
FOLIO Parser.

Parses the FOLIO (First-Order Logic) dataset into unified format.
Dataset: yale-nlp/FOLIO
Fields: premises (str), conclusion (str), label (str: True/False/Unknown)
Splits: train (1,004), validation (204)
Note: FOLIO uses validation as the test split; no separate test set.
"""

from typing import Any, Dict, List


def parse_folio(example: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a single FOLIO example into unified format.

    FOLIO is a natural language inference dataset using first-order logic.
    Each example has premises, a conclusion, and a label
    (True/False/Unknown).

    Args:
        example: Raw example dict with 'premises', 'conclusion', 'label'

    Returns:
        Unified format dict
    """
    premises = example.get("premises", "").strip()
    conclusion = example.get("conclusion", "").strip()
    label = example.get("label", "Unknown").strip()

    # Normalize label
    label_map = {
        "True": "True",
        "true": "True",
        "False": "False",
        "false": "False",
        "Unknown": "Unknown",
        "unknown": "Unknown",
        "Uncertain": "Unknown",
        "uncertain": "Unknown",
    }
    gold_answer = label_map.get(label, label)

    # For FOLIO, the "question" is the premises and the conclusion is metadata
    # The prompt template will combine them appropriately
    question = premises

    return {
        "question": question,
        "gold_answer": gold_answer,
        "gold_cot": "",  # FOLIO doesn't come with gold CoT
        "answer_type": "nli",
        "benchmark": "folio",
        "metadata": {
            "premises": premises,
            "conclusion": conclusion,
            "raw_label": label,
        },
    }


def parse_folio_batch(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Parse a batch of FOLIO examples."""
    return [parse_folio(ex) for ex in examples]
