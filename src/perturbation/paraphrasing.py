#src/perturbations/paraphrasing.py
"""
Paraphrasing Perturbation Test.

Paraphrases reasoning steps (semantically equivalent rewrites)
and checks if the model's answer remains stable. If paraphrasing
changes the answer, the model is sensitive to surface form.
"""

import re
import random
from typing import Any, Dict, List, Optional

from src.utils.cot_parser import ParsedCoT, ReasoningStep
from src.utils.answer_extractor import AnswerExtractor
from src.utils.logger import get_logger

logger = get_logger("paraphrasing")


class ParaphrasingTest:
    """Test CoT sensitivity to surface-level paraphrasing."""

    # Simple rule-based paphrase substitutions (deterministic, no API needed)
    PARAPHRASE_RULES = [
        (r"\bFirst\b", "To begin"),
        (r"\bSecond\b", "Next"),
        (r"\bThird\b", "After that"),
        (r"\bTherefore\b", "Thus"),
        (r"\bHence\b", "Consequently"),
        (r"\bWe know that\b", "It is known that"),
        (r"\bwe get\b", "we obtain"),
        (r"\bwe have\b", "we find"),
        (r"\bSo\b", "Therefore"),
        (r"\bThus\b", "Hence"),
        (r"\bwhich gives us\b", "resulting in"),
        (r"\bthis means\b", "this implies"),
        (r"\bwe can see that\b", "it follows that"),
        (r"\bLet's\b", "Let us"),
        (r"\bcan't\b", "cannot"),
        (r"\bdon't\b", "do not"),
        (r"\bwon't\b", "will not"),
        (r"\bisn't\b", "is not"),
        (r"\baren't\b", "are not"),
        (r"\b(\d+) multiplied by (\d+)\b", r"\1 times \2"),
        (r"\b(\d+) times (\d+)\b", r"\1 multiplied by \2"),
        (r"\bsubtract\b", "take away"),
        (r"\badd\b", "sum"),
    ]

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.answer_extractor = AnswerExtractor()

    def run(
        self,
        inference_engine,
        prompt: str,
        parsed_cot: ParsedCoT,
        original_answer: str,
        answer_type: str,
        benchmark_key: str = "",
    ) -> Dict[str, Any]:
        """Run paraphrasing test on each step.

        Args:
            inference_engine: InferenceEngine instance
            prompt: Original prompt
            parsed_cot: Full parsed CoT
            original_answer: Original model answer
            answer_type: Answer type
            benchmark_key: Benchmark identifier

        Returns:
            Dict with paraphrasing results
        """
        if not parsed_cot.steps or len(parsed_cot.steps) < 2:
            return self._empty_result()

        steps = parsed_cot.steps

        # Paraphrase all steps
        paraphrased_steps = []
        changes_made = 0
        for step in steps:
            para = self._paraphrase_step(step.text)
            paraphrased_steps.append(para)
            if para != step.text:
                changes_made += 1

        if changes_made == 0:
            return {
                "paraphrased_answer": original_answer,
                "answer_changed": False,
                "changes_made": 0,
                "is_surface_sensitive": False,
                "details": "No paraphrasing opportunities found",
            }

        # Build prompt with paraphrased CoT
        paraphrased_cot = "\n".join(paraphrased_steps)
        para_prompt = prompt + paraphrased_cot + "\n\nFinal Answer:"

        try:
            result = inference_engine.generate_cot(
                para_prompt, max_new_tokens=256
            )
            para_answer = self.answer_extractor.extract(
                result["raw_output"], answer_type, benchmark_key
            )

            answer_changed = not self.answer_extractor.check_answer(
                para_answer, original_answer, answer_type
            )

            return {
                "paraphrased_answer": para_answer,
                "answer_changed": answer_changed,
                "changes_made": changes_made,
                "is_surface_sensitive": answer_changed,
                "details": f"Paraphrased {changes_made} steps",
            }

        except Exception as e:
            logger.warning(f"Paraphrasing test failed: {e}")
            return self._empty_result()

    def _paraphrase_step(self, text: str) -> str:
        """Apply deterministic rule-based paraphrasing to a step."""
        result = text
        # Randomly select 2-3 rules to apply (for variety)
        rules = list(self.PARAPHRASE_RULES)
        self.rng.shuffle(rules)
        for pattern, replacement in rules[:3]:
            result = re.sub(pattern, replacement, result, count=1, flags=re.IGNORECASE)
        return result

    def run_batch(
        self,
        inference_engine,
        examples: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Run paraphrasing test on a batch."""
        results = []
        for i, ex in enumerate(examples):
            try:
                result = self.run(
                    inference_engine,
                    prompt=ex["prompt"],
                    parsed_cot=ex["parsed_cot"],
                    original_answer=ex["predicted_answer"],
                    answer_type=ex["answer_type"],
                    benchmark_key=ex.get("benchmark", ""),
                )
                results.append(result)
            except Exception as e:
                logger.warning(f"Paraphrasing batch failed for {i}: {e}")
                results.append(self._empty_result())

            if (i + 1) % 10 == 0:
                logger.info(f"Paraphrasing: {i + 1}/{len(examples)}")

        return results

    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        return {
            "paraphrased_answer": None,
            "answer_changed": False,
            "changes_made": 0,
            "is_surface_sensitive": False,
            "details": "",
        }
