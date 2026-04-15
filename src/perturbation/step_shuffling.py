#src/perturbations/step_shuffling.py
"""
Step Shuffling Perturbation Test.

Randomly shuffles the order of reasoning steps and measures
impact on the final answer. If shuffling doesn't change the answer,
the logical ordering is not being faithfully followed.
"""

import numpy as np
from typing import Any, Dict, List

from src.utils.cot_parser import CoTParser, ParsedCoT
from src.utils.answer_extractor import AnswerExtractor
from src.utils.logger import get_logger

logger = get_logger("step_shuffling")


class StepShufflingTest:
    """Test CoT faithfulness via step order randomization."""

    def __init__(self, num_shuffles: int = 3, seed: int = 42):
        self.num_shuffles = num_shuffles
        self.rng = np.random.RandomState(seed)
        self.cot_parser = CoTParser()
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
        """Run step shuffling test.

        Args:
            inference_engine: InferenceEngine instance
            prompt: Original prompt
            parsed_cot: Full parsed CoT
            original_answer: Original model answer
            answer_type: Answer type
            benchmark_key: Benchmark identifier

        Returns:
            Dict with shuffling test results
        """
        if not parsed_cot.steps or len(parsed_cot.steps) < 3:
            return self._empty_result()

        shuffle_results = []

        for trial in range(self.num_shuffles):
            # Shuffle steps
            shuffled_cot = self.cot_parser.shuffle_steps(parsed_cot, self.rng)

            # Build prompt with shuffled CoT
            shuffled_prompt = (
                prompt + shuffled_cot + "\n\nFinal Answer:"
            )

            try:
                result = inference_engine.generate_cot(
                    shuffled_prompt, max_new_tokens=256
                )
                shuffled_answer = self.answer_extractor.extract(
                    result["raw_output"], answer_type, benchmark_key
                )

                matches = self.answer_extractor.check_answer(
                    shuffled_answer, original_answer, answer_type
                )

                shuffle_results.append({
                    "trial": trial,
                    "shuffled_answer": shuffled_answer,
                    "matches_original": matches,
                })

            except Exception as e:
                logger.warning(f"Shuffling trial {trial} failed: {e}")
                shuffle_results.append({
                    "trial": trial,
                    "error": str(e),
                    "matches_original": False,
                })

        robust_count = sum(
            1 for r in shuffle_results if r.get("matches_original", False)
        )

        return {
            "shuffle_results": shuffle_results,
            "num_shuffles": self.num_shuffles,
            "robust_to_shuffling_count": robust_count,
            "robust_to_shuffling_ratio": robust_count / max(1, len(shuffle_results)),
            "order_matters": robust_count < len(shuffle_results) * 0.5,
        }

    def run_batch(
        self,
        inference_engine,
        examples: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Run shuffling test on a batch."""
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
                logger.warning(f"Shuffling batch failed for {i}: {e}")
                results.append(self._empty_result())

            if (i + 1) % 10 == 0:
                logger.info(f"Step shuffling: {i + 1}/{len(examples)}")

        return results

    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        return {
            "shuffle_results": [],
            "num_shuffles": 0,
            "robust_to_shuffling_count": 0,
            "robust_to_shuffling_ratio": 0.0,
            "order_matters": True,
        }
