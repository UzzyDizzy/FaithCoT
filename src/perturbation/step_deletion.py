#src/perturbations/step_deletion.py
"""
Step Deletion Perturbation Test.

Removes individual reasoning steps and measures impact on the final answer.
This is the core test for the Causal Necessity Score (CNS) metric.
"""

from typing import Any, Dict, List

from src.utils.cot_parser import CoTParser, ParsedCoT
from src.utils.answer_extractor import AnswerExtractor
from src.utils.logger import get_logger

logger = get_logger("step_deletion")


class StepDeletionTest:
    """Test CoT faithfulness via individual step removal."""

    def __init__(self):
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
        """Remove each step individually and check answer change.

        Args:
            inference_engine: InferenceEngine instance
            prompt: Original prompt
            parsed_cot: Full parsed CoT
            original_answer: Original model answer
            answer_type: Answer type
            benchmark_key: Benchmark identifier

        Returns:
            Dict with per-step deletion results
        """
        if not parsed_cot.steps:
            return self._empty_result()

        steps = parsed_cot.steps
        deletion_results = []

        for i in range(len(steps)):
            modified_cot = self.cot_parser.remove_step(parsed_cot, i)
            modified_prompt = (
                prompt + modified_cot + "\n\nFinal Answer:"
            )

            try:
                result = inference_engine.generate_cot(
                    modified_prompt, max_new_tokens=256
                )
                new_answer = self.answer_extractor.extract(
                    result["raw_output"], answer_type, benchmark_key
                )

                answer_changed = not self.answer_extractor.check_answer(
                    new_answer, original_answer, answer_type
                )

                deletion_results.append({
                    "deleted_step": i,
                    "step_text": steps[i].text[:100],
                    "step_type": steps[i].step_type,
                    "new_answer": new_answer,
                    "answer_changed": answer_changed,
                })

            except Exception as e:
                logger.warning(f"Step deletion failed at step {i}: {e}")
                deletion_results.append({
                    "deleted_step": i,
                    "error": str(e),
                    "answer_changed": False,
                })

        changed_count = sum(
            1 for r in deletion_results if r.get("answer_changed", False)
        )

        return {
            "deletion_results": deletion_results,
            "num_steps": len(steps),
            "steps_causing_change": changed_count,
            "causal_step_ratio": changed_count / max(1, len(deletion_results)),
            "most_causal_step": (
                max(
                    (r for r in deletion_results if r.get("answer_changed")),
                    key=lambda r: r["deleted_step"],
                    default=None,
                )
            ),
        }

    def run_batch(
        self,
        inference_engine,
        examples: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Run step deletion on a batch."""
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
                logger.warning(f"Step deletion batch failed for {i}: {e}")
                results.append(self._empty_result())

            if (i + 1) % 10 == 0:
                logger.info(f"Step deletion: {i + 1}/{len(examples)}")

        return results

    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        return {
            "deletion_results": [],
            "num_steps": 0,
            "steps_causing_change": 0,
            "causal_step_ratio": 0.0,
            "most_causal_step": None,
        }
