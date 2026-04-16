#src/metrics/casual_necessity_score.py
"""
Causal Necessity Score (CNS) Metric.

Measures how much removing or corrupting each reasoning step changes
the model's final answer:

    CNS(s_i) = 1 if removing s_i changes the answer, 0 otherwise
    (Soft version: CNS(s_i) = |P(a* | original) - P(a* | without s_i)|)

Higher CNS indicates the step is more causally necessary for the answer.
This is computed via the step deletion perturbation test.
"""

import numpy as np
from typing import Any, Dict, List, Optional

from src.utils.cot_parser import CoTParser, ParsedCoT
from src.utils.answer_extractor import AnswerExtractor
from src.utils.logger import get_logger

logger = get_logger("cns_metric")


class CausalNecessityScore:
    """Compute Causal Necessity Score via step deletion."""

    def __init__(self, threshold: float = 0.05):
        """Initialize CNS metric.

        Args:
            threshold: Minimum probability change to count as causally necessary
        """
        self.threshold = threshold
        self.cot_parser = CoTParser()
        self.answer_extractor = AnswerExtractor()

    def compute(
        self,
        inference_engine,
        prompt: str,
        parsed_cot: ParsedCoT,
        original_answer: str,
        answer_type: str,
        benchmark_key: str = "",
    ) -> Dict[str, Any]:
        """Compute CNS for each step by removing it and re-generating.

        For each step s_i:
        1. Create CoT with s_i removed
        2. Feed prompt + modified_cot to model and extract answer
        3. CNS(s_i) = 1 if answer changed, 0 otherwise

        Args:
            inference_engine: InferenceEngine instance
            prompt: Original prompt (without CoT)
            parsed_cot: Parsed CoT with steps
            original_answer: The model's original answer
            answer_type: Answer type for comparison
            benchmark_key: Benchmark identifier

        Returns:
            Dict with per-step CNS values and aggregate statistics
        """
        if not parsed_cot.steps:
            return self._empty_result()

        steps = parsed_cot.steps
        cns_values = []
        answer_changes = []

        for i in range(len(steps)):
            # Create CoT with step i removed
            modified_cot = self.cot_parser.remove_step(parsed_cot, i)

            # Create modified prompt with the remaining reasoning
            modified_prompt = (
                prompt +
                "\n\n" +
                modified_cot +
                "\n\nAnswer:"
            )

            # Generate answer with modified CoT
            try:
                result = inference_engine.generate_cot(
                    modified_prompt, max_new_tokens=256
                )
                modified_answer = self.answer_extractor.extract(
                    result["raw_output"], answer_type, benchmark_key
                )

                # Compare answers
                answer_changed = not self.answer_extractor.check_answer(
                    modified_answer, original_answer, answer_type
                )
                cns = 1.0 if answer_changed else 0.0

                cns_values.append(cns)
                answer_changes.append({
                    "step_index": i,
                    "step_text": steps[i].text[:100],
                    "original_answer": original_answer,
                    "modified_answer": modified_answer,
                    "answer_changed": answer_changed,
                })

            except Exception as e:
                logger.warning(f"CNS computation failed for step {i}: {e}")
                cns_values.append(0.0)
                answer_changes.append({
                    "step_index": i,
                    "error": str(e),
                    "answer_changed": False,
                })

        cns_array = np.array(cns_values, dtype=np.float64)
        causal_mask = cns_array > self.threshold

        return {
            "cns_values": cns_values,
            "mean_cns": float(cns_array.mean()) if len(cns_array) > 0 else 0.0,
            "max_cns": float(cns_array.max()) if len(cns_array) > 0 else 0.0,
            "num_causal_steps": int(causal_mask.sum()),
            "num_non_causal_steps": int((~causal_mask).sum()),
            "causal_ratio": (
                float(causal_mask.sum()) / len(cns_array)
                if len(cns_array) > 0 else 0.0
            ),
            "answer_changes": answer_changes,
            "num_steps": len(steps),
        }

    def compute_batch(
        self,
        inference_engine,
        examples: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Compute CNS for a batch of examples.

        Args:
            inference_engine: InferenceEngine instance
            examples: List of dicts with prompt, parsed_cot, predicted_answer, etc.

        Returns:
            List of CNS result dicts
        """
        results = []
        for i, ex in enumerate(examples):
            prompt = ex.get("prompt", "")
            parsed_cot = ex.get("parsed_cot")
            original_answer = ex.get("predicted_answer", "")
            answer_type = ex.get("answer_type", "numeric")
            benchmark_key = ex.get("benchmark", "")

            if parsed_cot is None or not original_answer:
                results.append(self._empty_result())
                continue

            try:
                result = self.compute(
                    inference_engine, prompt, parsed_cot,
                    original_answer, answer_type, benchmark_key
                )
                results.append(result)
            except Exception as e:
                logger.warning(f"CNS batch failed for example {i}: {e}")
                results.append(self._empty_result())

            if (i + 1) % 10 == 0:
                logger.info(f"CNS: Computed {i + 1}/{len(examples)}")

        return results

    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        """Return empty result."""
        return {
            "cns_values": [],
            "mean_cns": 0.0,
            "max_cns": 0.0,
            "num_causal_steps": 0,
            "num_non_causal_steps": 0,
            "causal_ratio": 0.0,
            "answer_changes": [],
            "num_steps": 0,
        }
