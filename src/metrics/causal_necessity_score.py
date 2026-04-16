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
        parsed_cot,
        original_answer: str,
        answer_type: str,
        benchmark_key: str = "",
    ):

        if not parsed_cot.steps or not original_answer:
            return self._empty_result()

        steps = parsed_cot.steps

        print("\n====== CNS DEBUG START ======")
        print("Steps:", len(steps))
        print("Answer:", original_answer)

        # 🔥 FULL CONTEXT LOGPROB
        full_context = prompt + "\n" + "\n".join(s.text for s in steps)

        lp_full = inference_engine.get_answer_log_prob(full_context, original_answer)

        if lp_full is None:
            print("[CNS DEBUG] ❌ lp_full failed")
            return self._empty_result()

        print(f"[CNS DEBUG] lp_full={lp_full:.4f}")

        cns_values = []

        for i in range(len(steps)):

            reduced_steps = [s.text for j, s in enumerate(steps) if j != i]
            reduced_context = prompt + "\n" + "\n".join(reduced_steps)

            lp_reduced = inference_engine.get_answer_log_prob(
                reduced_context,
                original_answer
            )

            if lp_reduced is None:
                print(f"[CNS DEBUG] ❌ step {i} failed")
                cns_values.append(0.0)
                continue

            delta = abs(lp_full - lp_reduced)

            cns = 1.0 if delta > self.threshold else 0.0

            print(f"[CNS DEBUG] step={i}, delta={delta:.4f}, cns={cns}")

            cns_values.append(cns)

        arr = np.array(cns_values, dtype=np.float64)

        return {
            "cns_values": cns_values,
            "mean_cns": float(arr.mean()),
            "max_cns": float(arr.max()),
            "num_causal_steps": int((arr > 0).sum()),
            "num_non_causal_steps": int((arr == 0).sum()),
            "causal_ratio": float((arr > 0).sum() / len(arr)),
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
