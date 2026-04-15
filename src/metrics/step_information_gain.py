#src/metrics/step_information_gain.py
"""
Step Information Gain (SIG) Metric.

Measures the conditional mutual information between each reasoning step
and the final answer, given all previous steps:

    SIG(s_i) = I(A; S_i | S_{<i}) = H(A | S_{<i}) - H(A | S_{<=i})

Where:
    - A is the answer distribution
    - S_i is the i-th reasoning step
    - H(A | context) is the entropy of the answer distribution given context

Higher SIG indicates the step provides more information about the answer.
"""

import numpy as np
from typing import Any, Dict, List, Optional

from src.utils.cot_parser import ParsedCoT
from src.utils.logger import get_logger

logger = get_logger("sig_metric")


class StepInformationGain:
    """Compute Step Information Gain for each reasoning step."""

    def __init__(self, threshold: float = 0.01):
        """Initialize SIG metric.

        Args:
            threshold: Minimum SIG value to consider a step informative
        """
        self.threshold = threshold

    def compute(
        self,
        inference_engine,
        prompt: str,
        parsed_cot: ParsedCoT,
    ) -> Dict[str, Any]:
        """Compute SIG for each step in a CoT.

        For each step s_i, we compute:
        1. H(A | prompt + s_1 + ... + s_{i-1}) — entropy BEFORE this step
        2. H(A | prompt + s_1 + ... + s_i) — entropy AFTER this step
        3. SIG(s_i) = H_before - H_after

        Args:
            inference_engine: InferenceEngine instance with loaded model
            prompt: Original prompt
            parsed_cot: Parsed CoT with individual steps

        Returns:
            Dict with per-step SIG values and aggregate statistics
        """
        if not parsed_cot.steps:
            return self._empty_result()

        steps = parsed_cot.steps
        entropies = []
        sig_values = []

        # Compute entropy with just the prompt (no reasoning)
        context = prompt
        result = inference_engine.get_answer_log_probs(context)
        h_initial = result["entropy"]
        entropies.append(h_initial)

        # Compute entropy after each step
        for i, step in enumerate(steps):
            context = prompt + "\n".join(s.text for s in steps[: i + 1])
            result = inference_engine.get_answer_log_probs(context)
            h_after = result["entropy"]
            entropies.append(h_after)

            # SIG = entropy reduction from adding this step
            sig = max(0.0, entropies[i] - h_after)
            sig_values.append(sig)

        sig_array = np.array(sig_values, dtype=np.float64)

        # Classify steps as informative or noise
        informative_mask = sig_array > self.threshold
        num_informative = int(informative_mask.sum())
        num_noise = len(sig_array) - num_informative

        return {
            "sig_values": sig_values,
            "entropies": entropies,
            "mean_sig": float(sig_array.mean()) if len(sig_array) > 0 else 0.0,
            "max_sig": float(sig_array.max()) if len(sig_array) > 0 else 0.0,
            "total_information_gain": float(sig_array.sum()),
            "num_steps": len(steps),
            "num_informative_steps": num_informative,
            "num_noise_steps": num_noise,
            "informative_ratio": (
                num_informative / len(sig_array) if len(sig_array) > 0 else 0.0
            ),
            "step_labels": [
                "informative" if m else "noise" for m in informative_mask
            ],
        }

    def compute_batch(
        self,
        inference_engine,
        examples: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Compute SIG for a batch of examples.

        Args:
            inference_engine: InferenceEngine instance
            examples: List of dicts with 'prompt' and 'parsed_cot'

        Returns:
            List of SIG result dicts
        """
        results = []
        for i, ex in enumerate(examples):
            prompt = ex.get("prompt", "")
            parsed_cot = ex.get("parsed_cot")

            if parsed_cot is None:
                results.append(self._empty_result())
                continue

            try:
                result = self.compute(inference_engine, prompt, parsed_cot)
                results.append(result)
            except Exception as e:
                logger.warning(f"SIG computation failed for example {i}: {e}")
                results.append(self._empty_result())

            if (i + 1) % 20 == 0:
                logger.info(f"SIG: Computed {i + 1}/{len(examples)}")

        return results

    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        """Return empty result for examples with no steps."""
        return {
            "sig_values": [],
            "entropies": [],
            "mean_sig": 0.0,
            "max_sig": 0.0,
            "total_information_gain": 0.0,
            "num_steps": 0,
            "num_informative_steps": 0,
            "num_noise_steps": 0,
            "informative_ratio": 0.0,
            "step_labels": [],
        }
