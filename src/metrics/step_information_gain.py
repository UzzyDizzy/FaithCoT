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

    def _safe_context(self, prompt, steps, max_chars=2000):
        text = prompt + "\n" + "\n".join(s.text for s in steps)
        return text[-max_chars:]

    def _safe_entropy(self, inference_engine, context):
        if not context or len(context.strip()) < 10:
            return None

        try:
            result = inference_engine.get_answer_log_probs(context)
            if result is None or "entropy" not in result:
                return None
            return result["entropy"]
        except Exception:
            return None

    def compute(
        self,
        inference_engine,
        prompt: str,
        parsed_cot,
        original_answer: str,
        answer_type: str,
        benchmark_key: str = "",
    ):

        if not parsed_cot or not parsed_cot.steps:
            print("[SIG DEBUG] ❌ No parsed steps")
            return self._empty_result()

        if not original_answer:
            print("[SIG DEBUG] ❌ Empty original answer")
            return self._empty_result()

        steps = [
            s for s in parsed_cot.steps
            if s.text and len(s.text.strip()) > 5
        ]

        if not steps:
            print("[SIG DEBUG] ❌ Steps filtered out")
            return self._empty_result()

        print(f"[SIG DEBUG] Steps count: {len(steps)}")
        print(f"[SIG DEBUG] Answer: {repr(original_answer)}")

        sig_values = []

        # ---- BASELINE ----
        lp_prev = inference_engine.get_answer_log_prob(prompt, original_answer)

        if lp_prev is None:
            print("[SIG DEBUG] ❌ lp_prev is None (BASELINE FAILED)")
            return self._empty_result()

        print(f"[SIG DEBUG] lp_prev={lp_prev:.4f}")

        # ---- LOOP ----
        for i in range(len(steps)):

            context_now = prompt + "\n" + "\n".join(
                s.text for s in steps[: i + 1]
            )

            lp_now = inference_engine.get_answer_log_prob(
                context_now,
                original_answer
            )

            if lp_now is None:
                print(f"[SIG DEBUG] ❌ lp_now None at step {i}")
                sig_values.append(0.0)
                continue

            sig = max(0.0, lp_now - lp_prev)

            print(f"[SIG DEBUG] step={i}, lp_now={lp_now:.4f}, sig={sig:.4f}")

            sig_values.append(sig)
            lp_prev = lp_now

        if not sig_values:
            print("[SIG DEBUG] ❌ sig_values empty after loop")
            return self._empty_result()

        sig_array = np.array(sig_values, dtype=np.float64)
        informative_mask = sig_array > self.threshold

        print(f"[SIG DEBUG] ✅ FINAL SIG: {sig_values}")

        return {
            "sig_values": sig_values,
            "mean_sig": float(sig_array.mean()),
            "max_sig": float(sig_array.max()),
            "total_information_gain": float(sig_array.sum()),
            "num_steps": len(sig_values),
            "num_informative_steps": int(informative_mask.sum()),
            "num_noise_steps": int(len(sig_array) - informative_mask.sum()),
            "informative_ratio": float(informative_mask.sum() / len(sig_array)),
            "step_labels": [
                "informative" if m else "noise"
                for m in informative_mask
            ],
        }

    def compute_batch(self, inference_engine, examples):

        results = []

        for i, ex in enumerate(examples):

            prompt = ex.get("prompt", "")
            parsed_cot = ex.get("parsed_cot")
            original_answer = ex.get("predicted_answer", "")
            answer_type = ex.get("answer_type", "numeric")

            print(f"\n========== SIG DEBUG EXAMPLE {i} ==========")

            try:
                result = self.compute(
                    inference_engine,
                    prompt,
                    parsed_cot,
                    original_answer,
                    answer_type,
                )

                if not result["sig_values"]:
                    print("[SIG DEBUG] ❌ EMPTY SIG RETURNED")

                results.append(result)

            except Exception as e:
                print(f"[SIG DEBUG] ❌ Exception: {e}")
                results.append(self._empty_result())

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
