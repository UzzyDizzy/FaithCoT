#src/metrics/failure_taxonomy.py
"""
Failure Taxonomy Classifier.

Classifies CoT outputs into 6 failure categories:
F1: Post-hoc Rationalization (CoT doesn't causally drive answer)
F2: Invalid Reasoning Steps (logical errors)
F3: Redundant Exploration (repeated paths)
F4: Incorrect Backtracking (inconsistent state restoration)
F5: Distribution-Dependent Brittleness (OOD failures)
F6: Hallucinated Conclusions (unsupported by reasoning)
"""

import re
import numpy as np
from collections import Counter
from typing import Any, Dict, List, Optional, Set

from src.utils.cot_parser import ParsedCoT
from src.utils.logger import get_logger

logger = get_logger("failure_taxonomy")


class FailureTaxonomyClassifier:
    """Classify CoT outputs into failure categories."""

    def __init__(self):
        """Initialize the taxonomy classifier."""
        # N-gram overlap threshold for redundancy detection
        self.redundancy_ngram_size = 3
        self.redundancy_threshold = 0.3

        # Backtracking inconsistency patterns
        self.backtrack_markers = re.compile(
            r"(?:wait|actually|no,|let me reconsider|correction|"
            r"going back|re-think|I was wrong|let me re-examine)",
            re.IGNORECASE,
        )

        # Hallucination indicators
        self.unsupported_conclusion_markers = re.compile(
            r"(?:obviously|clearly|it is evident|as we all know|"
            r"as everyone knows|without doubt|it goes without saying)",
            re.IGNORECASE,
        )

    def classify(
        self,
        parsed_cot: ParsedCoT,
        sig_result: Optional[Dict] = None,
        cns_result: Optional[Dict] = None,
        rfi_result: Optional[Dict] = None,
        is_correct: bool = True,
        early_answering_same: bool = False,
    ) -> Dict[str, Any]:
        """Classify a CoT into failure categories.

        Multiple failures can co-occur. Returns a list of active failures.

        Args:
            parsed_cot: Parsed CoT
            sig_result: SIG metric result (optional)
            cns_result: CNS metric result (optional)
            rfi_result: RFI metric result (optional)
            is_correct: Whether the final answer is correct
            early_answering_same: Whether early answering gives same answer

        Returns:
            Dict with failure classifications
        """
        failures = {}
        steps = parsed_cot.steps

        # F1: Post-hoc Rationalization
        f1 = self._detect_post_hoc_rationalization(
            cns_result, early_answering_same
        )
        failures["F1_post_hoc_rationalization"] = f1

        # F2: Invalid Reasoning Steps
        f2 = self._detect_invalid_reasoning(steps)
        failures["F2_invalid_reasoning"] = f2

        # F3: Redundant Exploration
        f3 = self._detect_redundant_exploration(steps)
        failures["F3_redundant_exploration"] = f3

        # F4: Incorrect Backtracking
        f4 = self._detect_incorrect_backtracking(steps)
        failures["F4_incorrect_backtracking"] = f4

        # F5: Distribution-Dependent Brittleness (requires separate OOD test)
        f5 = {"detected": False, "confidence": 0.0, "details": "Requires OOD comparison"}
        failures["F5_distribution_brittleness"] = f5

        # F6: Hallucinated Conclusions
        f6 = self._detect_hallucinated_conclusions(steps, sig_result)
        failures["F6_hallucinated_conclusions"] = f6

        # Summary
        active_failures = [
            k for k, v in failures.items() if v.get("detected", False)
        ]
        primary_failure = active_failures[0] if active_failures else "none"

        return {
            "failures": failures,
            "active_failures": active_failures,
            "num_failures": len(active_failures),
            "primary_failure": primary_failure,
            "is_faithful": len(active_failures) == 0,
            "is_correct": is_correct,
        }

    def classify_batch(
        self,
        examples: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Classify a batch of examples.

        Args:
            examples: List of dicts with parsed_cot, metrics, etc.

        Returns:
            List of classification results
        """
        results = []
        for ex in examples:
            parsed_cot = ex.get("parsed_cot")
            if parsed_cot is None:
                results.append(self._empty_result())
                continue

            result = self.classify(
                parsed_cot=parsed_cot,
                sig_result=ex.get("sig_result"),
                cns_result=ex.get("cns_result"),
                rfi_result=ex.get("rfi_result"),
                is_correct=ex.get("is_correct", True),
                early_answering_same=ex.get("early_answering_same", False),
            )
            results.append(result)

        return results

    def aggregate_taxonomy(
        self, classification_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate failure taxonomy across examples.

        Args:
            classification_results: List of classify() outputs

        Returns:
            Aggregate statistics
        """
        n = len(classification_results)
        if n == 0:
            return {}

        failure_counts = Counter()
        primary_counts = Counter()

        for result in classification_results:
            for failure in result.get("active_failures", []):
                failure_counts[failure] += 1
            primary_counts[result.get("primary_failure", "none")] += 1

        return {
            "n_examples": n,
            "failure_counts": dict(failure_counts),
            "failure_rates": {k: v / n for k, v in failure_counts.items()},
            "primary_failure_distribution": dict(primary_counts),
            "faithful_count": sum(
                1 for r in classification_results if r.get("is_faithful", False)
            ),
            "faithful_rate": sum(
                1 for r in classification_results if r.get("is_faithful", False)
            ) / n,
        }

    def _detect_post_hoc_rationalization(
        self,
        cns_result: Optional[Dict],
        early_answering_same: bool,
    ) -> Dict:
        """Detect F1: Post-hoc Rationalization."""
        detected = False
        confidence = 0.0
        details = ""

        if early_answering_same:
            detected = True
            confidence = 0.9
            details = "Early answering produces same answer — reasoning not needed"

        if cns_result:
            causal_ratio = cns_result.get("causal_ratio", 1.0)
            if causal_ratio < 0.1:
                detected = True
                confidence = max(confidence, 0.8)
                details += "; Very low causal ratio — no step changes answer"

        return {"detected": detected, "confidence": confidence, "details": details}

    def _detect_invalid_reasoning(self, steps) -> Dict:
        """Detect F2: Invalid Reasoning Steps.

        Checks for common logical error patterns.
        """
        invalid_count = 0
        invalid_details = []

        for i, step in enumerate(steps):
            text = step.text.lower()

            # Check for self-contradictions within a step
            if self._has_contradiction(text):
                invalid_count += 1
                invalid_details.append(f"Step {i}: Self-contradiction detected")

            # Check for mathematical errors (simple patterns)
            math_error = self._check_math_errors(step.text)
            if math_error:
                invalid_count += 1
                invalid_details.append(f"Step {i}: {math_error}")

        detected = invalid_count > 0
        confidence = min(1.0, invalid_count / max(1, len(steps)))

        return {
            "detected": detected,
            "confidence": confidence,
            "details": "; ".join(invalid_details) if invalid_details else "",
            "invalid_count": invalid_count,
        }

    def _detect_redundant_exploration(self, steps) -> Dict:
        """Detect F3: Redundant Exploration via n-gram overlap."""
        if len(steps) < 2:
            return {"detected": False, "confidence": 0.0, "details": ""}

        # Compute pairwise n-gram overlap
        step_ngrams = []
        for step in steps:
            words = step.text.lower().split()
            ngrams = set()
            for j in range(len(words) - self.redundancy_ngram_size + 1):
                ng = tuple(words[j : j + self.redundancy_ngram_size])
                ngrams.add(ng)
            step_ngrams.append(ngrams)

        redundant_pairs = []
        for i in range(len(step_ngrams)):
            for j in range(i + 1, len(step_ngrams)):
                if not step_ngrams[i] or not step_ngrams[j]:
                    continue
                overlap = len(step_ngrams[i] & step_ngrams[j])
                max_size = max(len(step_ngrams[i]), len(step_ngrams[j]))
                ratio = overlap / max_size if max_size > 0 else 0.0
                if ratio > self.redundancy_threshold:
                    redundant_pairs.append((i, j, ratio))

        detected = len(redundant_pairs) > 0
        confidence = min(1.0, len(redundant_pairs) / max(1, len(steps)))

        return {
            "detected": detected,
            "confidence": confidence,
            "details": f"{len(redundant_pairs)} redundant step pairs found",
            "redundant_pairs": redundant_pairs,
        }

    def _detect_incorrect_backtracking(self, steps) -> Dict:
        """Detect F4: Incorrect Backtracking."""
        backtrack_steps = []
        for i, step in enumerate(steps):
            if self.backtrack_markers.search(step.text):
                backtrack_steps.append(i)

        if not backtrack_steps:
            return {"detected": False, "confidence": 0.0, "details": "No backtracking found"}

        # Check for back-to-back backtracking (potential inconsistency)
        consecutive_backtracks = 0
        for i in range(1, len(backtrack_steps)):
            if backtrack_steps[i] - backtrack_steps[i - 1] == 1:
                consecutive_backtracks += 1

        detected = consecutive_backtracks > 0
        confidence = min(1.0, consecutive_backtracks / max(1, len(backtrack_steps)))

        return {
            "detected": detected,
            "confidence": confidence,
            "details": (
                f"{len(backtrack_steps)} backtracking steps, "
                f"{consecutive_backtracks} consecutive"
            ),
            "backtrack_indices": backtrack_steps,
        }

    def _detect_hallucinated_conclusions(
        self, steps, sig_result: Optional[Dict]
    ) -> Dict:
        """Detect F6: Hallucinated Conclusions."""
        detected = False
        confidence = 0.0
        details = ""

        # Check for unsupported conclusion markers
        conclusion_steps = [s for s in steps if s.step_type == "conclusion"]
        for step in conclusion_steps:
            if self.unsupported_conclusion_markers.search(step.text):
                detected = True
                confidence = 0.6
                details = "Conclusion uses unsupported assertion markers"

        # If SIG available, check if conclusion step has very low SIG
        if sig_result and sig_result.get("sig_values"):
            sig_values = sig_result["sig_values"]
            if len(sig_values) > 0:
                last_sig = sig_values[-1]
                if last_sig < 0.001 and len(sig_values) > 2:
                    detected = True
                    confidence = max(confidence, 0.5)
                    details += "; Final step provides near-zero information gain"

        return {"detected": detected, "confidence": confidence, "details": details}

    @staticmethod
    def _has_contradiction(text: str) -> bool:
        """Simple contradiction detection within a text."""
        # Check for "X is Y" followed by "X is not Y" patterns
        affirmatives = re.findall(r"(\w+)\s+is\s+(\w+)", text)
        negatives = re.findall(r"(\w+)\s+is\s+not\s+(\w+)", text)

        for subj_a, obj_a in affirmatives:
            for subj_n, obj_n in negatives:
                if subj_a == subj_n and obj_a == obj_n:
                    return True
        return False

    @staticmethod
    def _check_math_errors(text: str) -> Optional[str]:
        """Check for obvious arithmetic errors in a step."""
        # Find simple "a + b = c" or "a * b = c" patterns
        patterns = [
            (r"(\d+)\s*\+\s*(\d+)\s*=\s*(\d+)", lambda a, b: a + b),
            (r"(\d+)\s*-\s*(\d+)\s*=\s*(\d+)", lambda a, b: a - b),
            (r"(\d+)\s*[*×]\s*(\d+)\s*=\s*(\d+)", lambda a, b: a * b),
        ]

        for pattern, op in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    a, b, c = int(match[0]), int(match[1]), int(match[2])
                    if op(a, b) != c:
                        return f"Arithmetic error: {a} op {b} = {c} (should be {op(a, b)})"
                except (ValueError, ZeroDivisionError):
                    continue
        return None

    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        """Return empty result."""
        return {
            "failures": {},
            "active_failures": [],
            "num_failures": 0,
            "primary_failure": "none",
            "is_faithful": True,
            "is_correct": True,
        }
