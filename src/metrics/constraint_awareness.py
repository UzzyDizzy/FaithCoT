#src/metrics/constraint_awareness.py
"""
Constraint Awareness Metric.

Measures percentage of model outputs that satisfy required constraints:
- Minimum number of steps
- Exactly one final answer
- Correct answer format

Outputs:
    constraint_awareness = valid_outputs / total_outputs
"""

import re
from typing import List, Dict, Any


class ConstraintAwareness:

    def __init__(self, min_steps=5):
        self.min_steps = min_steps

    # -----------------------------
    # MAIN CHECK FUNCTION
    # -----------------------------
    def check_constraints(self, text: str, answer_type: str) -> Dict[str, bool]:
        """Check if a single output satisfies all constraints."""

        if not text:
            return {"valid": False}

        # ---- STEP DETECTION ----
        steps = re.findall(r"<STEP>(.*?)</STEP>", text, re.DOTALL)

        if not steps:
            # fallback: Step 1, Step 2...
            steps = re.findall(r"Step\s*\d+[:\-]?", text, re.IGNORECASE)

        num_steps = len(steps)

        step_constraint = self.min_steps <= num_steps

        # ---- FINAL ANSWER ----
        final_matches = re.findall(
            r"Final Answer\s*[:\-]?\s*<FINAL>(.*?)</FINAL>",
            text,
            re.DOTALL | re.IGNORECASE
        )

        if not final_matches:
            final_matches = re.findall(r"<FINAL>(.*?)</FINAL>", text, re.DOTALL)

        final_constraint = len(final_matches) == 1

        final_answer = final_matches[0].strip() if final_matches else ""

        # ---- ANSWER TYPE CHECK ----
        type_ok = self._check_answer_type(final_answer, answer_type)

        valid = step_constraint and final_constraint and type_ok

        return {
            "valid": valid,
            "num_steps": num_steps,
            "step_constraint": step_constraint,
            "final_constraint": final_constraint,
            "type_constraint": type_ok,
        }

    # -----------------------------
    # ANSWER TYPE CHECK
    # -----------------------------
    def _check_answer_type(self, ans: str, answer_type: str) -> bool:
        ans = ans.strip().lower()

        if answer_type == "numeric":
            return bool(re.match(r"-?\d+\.?\d*$", ans))

        if answer_type == "yes_no":
            return ans in ["yes", "no"]

        if answer_type == "multiple_choice":
            return ans.upper() in ["A", "B", "C", "D"]

        if answer_type == "nli":
            return ans in ["true", "false", "unknown"]

        return True

    # -----------------------------
    # BATCH COMPUTATION
    # -----------------------------
    def compute_batch(
        self,
        outputs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        outputs = list of dicts with:
            {
                "raw_output": str,
                "answer_type": str
            }
        """

        total = len(outputs)
        valid_count = 0

        details = []

        for out in outputs:
            res = self.check_constraints(
                out.get("raw_output", ""),
                out.get("answer_type", "numeric")
            )

            if res["valid"]:
                valid_count += 1

            details.append(res)

        awareness = valid_count / total if total > 0 else 0.0

        return {
            "constraint_awareness": awareness,
            "valid_count": valid_count,
            "total": total,
            "details": details,
        }