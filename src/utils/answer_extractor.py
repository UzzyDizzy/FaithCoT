#src/utils/answer_extractor.py
"""
Answer Extractor.

Extracts and normalizes final answers from model outputs for each benchmark type.
Handles numeric, yes/no, multiple choice, and NLI answer formats.
"""

import re
from typing import Optional


class AnswerExtractor:
    """Extract and normalize final answers from model outputs."""

    # =========================================================
    # MAIN ENTRY
    # =========================================================
    def extract(
        self,
        raw_output: str,
        answer_type: str,
        benchmark_key: str = "",
    ) -> Optional[str]:

        if not raw_output:
            return None

        extractors = {
            "numeric": self._extract_numeric,
            "yes_no": self._extract_yes_no,
            "multiple_choice": self._extract_multiple_choice,
            "nli": self._extract_nli,
        }

        extractor = extractors.get(answer_type, self._extract_generic)
        return extractor(raw_output, benchmark_key)

    # =========================================================
    # FINAL TAG (CORE)
    # =========================================================
    def _extract_final_tag(self, text: str) -> Optional[str]:
        """Extract Final Answer: <FINAL> ... </FINAL>"""

        match = re.search(
            r"Final Answer\s*[:\-]?\s*<FINAL>(.*?)</FINAL>",
            text,
            re.IGNORECASE | re.DOTALL
        )
        if match:
            return match.group(1).strip()

        # fallback
        match = re.search(r"<FINAL>(.*?)</FINAL>", text, re.DOTALL)
        if match:
            return match.group(1).strip()

        return None

    # =========================================================
    # NUMERIC
    # =========================================================
    def _extract_numeric(self, text: str, benchmark_key: str = "") -> Optional[str]:

        final = self._extract_final_tag(text)
        if final:
            return self._normalize_numeric(final)

        numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
        if numbers:
            return self._normalize_numeric(numbers[-1])

        return None

    # =========================================================
    # YES / NO
    # =========================================================
    def _extract_yes_no(self, text: str, benchmark_key: str = "") -> Optional[str]:

        final = self._extract_final_tag(text)
        if final:
            return final.strip().lower()

        yes_positions = [m.start() for m in re.finditer(r"\byes\b", text, re.IGNORECASE)]
        no_positions = [m.start() for m in re.finditer(r"\bno\b", text, re.IGNORECASE)]

        if yes_positions and no_positions:
            return "yes" if yes_positions[-1] > no_positions[-1] else "no"
        if yes_positions:
            return "yes"
        if no_positions:
            return "no"

        return None

    # =========================================================
    # MCQ
    # =========================================================
    def _extract_multiple_choice(self, text: str, benchmark_key: str = "") -> Optional[str]:

        final = self._extract_final_tag(text)
        if final:
            return final.upper()

        match = re.findall(r"\b([A-D])\b", text)
        if match:
            return match[-1].upper()

        return None

    # =========================================================
    # NLI
    # =========================================================
    def _extract_nli(self, text: str, benchmark_key: str = "") -> Optional[str]:

        final = self._extract_final_tag(text)
        if final:
            return self._normalize_nli(final)

        text = text.lower()
        if "true" in text:
            return "True"
        if "false" in text:
            return "False"
        if "unknown" in text:
            return "Unknown"

        return None

    # =========================================================
    # GENERIC
    # =========================================================
    def _extract_generic(self, text: str, benchmark_key: str = "") -> Optional[str]:

        final = self._extract_final_tag(text)
        if final:
            return final

        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        return lines[-1] if lines else None

    # =========================================================
    # NORMALIZATION
    # =========================================================
    @staticmethod
    def _normalize_numeric(value: str) -> str:

        value = value.strip()

        value = re.sub(r"\\\[|\\\]|\\\(|\\\)", "", value)
        value = re.sub(r"[*_`$]", "", value)
        value = re.sub(r"[^\d\.\-]", " ", value)
        value = " ".join(value.split())

        numbers = re.findall(r"-?\d+\.?\d*", value)
        if not numbers:
            return value.strip()

        value = numbers[0]

        try:
            num = float(value)
            if num == int(num):
                return str(int(num))
            return str(num)
        except:
            return value

    @staticmethod
    def _normalize_nli(label: str) -> str:

        label = label.strip().lower()
        if label in ("true", "entailment"):
            return "True"
        if label in ("false", "contradiction"):
            return "False"
        if label in ("unknown", "neutral"):
            return "Unknown"
        return label.capitalize()

    # =========================================================
    # CHECK ANSWER
    # =========================================================
    @staticmethod
    def check_answer(predicted, gold, answer_type):

        if predicted is None:
            return False

        pred = predicted.strip().lower()
        gold = gold.strip().lower()

        if answer_type == "numeric":
            try:
                return abs(float(pred) - float(gold)) < 1e-6
            except:
                return pred == gold

        if answer_type == "yes_no":

            def normalize(x):
                x = str(x).strip().lower()
                if x in ["yes", "true", "1"]:
                    return "yes"
                if x in ["no", "false", "0"]:
                    return "no"
                return x

            pred_n = normalize(predicted)
            gold_n = normalize(gold)

            return pred_n == gold_n

        if answer_type == "multiple_choice":
            return pred.upper() == gold.upper()

        if answer_type == "nli":
            return pred == gold

        return pred == gold