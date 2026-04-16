#src/utils/answer_extractor.py
"""
Answer Extractor.

Extracts and normalizes final answers from model outputs for each benchmark type.
Handles numeric, yes/no, multiple choice, and NLI answer formats.
"""

import re
from typing import Optional, Union


class AnswerExtractor:
    """Extract and normalize final answers from model outputs."""

    def extract(
        self,
        raw_output: str,
        answer_type: str,
        benchmark_key: str = "",
    ) -> Optional[str]:
        """Extract the final answer from raw model output.

        Args:
            raw_output: Full model output text
            answer_type: One of "numeric", "yes_no", "multiple_choice", "nli"
            benchmark_key: Benchmark identifier for format-specific handling

        Returns:
            Extracted answer string, or None if extraction fails
        """
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

    def _extract_numeric(self, text: str, benchmark_key: str = "") -> Optional[str]:
        """Extract numeric answer from math problems."""

        # Priority 0: Final Answer (ADD THIS BLOCK)
        final_match = re.search(
            r"Final Answer\s*[:=]\s*(-?\d+\.?\d*)",
            text,
            re.IGNORECASE,
        )
        if final_match:
            return self._normalize_numeric(final_match.group(1))

        # Priority 1: \boxed{...}
        boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
        if boxed:
            return self._normalize_numeric(boxed[-1])

        # Priority 2: #### <number> (GSM8K format)
        hash_match = re.search(r"####\s*(.+?)(?:\n|$)", text)
        if hash_match:
            return self._normalize_numeric(hash_match.group(1).strip())

        # Priority 3: "Final Answer: <number>"
        #final_match = re.search(
            #r"(?:Final Answer|The answer is|Answer)\s*[:=]\s*(.+?)(?:\n|$)",
            #text,
            #re.IGNORECASE,
        #)
        #if final_match:
           # return self._normalize_numeric(final_match.group(1).strip())

        # FINAL fallback: last token if it's a number-like string
        tokens = text.strip().split()
        if tokens:
            last = tokens[-1]
            if re.match(r"-?\d+\.?\d*", last):
                return self._normalize_numeric(last)

        # Priority 4: Last number in the text
        numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
        if numbers:
            return self._normalize_numeric(numbers[-1])

        return None

        

    def _extract_yes_no(self, text: str, benchmark_key: str = "") -> Optional[str]:
        """Extract yes/no answer."""
        # Check for explicit final answer
        final_match = re.search(
            r"(?:Final Answer|The answer is|Answer)\s*[:=]\s*(yes|no)",
            text,
            re.IGNORECASE,
        )
        if final_match:
            return final_match.group(1).strip().lower()

        # Check last line for yes/no
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        if lines:
            last = lines[-1].lower()
            if "yes" in last and "no" not in last:
                return "yes"
            if "no" in last and "yes" not in last:
                return "no"

        # Search entire text (last occurrence)
        yes_positions = [m.start() for m in re.finditer(r"\byes\b", text, re.IGNORECASE)]
        no_positions = [m.start() for m in re.finditer(r"\bno\b", text, re.IGNORECASE)]

        if yes_positions and no_positions:
            return "yes" if yes_positions[-1] > no_positions[-1] else "no"
        if yes_positions:
            return "yes"
        if no_positions:
            return "no"

        return None

    def _extract_multiple_choice(
        self, text: str, benchmark_key: str = ""
    ) -> Optional[str]:
        """Extract multiple choice answer (A, B, C, D, etc.)."""
        # Check for explicit final answer
        final_match = re.search(
            r"(?:Final Answer|The answer is|Answer)\s*[:=]\s*\(?([A-E])\)?",
            text,
            re.IGNORECASE,
        )
        if final_match:
            return final_match.group(1).upper()

        # Check last line
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        if lines:
            last = lines[-1]
            choice_match = re.search(r"\b([A-E])\b", last)
            if choice_match:
                return choice_match.group(1).upper()

        # Search entire text (last occurrence of standalone letter)
        matches = re.findall(r"(?:^|\s)\(?([A-E])\)?(?:\s|$|\.)", text, re.MULTILINE)
        if matches:
            return matches[-1].upper()

        return None

    def _extract_nli(self, text: str, benchmark_key: str = "") -> Optional[str]:
        """Extract NLI label (True/False/Unknown)."""
        # Check for explicit final answer
        final_match = re.search(
            r"(?:Final Answer|The answer is|Answer|Conclusion)\s*[:=]\s*(True|False|Unknown|Entailment|Contradiction|Neutral)",
            text,
            re.IGNORECASE,
        )
        if final_match:
            return self._normalize_nli(final_match.group(1).strip())

        # Check last line
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        if lines:
            last = lines[-1].lower()
            if "true" in last or "entailment" in last:
                return "True"
            if "false" in last or "contradiction" in last:
                return "False"
            if "unknown" in last or "neutral" in last:
                return "Unknown"

        return None

    def _extract_generic(self, text: str, benchmark_key: str = "") -> Optional[str]:
        """Generic answer extraction fallback."""
        final_match = re.search(
            r"(?:Final Answer|The answer is|Answer)\s*[:=]\s*(.+?)(?:\n|$)",
            text,
            re.IGNORECASE,
        )
        if final_match:
            return final_match.group(1).strip()

        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        return lines[-1] if lines else None

    @staticmethod
    def _normalize_numeric(value: str) -> str:
        """Robust numeric normalization."""

        value = value.strip()

        # REMOVE LATEX / MARKDOWN / SYMBOLS
        value = re.sub(r"\\\[|\\\]|\\\(|\\\)", "", value)   # remove \[ \]
        value = re.sub(r"[*_`$]", "", value)               # remove markdown
        value = re.sub(r"[^\d\.\-]", " ", value)           # keep only numbers
        value = " ".join(value.split())

        # Extract first valid number
        numbers = re.findall(r"-?\d+\.?\d*", value)
        if not numbers:
            return value.strip()

        value = numbers[0]

        # Normalize
        try:
            num = float(value)
            if num == int(num):
                return str(int(num))
            return str(num)
        except:
            return value

    @staticmethod
    def _normalize_nli(label: str) -> str:
        """Normalize NLI label."""
        label = label.strip().lower()
        if label in ("true", "entailment"):
            return "True"
        if label in ("false", "contradiction"):
            return "False"
        if label in ("unknown", "neutral"):
            return "Unknown"
        return label.capitalize()

    @staticmethod
    def check_answer(
        predicted: Optional[str],
        gold: str,
        answer_type: str,
    ) -> bool:
        """Check if predicted answer matches gold answer.

        Args:
            predicted: Extracted predicted answer
            gold: Gold standard answer
            answer_type: Answer type for comparison strategy

        Returns:
            True if answers match
        """
        if predicted is None:
            return False

        pred = predicted.strip().lower()
        gold_clean = gold.strip().lower()

        if answer_type == "numeric":
            # Normalize both to numbers
            try:
                pred_num = float(pred.replace(",", ""))
                gold_num = float(gold_clean.replace(",", ""))
                return abs(pred_num - gold_num) < 1e-6
            except ValueError:
                return pred == gold_clean

        if answer_type == "yes_no":
            # Normalize boolean strings
            pred_bool = pred in ("yes", "true", "1")
            gold_bool = gold_clean in ("yes", "true", "1")
            return pred_bool == gold_bool

        if answer_type == "multiple_choice":
            return pred.upper() == gold_clean.upper()

        if answer_type == "nli":
            return pred.lower() == gold_clean.lower()

        # Generic string comparison
        return pred == gold_clean
