#src/utils/cot_parser.py
"""
Chain-of-Thought Parser.

Parses raw model output into individual reasoning steps.
Handles multiple CoT formats: numbered steps, paragraph-based reasoning,
and thinking-tag-based reasoning (e.g., <think>...</think> from DeepSeek-R1).
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class ReasoningStep:
    """A single reasoning step extracted from a CoT."""
    index: int                    # 0-indexed position in the chain
    text: str                     # Raw text of this step
    step_type: str = "reasoning"  # "reasoning", "verification", "backtracking", "conclusion"
    tokens: Optional[List[str]] = None


@dataclass
class ParsedCoT:
    """Parsed chain-of-thought with individual steps."""
    raw_text: str                       # Original full CoT text
    steps: List[ReasoningStep]          # Individual reasoning steps
    final_answer_text: str = ""         # Text of the final answer portion
    num_steps: int = 0
    has_think_tags: bool = False        # Whether <think>...</think> tags were present
    thinking_text: str = ""             # Content inside <think> tags

    def __post_init__(self):
        self.num_steps = len(self.steps)


class CoTParser:
    """Parse chain-of-thought outputs into structured reasoning steps."""

    # Patterns for detecting step boundaries
    STEP_PATTERNS = [
        # "Step 1:", "Step 2:", etc.
        r"(?:^|\n)\s*(?:Step\s+\d+)\s*[:.]",
        # "1.", "2.", "3.", etc. (numbered lists)
        r"(?:^|\n)\s*\d+\s*[.)]\s+",
        # "First,", "Second,", "Third,", etc.
        r"(?:^|\n)\s*(?:First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth|Ninth|Tenth|Next|Then|Finally|Therefore|Thus|Hence|So)\s*[,:]",
    ]

    # Patterns for detecting thinking tags (DeepSeek-R1 style)
    THINK_TAG_PATTERN = re.compile(
        r"<think>(.*?)</think>",
        re.DOTALL | re.IGNORECASE,
    )

    # Patterns for detecting backtracking
    BACKTRACK_PATTERNS = [
        r"(?:wait|actually|no|let me reconsider|I made an error|correction|let me re-examine)",
        r"(?:going back|re-think|revise|recalculate|I was wrong)",
    ]

    # Patterns for detecting verification
    VERIFY_PATTERNS = [
        r"(?:let me verify|let me check|double.check|to confirm|verifying|checking)",
        r"(?:sanity check|cross.check|validate|does this make sense)",
    ]

    def __init__(self):
        self._compiled_step_patterns = [
            re.compile(p, re.MULTILINE | re.IGNORECASE) for p in self.STEP_PATTERNS
        ]
        self._backtrack_re = re.compile(
            "|".join(self.BACKTRACK_PATTERNS), re.IGNORECASE
        )
        self._verify_re = re.compile(
            "|".join(self.VERIFY_PATTERNS), re.IGNORECASE
        )

    def parse(self, raw_output: str) -> ParsedCoT:
        """Parse raw model output into structured CoT.

        Args:
            raw_output: Raw text output from the model

        Returns:
            ParsedCoT instance with extracted steps
        """
        if not raw_output or not raw_output.strip():
            return ParsedCoT(
                raw_text=raw_output or "",
                steps=[],
                final_answer_text="",
            )

        # Check for <think> tags (DeepSeek-R1 style)
        has_think_tags = False
        thinking_text = ""
        reasoning_text = raw_output

        think_match = self.THINK_TAG_PATTERN.search(raw_output)
        if think_match:
            has_think_tags = True
            thinking_text = think_match.group(1).strip()
            # Use the thinking content for step parsing
            reasoning_text = thinking_text

        # Extract final answer portion
        final_answer_text = self._extract_final_answer_text(raw_output)

        # Parse into steps
        steps = self._split_into_steps(reasoning_text)

        # Classify step types
        for step in steps:
            step.step_type = self._classify_step(step.text)

        return ParsedCoT(
            raw_text=raw_output,
            steps=steps,
            final_answer_text=final_answer_text,
            has_think_tags=has_think_tags,
            thinking_text=thinking_text,
        )

    def _split_into_steps(self, text: str) -> List[ReasoningStep]:
        """Split reasoning text into individual steps.

        Tries multiple splitting strategies and uses the one that
        produces the most granular, meaningful steps.
        """
        if not text.strip():
            return []

        # Strategy 1: Split by explicit step markers
        for pattern in self._compiled_step_patterns:
            matches = list(pattern.finditer(text))
            if len(matches) >= 2:
                steps = self._split_by_matches(text, matches)
                if len(steps) >= 2:
                    return steps

        # Strategy 2: Split by sentence boundaries with newlines
        steps = self._split_by_sentences(text)
        if len(steps) >= 2:
            return steps

        # Strategy 3: Single step (entire text)
        return [
            ReasoningStep(index=0, text=text.strip(), step_type="reasoning")
        ]

    def _split_by_matches(
        self, text: str, matches: List[re.Match]
    ) -> List[ReasoningStep]:
        """Split text using regex match positions."""
        steps = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            step_text = text[start:end].strip()
            if step_text:
                steps.append(
                    ReasoningStep(index=i, text=step_text, step_type="reasoning")
                )
        return steps

    def _split_by_sentences(self, text: str) -> List[ReasoningStep]:
        """Split by sentence-like boundaries (newlines + periods)."""
        # Split on double newlines first
        paragraphs = re.split(r"\n\s*\n", text)
        if len(paragraphs) >= 2:
            steps = []
            for i, para in enumerate(paragraphs):
                para = para.strip()
                if para and len(para) > 10:  # Skip very short fragments
                    steps.append(
                        ReasoningStep(index=i, text=para, step_type="reasoning")
                    )
            if len(steps) >= 2:
                return steps

        # Split on single newlines
        lines = text.split("\n")
        steps = []
        idx = 0
        for line in lines:
            line = line.strip()
            if line and len(line) > 10:
                steps.append(
                    ReasoningStep(index=idx, text=line, step_type="reasoning")
                )
                idx += 1
        return steps

    def _classify_step(self, step_text: str) -> str:
        """Classify a reasoning step into categories."""
        if self._backtrack_re.search(step_text):
            return "backtracking"
        if self._verify_re.search(step_text):
            return "verification"
        # Check if it's a conclusion-like step
        conclusion_patterns = [
            r"(?:therefore|thus|hence|so|in conclusion|final answer|the answer is)",
        ]
        for pattern in conclusion_patterns:
            if re.search(pattern, step_text, re.IGNORECASE):
                return "conclusion"
        return "reasoning"

    def _extract_final_answer_text(self, text: str) -> str:
        """Extract the final answer portion from the full output."""
        # Common patterns for final answers
        patterns = [
            r"(?:Final Answer|The answer is|Answer|ANSWER)\s*[:=]\s*(.*?)(?:\n|$)",
            r"\\boxed\{([^}]+)\}",
            r"####\s*(.*?)(?:\n|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()

        # Fallback: last line
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        return lines[-1] if lines else ""

    def get_partial_cot(
        self, parsed_cot: ParsedCoT, num_steps: int
    ) -> str:
        """Get CoT truncated to first num_steps steps.

        Used for early answering perturbation test.

        Args:
            parsed_cot: Parsed CoT instance
            num_steps: Number of steps to include

        Returns:
            Truncated CoT text
        """
        if num_steps <= 0:
            return ""
        steps = parsed_cot.steps[:num_steps]
        return "\n".join(s.text for s in steps)

    def remove_step(self, parsed_cot: ParsedCoT, step_index: int) -> str:
        """Get CoT with a specific step removed.

        Used for step deletion perturbation test.

        Args:
            parsed_cot: Parsed CoT instance
            step_index: Index of step to remove (0-based)

        Returns:
            CoT text with the step removed
        """
        steps = [s for i, s in enumerate(parsed_cot.steps) if i != step_index]
        return "\n".join(s.text for s in steps)

    def shuffle_steps(self, parsed_cot: ParsedCoT, rng) -> str:
        """Get CoT with steps in random order.

        Used for step shuffling perturbation test.

        Args:
            parsed_cot: Parsed CoT instance
            rng: numpy random generator for reproducibility

        Returns:
            CoT text with shuffled steps
        """
        steps = list(parsed_cot.steps)
        indices = list(range(len(steps)))
        rng.shuffle(indices)
        shuffled = [steps[i] for i in indices]
        return "\n".join(s.text for s in shuffled)
