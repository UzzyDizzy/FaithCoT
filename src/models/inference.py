#src/models/inference.py
"""
Inference Engine.

Generates CoT reasoning from loaded models with support for:
- Full CoT generation
- Partial CoT (early answering)
- Log-probability extraction for information-theoretic metrics
- Batch inference with AMP
"""

import torch
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.cot_parser import CoTParser, ParsedCoT
from src.utils.answer_extractor import AnswerExtractor
from src.utils.logger import get_logger

logger = get_logger("inference")


class InferenceEngine:
    """Engine for generating and analyzing CoT reasoning."""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        generation_config: dict,
        device: str = "cuda",
        use_amp: bool = True,
    ):
        """Initialize the inference engine.

        Args:
            model: Loaded HuggingFace model
            tokenizer: Corresponding tokenizer
            generation_config: Generation parameters
            device: Device string ("cuda" or "cpu")
            use_amp: Whether to use Automatic Mixed Precision
        """
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        self.device = device
        self.use_amp = use_amp and torch.cuda.is_available()
        self.cot_parser = CoTParser()
        self.answer_extractor = AnswerExtractor()

    def _clean_text(self, texts):
        """Universal tokenizer cleanup (model-agnostic)."""
        cleaned = []
        for text in texts:
            text = text.replace("Ġ", " ")      # GPT2 / LLaMA BPE
            text = text.replace("▁", " ")      # SentencePiece (Qwen, T5, etc.)
            text = text.replace("\u00a0", " ") # non-breaking spaces
            text = " ".join(text.split())      # normalize whitespace
            cleaned.append(text.strip())
        return cleaned

    @torch.no_grad()
    def generate_cot(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate CoT reasoning for a single prompt.

        Args:
            prompt: Input prompt
            max_new_tokens: Override max tokens

        Returns:
            Dict with raw_output, parsed_cot, generated_tokens, etc.
        """
        config = dict(self.generation_config)
        if max_new_tokens:
            config["max_new_tokens"] = max_new_tokens

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.model.device)

        input_length = inputs["input_ids"].shape[1]

        # Generate with AMP
        if self.use_amp:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = self.model.generate(
                    **inputs,
                    **config,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
        else:
            outputs = self.model.generate(
                **inputs,
                **config,
                output_scores=True,
                return_dict_in_generate=True,
            )

        # Decode
        generated_ids = outputs.sequences[:, input_length:]

        decoded = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        raw_output = self._clean_text(decoded)[0]

        # Parse CoT
        parsed_cot = self.cot_parser.parse(raw_output)

        # Extract log-probabilities for SIG computation
        log_probs = self._extract_log_probs(outputs.scores)

        return {
            "raw_output": raw_output,
            "parsed_cot": parsed_cot,
            "num_generated_tokens": len(generated_ids),
            "num_steps": parsed_cot.num_steps,
            "log_probs": log_probs,
            "input_length": input_length,
        }

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: Optional[int] = None,
        batch_size: int = 4,
    ) -> List[Dict[str, Any]]:
        """Generate CoT reasoning for a batch of prompts.

        Args:
            prompts: List of input prompts
            max_new_tokens: Override max tokens
            batch_size: Batch size for inference

        Returns:
            List of result dicts (same format as generate_cot)
        """
        results = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            batch_results = self._generate_batch_internal(
                batch_prompts, max_new_tokens
            )
            results.extend(batch_results)
            if (i + batch_size) % (batch_size * 5) == 0:
                logger.info(
                    f"Generated {min(i + batch_size, len(prompts))}/{len(prompts)}"
                )
        return results

    @torch.no_grad()
    def get_answer_log_probs(
        self,
        prompt_with_cot: str,
        answer_tokens: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Get log-probabilities for answer tokens given prompt + CoT.

        Used for Step Information Gain (SIG) computation.

        Args:
            prompt_with_cot: Full prompt including CoT prefix
            answer_tokens: Optional specific answer tokens to evaluate

        Returns:
            Dict with log_probs, entropy, token_probs
        """
        inputs = self.tokenizer(
            prompt_with_cot,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.model.device)

        if self.use_amp:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = self.model(
                    **inputs,
                    output_hidden_states=False,
                )
        else:
            outputs = self.model(**inputs, output_hidden_states=False)

        # Get logits for the last token (next token prediction)
        last_logits = outputs.logits[0, -1, :]  # [vocab_size]
        log_probs = torch.log_softmax(last_logits.float(), dim=-1)
        probs = torch.softmax(last_logits.float(), dim=-1)

        # Compute entropy H(A | context)
        entropy = -(probs * log_probs).sum().item()
        # Clamp to avoid numerical issues
        entropy = max(0.0, entropy)

        return {
            "log_probs": log_probs.cpu().numpy(),
            "entropy": entropy,
            "top_tokens": self._get_top_tokens(log_probs, k=10),
        }

    @torch.no_grad()
    def get_sequence_log_prob(
        self,
        prompt: str,
        continuation: str,
    ) -> float:
        """Get the total log-probability of a continuation given a prompt.

        Args:
            prompt: Context/prompt text
            continuation: Text to compute log-prob for

        Returns:
            Total log-probability (sum of per-token log-probs)
        """
        full_text = prompt + continuation
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.model.device)

        prompt_inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        prompt_len = prompt_inputs["input_ids"].shape[1]

        if self.use_amp:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = self.model(**inputs)
        else:
            outputs = self.model(**inputs)

        # Get log-probs for continuation tokens
        logits = outputs.logits[0]  # [seq_len, vocab_size]
        log_probs = torch.log_softmax(logits.float(), dim=-1)

        # Sum log-probs of continuation tokens
        total_log_prob = 0.0
        input_ids = inputs["input_ids"][0]
        for i in range(prompt_len, len(input_ids)):
            token_id = input_ids[i].item()
            total_log_prob += log_probs[i - 1, token_id].item()

        return total_log_prob

    def extract_answer(
        self,
        raw_output: str,
        answer_type: str,
        benchmark_key: str = "",
    ) -> Optional[str]:
        """Extract the final answer from raw model output.

        Args:
            raw_output: Raw model output text
            answer_type: Type of answer expected
            benchmark_key: Benchmark identifier

        Returns:
            Extracted answer string or None
        """
        return self.answer_extractor.extract(raw_output, answer_type, benchmark_key)

    def _generate_batch_internal(
        self,
        prompts: List[str],
        max_new_tokens: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Internal batch generation with left-padding."""
        config = dict(self.generation_config)
        if max_new_tokens:
            config["max_new_tokens"] = max_new_tokens

        # Left-pad for batch generation
        self.tokenizer.padding_side = "left"
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(self.model.device)

        input_lengths = [
            (inputs["attention_mask"][i] == 1).sum().item()
            for i in range(len(prompts))
        ]

        if self.use_amp:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = self.model.generate(**inputs, **config)
        else:
            outputs = self.model.generate(**inputs, **config)

        # Decode each sequence
        generated_ids = outputs[:, inputs["input_ids"].shape[1]:]

        decoded_outputs = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        cleaned_outputs = self._clean_text(decoded_outputs)

        results = []
        
        for i in range(len(prompts)):
            raw_output = cleaned_outputs[i]
            parsed_cot = self.cot_parser.parse(raw_output)

            results.append({
                "raw_output": raw_output,
                "parsed_cot": parsed_cot,
                "num_generated_tokens": len(generated_ids[i]),
                "num_steps": parsed_cot.num_steps,
                "log_probs": None,
                "input_length": input_lengths[i],
            })

        # Reset padding side
        self.tokenizer.padding_side = "right"
        return results

    def _extract_log_probs(
        self, scores: Tuple[torch.Tensor, ...]
    ) -> Optional[np.ndarray]:
        """Extract per-token log-probabilities from generation scores.

        Args:
            scores: Tuple of score tensors from generation

        Returns:
            Numpy array of shape [num_tokens] with log-probs, or None
        """
        if scores is None or len(scores) == 0:
            return None

        log_probs = []
        for score in scores:
            # score shape: [batch_size, vocab_size]
            lp = torch.log_softmax(score[0].float(), dim=-1)
            # Get the max log-prob (chosen token)
            max_lp = lp.max().item()
            log_probs.append(max_lp)

        return np.array(log_probs, dtype=np.float32)

    def _get_top_tokens(
        self, log_probs: torch.Tensor, k: int = 10
    ) -> List[Tuple[str, float]]:
        """Get top-k tokens with their log-probabilities.

        Args:
            log_probs: Log-probability tensor [vocab_size]
            k: Number of top tokens

        Returns:
            List of (token_text, log_prob) tuples
        """
        top_values, top_indices = torch.topk(log_probs, k)
        result = []
        for val, idx in zip(top_values.tolist(), top_indices.tolist()):
            token_text = self.tokenizer.decode([idx])
            result.append((token_text, val))
        return result
