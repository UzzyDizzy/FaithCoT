"""
Inference Engine (FULL FIXED VERSION)

Includes:
- Stable generation
- Correct log-prob computation (CRITICAL FIX)
- Proper device handling
- Debug logs for SIG/CNS failures
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

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        generation_config: dict,
        device: str = "cuda",
        use_amp: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config

        # 🔥 FIX: ALWAYS TRUST MODEL DEVICE
        self.device = model.device

        self.use_amp = use_amp and torch.cuda.is_available()

        self.cot_parser = CoTParser()
        self.answer_extractor = AnswerExtractor()

    # =========================================================
    # CLEAN TEXT
    # =========================================================
    def _clean_text(self, texts):
        cleaned = []
        for text in texts:
            text = text.replace("Ġ", " ")
            text = text.replace("▁", " ")
            text = text.replace("\u00a0", " ")
            text = " ".join(text.split())
            cleaned.append(text.strip())
        return cleaned

    # =========================================================
    # GENERATE SINGLE
    # =========================================================
    @torch.no_grad()
    def generate_cot(self, prompt: str, max_new_tokens=None):

        print("\n====== GENERATION DEBUG ======")
        print("PROMPT:", prompt[:120])

        config = dict(self.generation_config)
        if max_new_tokens:
            config["max_new_tokens"] = max_new_tokens

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[1]

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            outputs = self.model.generate(
                **inputs,
                **config,
                output_scores=True,
                return_dict_in_generate=True,
            )

        gen_ids = outputs.sequences[:, input_len:]

        decoded = self.tokenizer.batch_decode(
            gen_ids,
            skip_special_tokens=True,
        )

        raw_output = self._clean_text(decoded)[0]

        print("OUTPUT:", raw_output[:200])

        parsed = self.cot_parser.parse(raw_output)

        return {
            "raw_output": raw_output,
            "parsed_cot": parsed,
            "num_generated_tokens": gen_ids.shape[1],
            "num_steps": parsed.num_steps,
            "log_probs": self._extract_log_probs(outputs.scores),
            "input_length": input_len,
        }

    # =========================================================
    # GENERATE BATCH
    # =========================================================
    @torch.no_grad()
    def generate_batch(self, prompts, max_new_tokens=None, batch_size=4):

        results = []

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]

            batch_res = self._generate_batch_internal(batch, max_new_tokens)
            results.extend(batch_res)

            logger.info(f"Generated {min(i+batch_size, len(prompts))}/{len(prompts)}")

        return results

    # =========================================================
    # 🔥 FIXED LOGPROB (ROOT FIX)
    # =========================================================
    @torch.no_grad()
    def get_answer_log_prob(self, context: str, answer: str):

        print("\n====== LOGPROB DEBUG ======")

        if not context or len(context.strip()) < 5:
            print("[LP DEBUG] ❌ Empty context")
            return None

        if answer is None or str(answer).strip() == "":
            print("[LP DEBUG] ❌ Empty answer")
            return None

        answer = str(answer).strip()

        # 🔥 CRITICAL FIX: newline separator
        full_text = context.strip() + "\n" + answer

        try:
            enc_full = self.tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            ).to(self.model.device)

            enc_ctx = self.tokenizer(
                context.strip(),
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            ).to(self.model.device)

            input_ids = enc_full.input_ids
            attention_mask = enc_full.attention_mask

            context_len = enc_ctx.input_ids.shape[1]

            print(f"[LP DEBUG] context_len={context_len}, total_len={input_ids.shape[1]}")

            if context_len >= input_ids.shape[1]:
                print("[LP DEBUG] ❌ Answer truncated")
                return None

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            logits = outputs.logits
            log_probs = torch.log_softmax(logits.float(), dim=-1)

            total_log_prob = 0.0

            for i in range(context_len, input_ids.shape[1]):
                token_id = input_ids[0, i]
                total_log_prob += log_probs[0, i - 1, token_id].item()

            print(f"[LP DEBUG] ✅ log_prob={total_log_prob:.4f}")

            return float(total_log_prob)

        except Exception as e:
            print(f"[LP DEBUG] ❌ Exception: {e}")
            return None

    # =========================================================
    # SEQUENCE LOGPROB
    # =========================================================
    @torch.no_grad()
    def get_sequence_log_prob(self, prompt: str, continuation: str):

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

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            outputs = self.model(**inputs)

        logits = outputs.logits[0]
        log_probs = torch.log_softmax(logits.float(), dim=-1)

        total = 0.0
        input_ids = inputs["input_ids"][0]

        for i in range(prompt_len, len(input_ids)):
            token_id = input_ids[i].item()
            total += log_probs[i - 1, token_id].item()

        return total

    # =========================================================
    # ANSWER EXTRACTION
    # =========================================================
    def extract_answer(self, raw_output, answer_type, benchmark_key=""):
        return self.answer_extractor.extract(raw_output, answer_type, benchmark_key)

    # =========================================================
    # INTERNAL BATCH
    # =========================================================
    def _generate_batch_internal(self, prompts, max_new_tokens=None):

        config = dict(self.generation_config)
        if max_new_tokens:
            config["max_new_tokens"] = max_new_tokens

        self.tokenizer.padding_side = "left"

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(self.model.device)

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            outputs = self.model.generate(**inputs, **config)

        gen_ids = outputs[:, inputs["input_ids"].shape[1]:]

        decoded = self.tokenizer.batch_decode(
            gen_ids,
            skip_special_tokens=True,
        )

        cleaned = self._clean_text(decoded)

        results = []

        for i, text in enumerate(cleaned):

            if i < 2:
                print("\n====== BATCH DEBUG ======")
                print("RAW:", text[:200])

            parsed = self.cot_parser.parse(text)

            results.append({
                "raw_output": text,
                "parsed_cot": parsed,
                "num_generated_tokens": gen_ids[i].shape[0],
                "num_steps": parsed.num_steps,
                "log_probs": None,
                "input_length": None,
            })

        self.tokenizer.padding_side = "right"

        return results

    # =========================================================
    # LOGPROBS FROM GENERATION
    # =========================================================
    def _extract_log_probs(self, scores):

        if scores is None:
            return None

        vals = []

        for s in scores:
            lp = torch.log_softmax(s[0].float(), dim=-1)
            vals.append(lp.max().item())

        return np.array(vals, dtype=np.float32)

    # =========================================================
    # TOP TOKENS
    # =========================================================
    def _get_top_tokens(self, log_probs, k=10):

        top_vals, top_idx = torch.topk(log_probs, k)

        return [
            (self.tokenizer.decode([i]), v)
            for i, v in zip(top_idx.tolist(), top_vals.tolist())
        ]