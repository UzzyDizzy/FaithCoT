#src/models/api_models.py
"""
API Model Wrappers.

Unified interface for OpenAI and DeepSeek API models.
Reads API keys from .env.local file.
"""

import os
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from src.utils.cot_parser import CoTParser
from src.utils.answer_extractor import AnswerExtractor
from src.utils.logger import get_logger

logger = get_logger("api_models")

# Load environment variables from .env.local
_env_path = os.path.join(
    os.path.dirname(__file__), "..", "..", ".env.local"
)
load_dotenv(_env_path)


class APIModel:
    """Unified API model wrapper for OpenAI and DeepSeek."""

    def __init__(
        self,
        provider: str,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        """Initialize API model.

        Args:
            provider: "openai" or "deepseek"
            model_name: Model name (e.g., "gpt-4o", "deepseek-chat")
            base_url: API base URL (None for default OpenAI)
            api_key: API key (None to read from env)
            max_retries: Number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.provider = provider
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.cot_parser = CoTParser()
        self.answer_extractor = AnswerExtractor()

        # Resolve API key
        if api_key:
            self.api_key = api_key
        elif provider == "openai":
            self.api_key = os.getenv("OPENAI_API_KEY", "")
        elif provider == "deepseek":
            self.api_key = os.getenv("DEEPSEEK_API_KEY", "")
        else:
            self.api_key = ""

        # Initialize OpenAI client
        from openai import OpenAI

        client_kwargs = {"api_key": self.api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        elif provider == "deepseek":
            client_kwargs["base_url"] = "https://api.deepseek.com"

        self.client = OpenAI(**client_kwargs)
        logger.info(f"Initialized {provider} API model: {model_name}")

    def generate_cot(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Generate CoT reasoning via API.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Dict with raw_output, parsed_cot, usage info
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a careful reasoning assistant. "
                                "Show your step-by-step reasoning clearly."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

                raw_output = response.choices[0].message.content or ""
                parsed_cot = self.cot_parser.parse(raw_output)

                # Extract reasoning content if available (DeepSeek)
                reasoning_content = ""
                if hasattr(response.choices[0].message, "reasoning_content"):
                    reasoning_content = (
                        response.choices[0].message.reasoning_content or ""
                    )

                usage = {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                }

                return {
                    "raw_output": raw_output,
                    "parsed_cot": parsed_cot,
                    "reasoning_content": reasoning_content,
                    "num_steps": parsed_cot.num_steps,
                    "usage": usage,
                    "model": self.model_name,
                    "provider": self.provider,
                }

            except Exception as e:
                logger.warning(
                    f"API call failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error(f"All API retries failed for {self.model_name}")
                    return {
                        "raw_output": "",
                        "parsed_cot": self.cot_parser.parse(""),
                        "reasoning_content": "",
                        "num_steps": 0,
                        "usage": {},
                        "model": self.model_name,
                        "provider": self.provider,
                        "error": str(e),
                    }

    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 2048,
        temperature: float = 0.0,
        delay: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Generate CoT for multiple prompts sequentially.

        Args:
            prompts: List of input prompts
            max_tokens: Max tokens per response
            temperature: Sampling temperature
            delay: Delay between requests (rate limiting)

        Returns:
            List of result dicts
        """
        results = []
        for i, prompt in enumerate(prompts):
            result = self.generate_cot(prompt, max_tokens, temperature)
            results.append(result)
            if i < len(prompts) - 1:
                time.sleep(delay)
            if (i + 1) % 10 == 0:
                logger.info(f"API: Generated {i + 1}/{len(prompts)}")
        return results
