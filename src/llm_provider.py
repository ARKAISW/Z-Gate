"""llm_provider.py — Pluggable LLM sentiment and reflection provider.

Fallback chain:
  1. Featherless AI (OpenAI-compatible, FEATHERLESS_API_KEY)
  2. Groq (OpenAI-compatible, GROQ_API_KEY)
  3. Local Ollama (OLLAMA_BASE_URL, default http://localhost:11434/v1)
  4. none (graceful degradation: sentiment_modifier = 0.0)

Safety constraints:
  - summarize_sentiment() NEVER raises an exception — always degrades to neutral.
  - LLM is used ONLY for sentiment classification and post-trade reflection.
  - LLM NEVER decides trades, sees account values, or executes orders.
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Literal

import numpy as np
from openai import OpenAI

from src.models import SentimentResult

logger = logging.getLogger(__name__)

# System prompt for financial sentiment classification
SENTIMENT_SYSTEM_PROMPT = """You are a financial news sentiment classifier. You will be given up to 5 recent
headlines about a specific asset (a stock ticker, ETF, or cryptocurrency). Your only
job is to judge the near-term sentiment these headlines imply for that asset, from the
perspective of a short-term systematic trader.

Respond with ONLY a single JSON object — no other text, no markdown fences:
{
  "sentiment": "positive" | "neutral" | "negative",
  "confidence": <float between 0.0 and 1.0>,
  "reason": "<one sentence, max 20 words>"
}

Rules:
- If headlines are mixed or unclear, respond "neutral" with low confidence.
- If headlines are not about this asset, or too vague to judge, respond "neutral"
  with confidence 0.0 and reason "insufficient information".
- Do not invent information. Judge tone and implied short-term reaction only.
- Do not give price targets or investment advice.
"""

# In-memory cycle cache to avoid duplicate LLM calls during the same poll interval
_SENTIMENT_CACHE: dict[tuple[str, tuple[str, ...]], SentimentResult] = {}


def _clean_json_text(text: str) -> str:
    """Strip markdown fences (```json ... ```) if present."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try finding the first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1].strip()
    return text


class LLMProvider:
    """Multi-backend LLM client with automatic fallback."""

    def __init__(
        self,
        primary_provider: str | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        self.primary_provider = primary_provider or os.environ.get("LLM_PROVIDER", "featherless").lower()
        self.timeout = timeout_seconds

    def generate_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
    ) -> tuple[str, Literal["featherless", "groq", "ollama", "none"]]:
        """Generate text completion trying providers in order."""
        providers_to_try = self._build_provider_chain()

        for provider in providers_to_try:
            if provider == "none":
                break
            try:
                text = self._call_provider(provider, system_prompt, user_prompt, temperature)
                if text:
                    return text, provider  # type: ignore
            except Exception as exc:
                logger.warning("LLM provider '%s' failed: %s. Trying fallback...", provider, exc)

        return "", "none"

    def summarize_sentiment(
        self,
        asset: str,
        headlines: list[str],
    ) -> SentimentResult:
        """Classify headlines into a structured SentimentResult.

        Never raises — always returns a valid SentimentResult object.
        """
        if not headlines:
            return SentimentResult(
                asset=asset,
                sentiment="neutral",
                confidence=0.0,
                reason="No headlines provided",
                modifier=0.0,
                provider_used="none",
            )

        cache_key = (asset, tuple(sorted(headlines)))
        if cache_key in _SENTIMENT_CACHE:
            return _SENTIMENT_CACHE[cache_key]

        headlines_text = "\n".join(f"{i+1}. {h}" for i, h in enumerate(headlines[:5]))
        user_prompt = f"Asset: {asset}\n\nHeadlines:\n{headlines_text}"

        raw_text, provider_used = self.generate_completion(
            system_prompt=SENTIMENT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.1,
        )

        if provider_used == "none" or not raw_text:
            res = SentimentResult(
                asset=asset,
                sentiment="neutral",
                confidence=0.0,
                reason="LLM sentiment unavailable (all providers failed)",
                modifier=0.0,
                provider_used="none",
                raw_response=None,
            )
            _SENTIMENT_CACHE[cache_key] = res
            return res

        # Parse JSON
        try:
            cleaned = _clean_json_text(raw_text)
            data = json.loads(cleaned)

            sentiment = str(data.get("sentiment", "neutral")).lower()
            if sentiment not in ("positive", "neutral", "negative"):
                sentiment = "neutral"

            confidence = float(np.clip(float(data.get("confidence", 0.0)), 0.0, 1.0))
            reason = str(data.get("reason", "headline analysis"))[:200]

            modifier = 0.15 * confidence if sentiment in ("positive", "negative") else 0.0

            result = SentimentResult(
                asset=asset,
                sentiment=sentiment,  # type: ignore
                confidence=confidence,
                reason=reason,
                modifier=modifier,
                provider_used=provider_used,
                raw_response=raw_text,
            )
            _SENTIMENT_CACHE[cache_key] = result
            return result

        except Exception as exc:
            logger.warning("Failed to parse LLM sentiment JSON: %s. Raw: %s", exc, raw_text)
            fallback = SentimentResult(
                asset=asset,
                sentiment="neutral",
                confidence=0.0,
                reason="Sentiment parse error",
                modifier=0.0,
                provider_used=provider_used,
                raw_response=raw_text,
            )
            _SENTIMENT_CACHE[cache_key] = fallback
            return fallback

    def _build_provider_chain(self) -> list[str]:
        chain = []
        primary = self.primary_provider.lower()
        if primary in ("groq", "cerebras", "featherless", "ollama"):
            chain.append(primary)

        defaults = ["groq", "cerebras", "featherless", "ollama", "none"]
        for p in defaults:
            if p not in chain:
                chain.append(p)
        return chain

    def _call_provider(
        self,
        provider: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
    ) -> str:
        if provider == "groq":
            api_key = os.environ.get("GROQ_API_KEY", "")
            if not api_key:
                raise ValueError("GROQ_API_KEY not set")
            model = os.environ.get("GROQ_MODEL", "groq/compound-mini")
            client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=api_key,
                timeout=self.timeout,
            )
        elif provider == "cerebras":
            api_key = os.environ.get("CEREBRAS_API_KEY", "")
            if not api_key:
                raise ValueError("CEREBRAS_API_KEY not set")
            model = os.environ.get("CEREBRAS_MODEL", "llama3.1-8b")
            client = OpenAI(
                base_url="https://api.cerebras.ai/v1",
                api_key=api_key,
                timeout=self.timeout,
            )
        elif provider == "featherless":
            api_key = os.environ.get("FEATHERLESS_API_KEY", "")
            if not api_key:
                raise ValueError("FEATHERLESS_API_KEY not set")
            model = os.environ.get("FEATHERLESS_MODEL", "Qwen/Qwen2.5-7B-Instruct")
            client = OpenAI(
                base_url="https://api.featherless.ai/v1",
                api_key=api_key,
                timeout=self.timeout,
            )
        elif provider == "ollama":
            base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
            model = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b-instruct")
            client = OpenAI(
                base_url=base_url,
                api_key="ollama",  # dummy key required by SDK
                timeout=self.timeout,
            )
        else:
            return ""

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=300,
        )
        return response.choices[0].message.content or ""


# Global singleton instance
_default_provider = LLMProvider()


def summarize_sentiment(asset: str, headlines: list[str]) -> SentimentResult:
    """Convenience function delegating to the global LLMProvider instance."""
    return _default_provider.summarize_sentiment(asset, headlines)
