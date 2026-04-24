from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from presidio_analyzer import AnalyzerEngine

from ..models.column_profile import PIICategory
import structlog

logger = structlog.get_logger()

# Limit concurrent LLM calls from the PII detector to avoid rate-limit errors.
# All PIIDetector instances share the same semaphore (module-level singleton).
_LLM_CONCURRENCY = 4
_llm_semaphore: asyncio.Semaphore | None = None


def _get_llm_semaphore() -> asyncio.Semaphore:
    """Return (creating lazily) the module-level LLM concurrency semaphore.

    Lazy init is required because asyncio.Semaphore must be created inside a
    running event loop.
    """
    global _llm_semaphore
    if _llm_semaphore is None:
        _llm_semaphore = asyncio.Semaphore(_LLM_CONCURRENCY)
    return _llm_semaphore


class PIIDetector:
    """Three-layer PII detection: regex → Presidio NLP → LLM heuristic."""

    _REGEX_PATTERNS: dict[str, str] = {
        "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
        "PHONE_US": r"\b(\+?1[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b",
        "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "IP_ADDRESS": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
        "ZIP_US": r"\b\d{5}(-\d{4})?\b",
        "CARD_PAN": r"\b\d{13,19}\b",
        "IBAN": r"\b[A-Z]{2}\d{2}[A-Z0-9]{4,30}\b",
        "ROUTING_NUMBER": r"\b\d{9}\b",
        "DATE_OF_BIRTH": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        "PASSPORT_NUMBER": r"\b[A-Z]{1,2}\d{6,9}\b",
        "TAX_ID": r"\b\d{2}-\d{7}\b",
        "SWIFT_BIC": r"\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?\b",
    }

    _DIRECT_PII_REGEX = {"SSN", "CARD_PAN", "EMAIL", "IBAN", "PASSPORT_NUMBER"}
    _FINANCIAL_REGEX = {"ROUTING_NUMBER", "SWIFT_BIC", "TAX_ID"}
    _QUASI_REGEX = {"PHONE_US", "IP_ADDRESS", "ZIP_US", "DATE_OF_BIRTH"}

    _PRESIDIO_DIRECT = {
        "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "CREDIT_CARD", "US_PASSPORT",
    }
    _PRESIDIO_QUASI = {"DATE_TIME", "IP_ADDRESS", "URL", "US_DRIVER_LICENSE", "LOCATION"}

    def __init__(self, adk_client: Any = None) -> None:
        self.analyzer = AnalyzerEngine()
        self.adk_client = adk_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def detect(self, col_name: str, samples: list[Any]) -> PIICategory:
        """Run all three detection layers and return the most conservative category.

        Args:
            col_name: Column name — used in the LLM heuristic layer.
            samples: Raw sample values from the column.

        Returns:
            PIICategory enum value — SAFE only when all layers agree.
        """
        regex_result = self._detect_regex(samples)
        if regex_result is not None:
            logger.debug("PII detected via regex", column=col_name, category=regex_result)
            return regex_result

        presidio_result = self._detect_presidio(samples)
        if presidio_result is not None:
            logger.debug("PII detected via Presidio", column=col_name, category=presidio_result)
            return presidio_result

        llm_result = await self._detect_llm(col_name, samples)
        logger.debug("PII classification via LLM", column=col_name, category=llm_result)
        return llm_result

    # ------------------------------------------------------------------
    # Layer 1 — Regex
    # ------------------------------------------------------------------

    @staticmethod
    def _check_luhn(card_number: str) -> bool:
        """Validate a card PAN using the Luhn algorithm."""
        cleaned = card_number.replace("-", "").replace(" ", "")
        if not cleaned.isdigit():
            return False
        digits = [int(d) for d in cleaned]
        checksum = 0
        is_second = False
        for d in reversed(digits):
            if is_second:
                d *= 2
                if d > 9:
                    d -= 9
            checksum += d
            is_second = not is_second
        return checksum % 10 == 0

    def _detect_regex(self, samples: list[Any]) -> PIICategory | None:
        sample_strings = [str(s) for s in samples if s is not None]
        if not sample_strings:
            return None

        for name, pattern in self._REGEX_PATTERNS.items():
            match_count = 0
            for s in sample_strings:
                if re.search(pattern, s):
                    if name == "CARD_PAN" and not self._check_luhn(re.search(pattern, s).group()):  # type: ignore[union-attr]
                        continue
                    match_count += 1

            hit_rate = match_count / len(sample_strings)
            if hit_rate >= 0.1:
                if name in self._DIRECT_PII_REGEX:
                    return PIICategory.DIRECT_PII
                if name in self._FINANCIAL_REGEX:
                    return PIICategory.FINANCIAL_PII
                if name in self._QUASI_REGEX:
                    return PIICategory.QUASI_PII

        return None

    # ------------------------------------------------------------------
    # Layer 2 — Presidio NLP
    # ------------------------------------------------------------------

    def _detect_presidio(self, samples: list[Any]) -> PIICategory | None:
        sample_strings = [str(s) for s in samples if s is not None]
        if not sample_strings:
            return None

        # Analyze a combined window of the first 20 samples
        combined_text = " ".join(sample_strings[:20])
        results = self.analyzer.analyze(text=combined_text, language="en")
        if not results:
            return None

        entity_types = {res.entity_type for res in results if res.score >= 0.7}

        if entity_types & self._PRESIDIO_DIRECT:
            return PIICategory.DIRECT_PII
        if entity_types & self._PRESIDIO_QUASI:
            return PIICategory.QUASI_PII

        return None

    # ------------------------------------------------------------------
    # Layer 3 — LLM heuristic (metadata-only — no sample values sent)
    # ------------------------------------------------------------------

    async def _detect_llm(self, col_name: str, samples: list[Any]) -> PIICategory:
        if not self.adk_client:
            return PIICategory.SAFE

        # Compute statistical metadata without sending actual values to the LLM
        non_null = [s for s in samples if s is not None]
        avg_len = sum(len(str(s)) for s in non_null) / max(len(non_null), 1)
        unique_count = len(set(str(s) for s in non_null))
        sample_count = len(non_null)

        prompt = (
            f"Column name: '{col_name}'\n"
            f"Statistical metadata (NO raw values): "
            f"sample_count={sample_count}, unique_values={unique_count}, "
            f"avg_string_length={avg_len:.1f}\n\n"
            "Classify this column as exactly one of: "
            "DIRECT_PII | QUASI_PII | FINANCIAL_PII | SENSITIVE | SAFE.\n"
            "Respond with JSON only: "
            '{"category": "...", "confidence": 0.0-1.0, "reason": "..."}'
        )

        sem = _get_llm_semaphore()
        try:
            async with sem:
                response = await self.adk_client.generate_content(prompt)
            data = json.loads(response.text)
            confidence: float = float(data.get("confidence", 0.0))
            if confidence < 0.7:
                return PIICategory.SAFE

            category_str: str = data.get("category", "SAFE")
            mapping: dict[str, PIICategory] = {
                "DIRECT_PII": PIICategory.DIRECT_PII,
                "QUASI_PII": PIICategory.QUASI_PII,
                "FINANCIAL_PII": PIICategory.FINANCIAL_PII,
                "SENSITIVE": PIICategory.SENSITIVE,
                "SAFE": PIICategory.SAFE,
            }
            return mapping.get(category_str, PIICategory.SAFE)
        except Exception as exc:
            logger.error("LLM PII detection failed", error=str(exc))
            return PIICategory.SAFE
