import re
import json
from typing import Any, Optional
import pandas as pd
from presidio_analyzer import AnalyzerEngine
from ..models.column_profile import PIICategory
from ..config import settings
import structlog

logger = structlog.get_logger()

class PIIDetector:
    def __init__(self, adk_client: Any = None):
        self.analyzer = AnalyzerEngine()
        self.adk_client = adk_client
        self.regex_patterns = {
            "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
            "PHONE_US": r"\b(\+?1[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b",
            "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "IP_ADDRESS": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
            "ZIP_US": r"\b\d{5}(-\d{4})?\b",
            "CARD_PAN": r"\b\d{13,19}\b",
            # Add more as needed
        }

    def _check_luhn(self, card_number: str) -> bool:
        card_number = card_number.replace("-", "").replace(" ", "")
        if not card_number.isdigit():
            return False
        digits = [int(d) for d in card_number]
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

    async def detect(self, col_name: str, samples: list[Any]) -> PIICategory:
        """Multi-layer detection strategy."""
        
        # Layer 1: Regex
        regex_result = self._detect_regex(samples)
        if regex_result:
            return regex_result

        # Layer 2: Presidio
        presidio_result = self._detect_presidio(samples)
        if presidio_result:
            return presidio_result

        # Layer 3: LLM Heuristic
        llm_result = await self._detect_llm(col_name, samples)
        return llm_result

    def _detect_regex(self, samples: list[Any]) -> Optional[PIICategory]:
        sample_strings = [str(s) for s in samples if s is not None]
        if not sample_strings:
            return None

        for name, pattern in self.regex_patterns.items():
            matches = 0
            for s in sample_strings:
                if re.search(pattern, s):
                    if name == "CARD_PAN" and not self._check_luhn(s):
                        continue
                    matches += 1
            
            if matches / len(sample_strings) >= 0.1: # 10% threshold
                if name in ["SSN", "CARD_PAN", "EMAIL"]:
                    return PIICategory.DIRECT_PII
                return PIICategory.QUASI_PII
        
        return None

    def _detect_presidio(self, samples: list[Any]) -> Optional[PIICategory]:
        sample_strings = [str(s) for s in samples if s is not None]
        if not sample_strings:
            return None
            
        combined_text = " ".join(sample_strings[:20]) # Analyze first 20 samples
        results = self.analyzer.analyze(text=combined_text, language='en')
        
        if not results:
            return None
            
        # Map Presidio entities to PIICategory
        direct_pii_entities = {"PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "LOCATION", "US_SSN", "CREDIT_CARD", "US_PASSPORT"}
        quasi_pii_entities = {"DATE_TIME", "IP_ADDRESS", "URL", "US_DRIVER_LICENSE"}
        
        entity_types = {res.entity_type for res in results if res.score >= 0.7}
        
        if entity_types & direct_pii_entities:
            return PIICategory.DIRECT_PII
        if entity_types & quasi_pii_entities:
            return PIICategory.QUASI_PII
            
        return None

    async def _detect_llm(self, col_name: str, samples: list[Any]) -> PIICategory:
        if not self.adk_client:
            # Fallback if no LLM client is provided
            return PIICategory.SAFE

        prompt = f"""
        Column name: '{col_name}'
        Sample values: {samples[:10]}
        
        Classify this column as one of the following PII categories:
        - DIRECT_PII: Names, SSN, Emails, precise locations.
        - QUASI_PII: Dates, ZIP codes, gender, race (identifiable when combined).
        - FINANCIAL_PII: Bank accounts, credit card numbers, salary.
        - SENSITIVE: Health data, political opinions, religion.
        - SAFE: Generic IDs, timestamps, counts, non-identifying metadata.
        
        Respond with JSON only:
        {{
            "category": "PIICategory",
            "confidence": 0.0-1.0,
            "reason": "short explanation"
        }}
        """
        
        try:
            # This is a placeholder for actual ADK/Gemini call
            # In a real implementation, we'd use self.adk_client.generate(prompt)
            response = await self.adk_client.generate_content(prompt)
            data = json.loads(response.text)
            category_str = data.get("category", "SAFE")
            
            # Map string to Enum
            mapping = {
                "DIRECT_PII": PIICategory.DIRECT_PII,
                "QUASI_PII": PIICategory.QUASI_PII,
                "FINANCIAL_PII": PIICategory.FINANCIAL_PII,
                "SENSITIVE": PIICategory.SENSITIVE,
                "SAFE": PIICategory.SAFE
            }
            return mapping.get(category_str, PIICategory.SAFE)
        except Exception as e:
            logger.error("LLM detection failed", error=str(e))
            return PIICategory.SAFE
