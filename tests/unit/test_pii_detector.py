import pytest
from unittest.mock import AsyncMock, MagicMock
from synthetic_data_agent.pii.detector import PIIDetector
from synthetic_data_agent.models.column_profile import PIICategory

@pytest.mark.asyncio
async def test_regex_detection_ssn():
    detector = PIIDetector()
    samples = ["900-12-3456", "999-99-9999", "not-a-ssn"]
    result = await detector.detect("social_security", samples)
    assert result == PIICategory.DIRECT_PII

@pytest.mark.asyncio
async def test_regex_detection_email():
    detector = PIIDetector()
    samples = ["test@example.com", "user.name+tag@gmail.co.uk", "invalid-email"]
    result = await detector.detect("email_address", samples)
    assert result == PIICategory.DIRECT_PII

@pytest.mark.asyncio
async def test_luhn_validation():
    detector = PIIDetector()
    # Valid Visa test number
    assert detector._check_luhn("4111111111111111") is True
    # Invalid
    assert detector._check_luhn("4111111111111112") is False

@pytest.mark.asyncio
async def test_llm_fallback():
    mock_adk = AsyncMock()
    mock_response = MagicMock()
    mock_response.text = '{"category": "SENSITIVE", "confidence": 0.9, "reason": "Health data"}'
    mock_adk.generate_content.return_value = mock_response
    
    detector = PIIDetector(adk_client=mock_adk)
    samples = ["Cancer", "Diabetes", "Healthy"]
    result = await detector.detect("medical_condition", samples)
    
    assert result == PIICategory.SENSITIVE
    mock_adk.generate_content.assert_called_once()
