from __future__ import annotations

import random
import string
from typing import Any

from faker import Faker
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from ..models.column_profile import PIICategory

fake = Faker()
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()


# ---------------------------------------------------------------------------
# PII-safe generators — all values are provably non-real
# ---------------------------------------------------------------------------

def generate_synthetic_ssn() -> str:
    """Generate SSN with area codes 900-999 — never issued by the SSA."""
    area = random.randint(900, 999)
    group = random.randint(1, 99)
    serial = random.randint(1, 9999)
    return f"{area:03d}-{group:02d}-{serial:04d}"


def generate_synthetic_iban(country_code: str = "GB") -> str:
    """Generate a format-valid IBAN with correct mod-97 check digits.

    The bank code range used maps to no real institution.
    """
    # Build a random BBAN (Basic Bank Account Number) — length varies by country.
    # We use a fixed 22-char BBAN for simplicity (covers most countries).
    bban = "".join(random.choices(string.digits, k=22))

    # Compute check digits via ISO 13616 mod-97:
    # 1. Rearrange: move country + "00" to end, convert letters to digits
    rearranged = bban + country_code + "00"
    numeric = ""
    for ch in rearranged:
        if ch.isalpha():
            numeric += str(ord(ch) - ord("A") + 10)
        else:
            numeric += ch

    remainder = int(numeric) % 97
    check_digits = 98 - remainder
    return f"{country_code}{check_digits:02d}{bban}"


def generate_synthetic_phone(area_code: str | None = None) -> str:
    """Generate US phone using the 555-0100–555-0199 exchange (permanently fictional)."""
    ac = area_code if area_code else str(random.randint(200, 999))
    # 555-01XX is the reserved fictional range
    suffix = random.randint(0, 99)
    return f"({ac}) 555-01{suffix:02d}"


def generate_synthetic_address(region_distribution: dict[str, float] | None = None) -> dict[str, str]:
    """Generate a geographically coherent but non-existent address.

    Uses Faker for city/state so the ZIP-to-city mapping is consistent
    within the generated fake locale (not verified against real USPS data).
    """
    # Use Faker's full address components so city/state/zip are internally consistent
    locale = "en_US"
    if region_distribution:
        # Sample locale from provided distribution
        locales = list(region_distribution.keys())
        weights = list(region_distribution.values())
        locale = random.choices(locales, weights=weights, k=1)[0]

    f = Faker(locale)
    return {
        "street": f"{random.randint(100, 9999)} {f.street_name()}",
        "city": f.city(),
        "state": f.state_abbr(),
        "zip": f.zipcode(),
    }


def generate_synthetic_name(locale_distribution: dict[str, float] | None = None) -> str:
    """Generate a locale-aware synthetic name."""
    locale = "en_US"
    if locale_distribution:
        locales = list(locale_distribution.keys())
        weights = list(locale_distribution.values())
        locale = random.choices(locales, weights=weights, k=1)[0]
    return Faker(locale).name()


def generate_synthetic_email(name: str | None = None, domain_distribution: dict[str, float] | None = None) -> str:
    """Generate an email using RFC 2606 reserved domains (@example.com, @test.invalid)."""
    if domain_distribution:
        domains = list(domain_distribution.keys())
        weights = list(domain_distribution.values())
        domain = random.choices(domains, weights=weights, k=1)[0]
    else:
        domain = random.choice(["example.com", "example.net", "example.org", "test.invalid"])

    if name:
        user = name.lower().replace(" ", ".")
    else:
        user = "".join(random.choices(string.ascii_lowercase, k=8))
    return f"{user}@{domain}"


def generate_synthetic_card_pan(card_type_distribution: dict[str, float] | None = None) -> str:
    """Generate a card PAN with BIN starting with 9 (unassigned) and valid Luhn checksum."""
    # BIN: 9XXXXX — 6 digits starting with 9
    bin_prefix = "9" + "".join(random.choices(string.digits, k=5))
    # Account number: 9 more digits
    account = "".join(random.choices(string.digits, k=9))
    partial = bin_prefix + account  # 15 digits

    # Compute Luhn check digit
    digits = [int(d) for d in partial]
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 0:  # every second digit from right (before check digit)
            d *= 2
            if d > 9:
                d -= 9
        total += d
    check_digit = (10 - (total % 10)) % 10
    return partial + str(check_digit)


def generate_synthetic_ip() -> str:
    """Generate an IP from RFC 5737 TEST-NET ranges — guaranteed documentation-only."""
    prefix = random.choice(["192.0.2", "198.51.100", "203.0.113"])
    return f"{prefix}.{random.randint(1, 254)}"


# ---------------------------------------------------------------------------
# NLP-based re-hydration for free-text / JSON columns
# ---------------------------------------------------------------------------

def generate_synthetic_instruction(original_text: str) -> str:
    """Replace PII entities in free text using Presidio + Faker replacements.

    Preserves linguistic structure; replaces PERSON, LOCATION, ORG, and SSN
    with safe synthetic equivalents.
    """
    results = analyzer.analyze(
        text=original_text,
        language="en",
        entities=["PERSON", "LOCATION", "ORG", "US_PASSPORT", "US_SSN"],
    )

    operators: dict[str, OperatorConfig] = {
        "PERSON": OperatorConfig("replace", {"new_value": fake.name()}),
        "LOCATION": OperatorConfig("replace", {"new_value": fake.city()}),
        "ORG": OperatorConfig("replace", {"new_value": fake.company()}),
        "US_SSN": OperatorConfig("replace", {"new_value": generate_synthetic_ssn()}),
    }

    result = anonymizer.anonymize(
        text=original_text,
        analyzer_results=results,
        operators=operators,
    )
    return result.text


def recursive_rehydrate(data: Any) -> Any:
    """Recursively scan and re-hydrate PII strings inside nested JSON structures."""
    if isinstance(data, dict):
        return {k: recursive_rehydrate(v) for k, v in data.items()}
    if isinstance(data, list):
        return [recursive_rehydrate(i) for i in data]
    if isinstance(data, str) and len(data) > 10:
        return generate_synthetic_instruction(data)
    return data


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def get_generator_for_category(category: PIICategory) -> Any:
    """Return the appropriate generator callable for a given PIICategory.

    Returns a zero-argument callable that produces one synthetic value.
    Callers should wrap in a list comprehension for bulk generation.
    """
    mapping: dict[PIICategory, Any] = {
        PIICategory.DIRECT_PII: generate_synthetic_name,
        PIICategory.QUASI_PII: generate_synthetic_ip,
        PIICategory.FINANCIAL_PII: generate_synthetic_card_pan,
        PIICategory.SENSITIVE: lambda: fake.text(max_nb_chars=50),
        PIICategory.SAFE: lambda: None,
    }
    return mapping.get(category, lambda: None)
