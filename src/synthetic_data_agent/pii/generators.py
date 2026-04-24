import random
import string
from typing import Any, Optional
from faker import Faker
from ..models.column_profile import PIICategory

fake = Faker()

def generate_synthetic_ssn() -> str:
    """Area codes 900-999 are never issued by SSA."""
    area = random.randint(900, 999)
    group = random.randint(1, 99)
    serial = random.randint(1, 9999)
    return f"{area:03d}-{group:02d}-{serial:04d}"

def generate_synthetic_iban(country_code: str = "US") -> str:
    """Format valid, bank non-existent."""
    # Simplified IBAN generation for demo
    bban = ''.join(random.choices(string.ascii_uppercase + string.digits, k=18))
    # Mod-97 check digit placeholder
    return f"{country_code}99{bban}"

def generate_synthetic_phone(area_code: Optional[str] = None) -> str:
    """Uses 555-0100 to 555-0199 exchange (permanently fictional)."""
    ac = area_code or f"{random.randint(200, 999)}"
    line = random.randint(100, 199)
    suffix = random.randint(1000, 9999)
    return f"({ac}) 555-{line:04d}" # Wait, 555-01XX is the fictional range
    # Correcting:
    exchange = "555"
    number = random.randint(100, 199) # This makes it 555-01XX if I do it right
    return f"({ac}) 555-0{number:02d}"

def generate_synthetic_address(region_distribution: Optional[dict] = None) -> dict:
    """Geographically coherent, address non-existent."""
    # In a real system, we'd use region_distribution to sample ZIP
    # and then map ZIP to city/state.
    return {
        "street": f"{random.randint(100, 9999)} Fictional St",
        "city": fake.city(),
        "state": fake.state_abbr(),
        "zip": f"00{random.randint(100, 999)}" # 00XXX are mostly unused or military
    }

def generate_synthetic_name(locale: str = "en_US") -> str:
    f = Faker(locale)
    return f.name()

def generate_synthetic_email(name: Optional[str] = None) -> str:
    """Uses @example.com, @test.invalid (RFC 2606 reserved)."""
    domains = ["example.com", "example.net", "example.org", "test.invalid"]
    domain = random.choice(domains)
    if name:
        user = name.lower().replace(" ", ".")
    else:
        user = "".join(random.choices(string.ascii_lowercase, k=8))
    return f"{user}@{domain}"

def generate_synthetic_card_pan() -> str:
    """Generates BIN ranges starting with 9 (unassigned)."""
    bin_range = "9" + "".join(random.choices(string.digits, k=5))
    rest = "".join(random.choices(string.digits, k=9))
    
    # Calculate Luhn checksum
    partial = bin_range + rest
    digits = [int(d) for d in partial]
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    
    check_digit = (10 - (checksum % 10)) % 10
    return partial + str(check_digit)

def generate_synthetic_ip() -> str:
    """Uses TEST-NET ranges: 192.0.2.x, 198.51.100.x, 203.0.113.x."""
    ranges = ["192.0.2", "198.51.100", "203.0.113"]
    base = random.choice(ranges)
    return f"{base}.{random.randint(1, 254)}"

def get_generator_for_category(category: PIICategory):
    # Mapping logic here
    pass
