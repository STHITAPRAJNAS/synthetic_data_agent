import re
from synthetic_data_agent.pii.generators import (
    generate_synthetic_ssn,
    generate_synthetic_email,
    generate_synthetic_card_pan,
    generate_synthetic_ip
)

def test_generate_ssn():
    ssn = generate_synthetic_ssn()
    assert re.match(r"^9\d{2}-\d{2}-\d{4}$", ssn)

def test_generate_email():
    email = generate_synthetic_email("John Doe")
    assert email.startswith("john.doe@")
    assert any(domain in email for domain in ["example.com", "example.net", "example.org", "test.invalid"])

def test_generate_card_pan():
    pan = generate_synthetic_card_pan()
    assert pan.startswith("9")
    assert len(pan) >= 15
    
    # Check Luhn
    digits = [int(d) for d in pan]
    odd_digits = digits[-1::-2]
    even_digits = digits[-2::-2]
    checksum = sum(odd_digits)
    for d in even_digits:
        checksum += sum(divmod(d * 2, 10))
    assert checksum % 10 == 0

def test_generate_ip():
    ip = generate_synthetic_ip()
    assert any(ip.startswith(prefix) for prefix in ["192.0.2", "198.51.100", "203.0.113"])
