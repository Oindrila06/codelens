import hashlib
import hmac

from codelens.security import verify_signature


def sign(secret: str, body: bytes) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def test_valid_signature():
    assert verify_signature("s", b"{}", sign("s", b"{}"))


def test_wrong_secret_rejected():
    assert not verify_signature("s", b"{}", sign("other", b"{}"))


def test_tampered_body_rejected():
    assert not verify_signature("s", b'{"a":1}', sign("s", b"{}"))


def test_missing_or_malformed_header_rejected():
    assert not verify_signature("s", b"{}", None)
    assert not verify_signature("s", b"{}", "sha1=abc")
    assert not verify_signature("s", b"{}", "")


def test_empty_secret_never_validates():
    assert not verify_signature("", b"{}", sign("", b"{}"))
