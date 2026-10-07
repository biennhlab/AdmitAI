from datetime import datetime, timedelta
from unittest.mock import patch

from src.auth import security


def test_explicit_zero_token_lifetime_is_not_replaced_by_default_expiry():
    captured = {}

    def encode(payload, _secret, algorithm):
        captured.update(payload)
        assert algorithm == "HS256"
        return "token"

    before = datetime.utcnow()
    with patch.object(security.jwt, "encode", side_effect=encode):
        token = security.create_access_token({"sub": "admin"}, timedelta(0))
    after = datetime.utcnow()

    assert token == "token"
    assert before <= captured["exp"] <= after
