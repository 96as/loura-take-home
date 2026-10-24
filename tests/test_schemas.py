import pytest
from pydantic import ValidationError

from app.schemas import TicketIn


def test_body_at_limit_ok():
    assert len(TicketIn(id="t-1", body="a" * 10_000).body) == 10_000


def test_body_over_limit_422():
    with pytest.raises(ValidationError):
        TicketIn(id="t-1", body="a" * 10_001)


def test_subject_at_limit_ok():
    assert len(TicketIn(id="t-1", subject="s" * 200, body="b").subject) == 200


def test_subject_over_limit_rejected():
    with pytest.raises(ValidationError):
        TicketIn(id="t-1", subject="s" * 201, body="b")


def test_id_at_limit_ok():
    assert TicketIn(id="a" * 64, body="b").id == "a" * 64


@pytest.mark.parametrize("bad_id", ["t 1", "", "a" * 65])
def test_id_bad_chars_rejected(bad_id):
    with pytest.raises(ValidationError):
        TicketIn(id=bad_id, body="b")


def test_empty_subject_ok():
    assert TicketIn(id="t-1", subject="", body="b").subject == ""
    assert TicketIn(id="t-1", body="b").subject == ""


@pytest.mark.parametrize("body", ["", "   ", "\n\t "])
def test_whitespace_body_rejected(body):
    with pytest.raises(ValidationError):
        TicketIn(id="t-1", body=body)
