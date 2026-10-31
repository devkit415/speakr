"""safe_json_loads accepts JSON only (#532).

Model output is untrusted. A Python literal (single quotes, True/None, a tuple
or a set) is not JSON, and is not evaluated: the fallback value is returned.
"""
import pytest

from src.utils.json_parser import safe_json_loads

_SENTINEL = object()


@pytest.mark.parametrize("text", [
    "{'title': 'Simple', 'summary': 'Test'}",
    "{'flag': True, 'value': None}",
    "{'a': (1, 2)}",
    "{'s': {1, 2}}",
    "[(1, 2)]",
    "(1, 2, 3)",
])
def test_a_python_literal_is_not_parsed(text):
    assert safe_json_loads(text, fallback_value=_SENTINEL) is _SENTINEL


@pytest.mark.parametrize("text,expected", [
    ('{"title": "Complete", "summary": "ok"}', {"title": "Complete", "summary": "ok"}),
    ('[1, 2, 3]', [1, 2, 3]),
    ('```json\n{"title": "Test"}\n```', {"title": "Test"}),
    ('Before {"title": "Extracted"} after', {"title": "Extracted"}),
])
def test_json_is_still_parsed(text, expected):
    assert safe_json_loads(text, fallback_value=_SENTINEL) == expected


@pytest.mark.parametrize("text", ["not json at all", ""])
def test_text_that_is_not_json_returns_the_fallback(text):
    assert safe_json_loads(text, fallback_value=_SENTINEL) is _SENTINEL
