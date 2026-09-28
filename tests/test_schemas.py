"""Tests for JSON extraction and repair.

This is the module most likely to be reused elsewhere, so it gets the most direct tests:
each repair in isolation, each in the pipeline, and the cases where repairing would be
worse than failing.
"""

from __future__ import annotations

import pytest

from analyzer.schemas import (
    EntitiesResponse,
    QAResponse,
    SentimentResponse,
    SummaryResponse,
    extract_balanced,
    normalize_quotes,
    parse_json,
    parse_model_output,
    python_literals,
    remove_trailing_commas,
    single_to_double_quotes,
    strip_code_fence,
)

# --- individual repairs -----------------------------------------------------------


def test_strip_code_fence_returns_the_block_contents() -> None:
    assert strip_code_fence('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_strip_code_fence_handles_an_unlabelled_fence() -> None:
    assert strip_code_fence('```\n{"a": 1}\n```') == '{"a": 1}'


def test_strip_code_fence_leaves_unfenced_text_alone() -> None:
    assert strip_code_fence('{"a": 1}') == '{"a": 1}'


def test_extract_balanced_discards_surrounding_prose() -> None:
    raw = 'Sure! Here you go:\n{"a": 1}\nHope that helps.'
    assert extract_balanced(raw) == '{"a": 1}'


def test_extract_balanced_ignores_braces_inside_strings() -> None:
    """A regex cannot do this, and these are the cases that matter."""
    raw = 'text {"a": "a } brace and a { brace"} tail'
    assert extract_balanced(raw) == '{"a": "a } brace and a { brace"}'


def test_extract_balanced_respects_escaped_quotes() -> None:
    raw = r'{"a": "he said \"hi\" }"} tail'
    assert extract_balanced(raw) == r'{"a": "he said \"hi\" }"}'


def test_extract_balanced_handles_an_array_root() -> None:
    assert extract_balanced("prefix [1, 2] suffix") == "[1, 2]"


def test_extract_balanced_returns_the_text_when_there_is_no_bracket() -> None:
    assert extract_balanced("no json here") == "no json here"


def test_normalize_quotes_replaces_typographic_quotes() -> None:
    assert normalize_quotes("“key”") == '"key"'


def test_python_literals_are_rewritten() -> None:
    assert (
        python_literals('{"a": True, "b": False, "c": None}')
        == '{"a": true, "b": false, "c": null}'
    )


def test_python_literals_inside_a_string_are_left_alone() -> None:
    # Rewriting these would corrupt a prose value, which is worse than failing to parse.
    assert python_literals('{"a": "True story about None"}') == '{"a": "True story about None"}'


def test_trailing_commas_are_removed() -> None:
    assert remove_trailing_commas('{"a": [1, 2,],}') == '{"a": [1, 2]}'


def test_a_comma_inside_a_string_is_left_alone() -> None:
    assert remove_trailing_commas('{"a": "one, two ]"}') == '{"a": "one, two ]"}'


def test_single_quotes_are_converted_when_used_throughout() -> None:
    assert single_to_double_quotes("{'a': 'b'}") == '{"a": "b"}'


def test_single_quote_conversion_is_skipped_when_double_quotes_are_present() -> None:
    """An apostrophe inside a real string must not be rewritten.

    Converting here would turn a valid document into an invalid one, so the repair
    declines rather than guessing.
    """
    raw = '{"a": "it\'s fine"}'
    assert single_to_double_quotes(raw) == raw


# --- the pipeline -----------------------------------------------------------------


def test_well_formed_json_needs_no_repair() -> None:
    value, repairs, _ = parse_json('{"a": 1}')
    assert value == {"a": 1}
    assert repairs == ()


def test_repairs_are_reported_in_the_order_applied() -> None:
    value, repairs, _ = parse_json('```json\n{"a": 1,}\n```')
    assert value == {"a": 1}
    assert repairs == ("strip_code_fence", "remove_trailing_commas")


def test_empty_output_is_an_error() -> None:
    with pytest.raises(ValueError, match="empty output"):
        parse_json("   ")


def test_unrecoverable_output_raises() -> None:
    with pytest.raises(ValueError, match="could not parse"):
        parse_json('{"a": ')


# --- schema validation ------------------------------------------------------------


def test_a_valid_response_parses_into_its_schema() -> None:
    outcome = parse_model_output('{"summary": "Short.", "key_points": ["one"]}', SummaryResponse)
    assert outcome.ok
    assert outcome.unwrap().summary == "Short."
    assert not outcome.needed_repair


def test_a_repaired_response_records_what_was_needed() -> None:
    outcome = parse_model_output("{'summary': 'Single quoted.'}", SummaryResponse)
    assert outcome.ok
    assert "single_to_double_quotes" in outcome.repairs


def test_an_extra_field_is_tolerated() -> None:
    # A model adding a field is being helpful; rejecting the whole response would trade a
    # usable answer for strictness that buys nothing.
    outcome = parse_model_output('{"summary": "Fine.", "confidence": 0.9}', SummaryResponse)
    assert outcome.ok


def test_a_missing_required_field_fails_with_its_location() -> None:
    outcome = parse_model_output('{"key_points": []}', SummaryResponse)
    assert not outcome.ok
    assert "summary" in outcome.error


def test_a_label_outside_the_allowed_set_fails() -> None:
    outcome = parse_model_output('{"label": "quite good", "score": 0.5}', SentimentResponse)
    assert not outcome.ok
    assert "label" in outcome.error


def test_a_score_outside_its_range_fails() -> None:
    outcome = parse_model_output('{"label": "positive", "score": 5.0}', SentimentResponse)
    assert not outcome.ok


def test_an_entity_type_outside_the_allowed_set_fails() -> None:
    raw = '{"entities": [{"text": "Acme", "type": "COMPANY"}]}'
    outcome = parse_model_output(raw, EntitiesResponse)
    assert not outcome.ok


def test_a_json_array_at_the_root_is_rejected() -> None:
    outcome = parse_model_output("[1, 2, 3]", SummaryResponse)
    assert not outcome.ok
    assert "expected a JSON object" in outcome.error


def test_parsing_never_raises_on_unusable_output() -> None:
    # A malformed response is data about the model; a pipeline over a thousand documents
    # must record it and continue.
    outcome = parse_model_output("I cannot answer that.", QAResponse)
    assert not outcome.ok
    assert outcome.value is None


def test_unwrapping_a_failed_parse_raises() -> None:
    outcome = parse_model_output("nonsense", QAResponse)
    with pytest.raises(ValueError, match="cannot unwrap"):
        outcome.unwrap()


def test_braces_inside_a_value_survive_extraction() -> None:
    raw = '{"answer": "It says {braces} inside.", "supported": false}'
    outcome = parse_model_output(raw, QAResponse)
    assert outcome.ok
    assert outcome.unwrap().answer == "It says {braces} inside."
