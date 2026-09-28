"""Structured output: response schemas, and recovering malformed model JSON.

Asking a model for JSON and calling ``json.loads`` on the result works most of the time,
and the remaining few percent is where production RAG and analysis pipelines actually
break. The failures are not random -- they are the same handful every time:

* the JSON wrapped in a Markdown fence
* a sentence of preamble before the object, or a sign-off after it
* smart quotes, because the model was writing prose a moment earlier
* Python literals: ``True``, ``False``, ``None``
* a trailing comma before ``}`` or ``]``
* single-quoted keys and strings throughout

Each has a deterministic fix. This module applies them in order, records which ones were
needed, and validates the result against a pydantic model. Recording the repair matters
as much as performing it: a pipeline quietly repairing 40% of its responses has a prompt
problem, and it will never find out if the repair is silent.

Repairs are ordered least- to most-invasive, and each is only attempted if the previous
attempt failed, so well-formed output is never touched.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from analyzer.types import EntityType, SentimentLabel

T = TypeVar("T", bound=BaseModel)

_FENCE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\s*\n(.*?)\n?\s*```", re.DOTALL)
_SMART_QUOTES = {
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "«": '"',
    "»": '"',
}


# --- response schemas -------------------------------------------------------------


class SummaryResponse(BaseModel):
    """What a model is asked to return for summarisation."""

    model_config = ConfigDict(extra="ignore")

    summary: str = Field(min_length=1)
    key_points: list[str] = Field(default_factory=list)


class SentimentResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    label: SentimentLabel
    score: float = Field(ge=-1.0, le=1.0)
    rationale: str = ""


class EntityItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str = Field(min_length=1)
    type: EntityType


class EntitiesResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    entities: list[EntityItem] = Field(default_factory=list)


class QAResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    answer: str = Field(min_length=1)
    supported: bool
    evidence: list[str] = Field(default_factory=list)


#: ``extra="ignore"`` on every schema above is deliberate. A model that adds a field is
#: being helpful, not wrong, and rejecting the whole response over it would trade a
#: usable answer for a strictness that buys nothing here. Missing or mistyped *required*
#: fields still fail, which is the part that matters.


# --- repair -----------------------------------------------------------------------

RepairName = Literal[
    "strip_code_fence",
    "extract_balanced",
    "normalize_quotes",
    "python_literals",
    "remove_trailing_commas",
    "single_to_double_quotes",
]


@dataclass(frozen=True)
class ParseOutcome(Generic[T]):
    """The result of parsing model output into a schema."""

    ok: bool
    value: T | None = None
    repairs: tuple[str, ...] = field(default_factory=tuple)
    error: str = ""
    raw_json: str = ""

    @property
    def needed_repair(self) -> bool:
        return bool(self.repairs)

    def unwrap(self) -> T:
        if self.value is None:
            raise ValueError(f"cannot unwrap a failed parse: {self.error}")
        return self.value


def strip_code_fence(text: str) -> str:
    """Return the contents of the first fenced block, or the text unchanged."""
    match = _FENCE.search(text)
    return match.group(1) if match else text


def extract_balanced(text: str) -> str:
    """Return the first balanced ``{...}`` or ``[...]``, ignoring surrounding prose.

    String-aware: a brace inside a string literal does not change the depth, and an
    escaped quote does not end the string. A regex cannot do this correctly, and the cases
    where it fails are exactly the ones worth handling -- JSON containing prose that
    contains punctuation.
    """
    start = next((i for i, char in enumerate(text) if char in "{["), None)
    if start is None:
        return text

    opening = text[start]
    closing = "}" if opening == "{" else "]"
    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return text[start:]


def normalize_quotes(text: str) -> str:
    """Replace typographic quotes with ASCII ones."""
    for smart, plain in _SMART_QUOTES.items():
        text = text.replace(smart, plain)
    return text


def python_literals(text: str) -> str:
    """Rewrite ``True``/``False``/``None`` outside strings as JSON literals."""
    return _outside_strings(
        text,
        lambda segment: re.sub(
            r"\b(True|False|None)\b",
            lambda m: {"True": "true", "False": "false", "None": "null"}[m.group(1)],
            segment,
        ),
    )


def remove_trailing_commas(text: str) -> str:
    """Drop a comma that immediately precedes ``}`` or ``]``, outside strings."""
    return _outside_strings(text, lambda segment: re.sub(r",(\s*[}\]])", r"\1", segment))


def single_to_double_quotes(text: str) -> str:
    """Convert single-quoted JSON to double-quoted.

    Only attempted when the text contains no double quotes at all, i.e. the model used
    single quotes throughout. Otherwise an apostrophe inside a legitimate double-quoted
    string would be rewritten and the damage would be worse than the original error.
    """
    if '"' in text:
        return text
    return text.replace("'", '"')


def _outside_strings(text: str, transform: Any) -> str:
    """Apply ``transform`` to the parts of ``text`` that are not inside a JSON string.

    Splitting first is what keeps a comma or the word "None" inside a prose value from
    being rewritten. Doing these substitutions over the whole string is the obvious
    implementation and it corrupts data.
    """
    pieces: list[str] = []
    buffer: list[str] = []
    in_string = False
    escaped = False

    for char in text:
        if in_string:
            buffer.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                pieces.append("".join(buffer))
                buffer = []
                in_string = False
            continue
        if char == '"':
            pieces.append(transform("".join(buffer)))
            buffer = [char]
            in_string = True
            continue
        buffer.append(char)

    tail = "".join(buffer)
    pieces.append(tail if in_string else transform(tail))
    return "".join(pieces)


#: Ordered least- to most-invasive. Each is applied cumulatively on top of the previous.
REPAIR_STEPS: tuple[tuple[str, Any], ...] = (
    ("strip_code_fence", strip_code_fence),
    ("extract_balanced", extract_balanced),
    ("normalize_quotes", normalize_quotes),
    ("python_literals", python_literals),
    ("remove_trailing_commas", remove_trailing_commas),
    ("single_to_double_quotes", single_to_double_quotes),
)


def parse_json(text: str) -> tuple[Any, tuple[str, ...], str]:
    """Parse JSON, repairing if needed. Returns (value, repairs applied, json text).

    Raises ``ValueError`` when no repair sequence produces valid JSON.
    """
    if not text.strip():
        raise ValueError("model returned empty output")

    try:
        return json.loads(text), (), text
    except json.JSONDecodeError:
        pass

    applied: list[str] = []
    candidate = text
    last_error = ""

    for name, step in REPAIR_STEPS:
        repaired = step(candidate)
        if repaired != candidate:
            applied.append(name)
            candidate = repaired
        try:
            return json.loads(candidate), tuple(applied), candidate
        except json.JSONDecodeError as exc:
            last_error = exc.msg

    raise ValueError(f"could not parse model output as JSON after repairs: {last_error}")


def parse_model_output(text: str, model: type[T]) -> ParseOutcome[T]:
    """Parse model output into ``model``, reporting which repairs were needed.

    Never raises: a malformed response is data about the model, and a pipeline processing
    a thousand documents should record it and continue rather than stop.
    """
    try:
        value, repairs, raw = parse_json(text)
    except ValueError as exc:
        return ParseOutcome(ok=False, error=str(exc))

    if not isinstance(value, dict):
        return ParseOutcome(
            ok=False,
            repairs=repairs,
            error=f"expected a JSON object, got {type(value).__name__}",
            raw_json=raw,
        )

    try:
        parsed = model.model_validate(value)
    except ValidationError as exc:
        first = exc.errors()[0]
        location = ".".join(str(part) for part in first["loc"]) or "<root>"
        return ParseOutcome(
            ok=False,
            repairs=repairs,
            error=f"schema violation at {location}: {first['msg']}",
            raw_json=raw,
        )

    return ParseOutcome(ok=True, value=parsed, repairs=repairs, raw_json=raw)
