"""Core data model.

Two conventions run through this module and matter downstream.

**Entities carry character offsets.** A UI highlighting entities in the original text
needs spans, not strings, and a string match would highlight the wrong occurrence when a
name appears twice.

**Every analyzer can abstain.** A sentiment score on text with no sentiment-bearing words
is noise, and a QA answer the document does not support is worse than no answer. Both are
representable here rather than being forced into a confident-looking default.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SentimentLabel = Literal["positive", "negative", "neutral", "mixed"]
EntityType = Literal[
    "PERSON",
    "ORG",
    "LOCATION",
    "DATE",
    "MONEY",
    "PERCENT",
    "EMAIL",
    "URL",
]
AnalysisKind = Literal["summary", "sentiment", "entities", "qa"]


class Document(BaseModel):
    """A text document submitted for analysis."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    title: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def n_chars(self) -> int:
        return len(self.text)


class Summary(BaseModel):
    """An extractive or generated summary."""

    model_config = ConfigDict(extra="forbid")

    text: str
    sentences: list[str] = Field(default_factory=list)
    method: str = ""
    compression_ratio: float | None = None


class SentimentResult(BaseModel):
    """A sentiment judgement, with the evidence that produced it."""

    model_config = ConfigDict(extra="forbid")

    label: SentimentLabel
    #: -1.0 (wholly negative) to +1.0 (wholly positive).
    score: float = Field(ge=-1.0, le=1.0)
    #: 0.0 when nothing in the text carried sentiment. A neutral label with zero
    #: confidence means "no signal", which is different from "balanced signal".
    confidence: float = Field(ge=0.0, le=1.0)
    positive_terms: list[str] = Field(default_factory=list)
    negative_terms: list[str] = Field(default_factory=list)
    negations: int = 0
    method: str = ""


class Entity(BaseModel):
    """A named entity and the span it occupies in the source text."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    type: EntityType
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _span_is_ordered(self) -> Entity:
        if self.end <= self.start:
            raise ValueError(
                f"entity {self.text!r}: end ({self.end}) must exceed start ({self.start})"
            )
        return self

    @property
    def span(self) -> tuple[int, int]:
        return (self.start, self.end)

    def overlaps(self, other: Entity) -> bool:
        return self.start < other.end and other.start < self.end


class QAAnswer(BaseModel):
    """An answer to a question about a document.

    ``supported`` is false when the document does not answer the question. An unsupported
    answer still carries whatever the analyzer found, so a caller can show it as a weak
    match rather than being told nothing at all.
    """

    model_config = ConfigDict(extra="forbid")

    question: str
    answer: str
    supported: bool
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    method: str = ""


class Analysis(BaseModel):
    """Everything asked for about one document, plus per-stage timing."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    title: str = ""
    n_chars: int = 0
    summary: Summary | None = None
    sentiment: SentimentResult | None = None
    entities: list[Entity] = Field(default_factory=list)
    qa: QAAnswer | None = None
    #: Milliseconds per analyzer, so a slow stage is attributable rather than hidden in a
    #: single total.
    timings_ms: dict[str, float] = Field(default_factory=dict)
    methods: dict[str, str] = Field(default_factory=dict)
    cached: bool = False

    @property
    def total_ms(self) -> float:
        return sum(self.timings_ms.values())

    @property
    def entity_types(self) -> list[str]:
        seen: list[str] = []
        for entity in self.entities:
            if entity.type not in seen:
                seen.append(entity.type)
        return seen


class SearchMatch(BaseModel):
    """A document matched by library search."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    title: str
    score: float
    snippet: str
    rank: int = Field(ge=1)


# --- evaluation ------------------------------------------------------------------


class SummaryMetrics(BaseModel):
    """Lexical overlap against reference summaries.

    ROUGE measures word overlap, not meaning. A correct paraphrase scores badly. It is
    here because it is free and comparable across methods, and the docs say plainly what
    it does not capture.
    """

    model_config = ConfigDict(extra="forbid")

    method: str
    n_examples: int = Field(ge=0)
    rouge1_f1: float = 0.0
    rouge2_f1: float = 0.0
    rougeL_f1: float = 0.0
    mean_compression: float = 0.0


class SentimentMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: str
    n_examples: int = Field(ge=0)
    accuracy: float = 0.0
    macro_f1: float = 0.0
    per_label_f1: dict[str, float] = Field(default_factory=dict)
    n_abstained: int = 0
    confusion: dict[str, dict[str, int]] = Field(default_factory=dict)


class EntityMetrics(BaseModel):
    """Entity extraction scored two ways.

    ``strict`` requires the span *and* the type to match; ``partial`` requires overlapping
    spans with the same type. The gap between them is the share of entities found with the
    wrong boundaries, which is a different problem from not finding them at all.
    """

    model_config = ConfigDict(extra="forbid")

    method: str
    n_examples: int = Field(ge=0)
    n_gold: int = 0
    n_predicted: int = 0
    strict_precision: float = 0.0
    strict_recall: float = 0.0
    strict_f1: float = 0.0
    partial_precision: float = 0.0
    partial_recall: float = 0.0
    partial_f1: float = 0.0
    per_type_f1: dict[str, float] = Field(default_factory=dict)


class QAMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: str
    n_examples: int = Field(ge=0)
    exact_match: float = 0.0
    token_f1: float = 0.0
    #: Share of answerable questions the analyzer wrongly declined.
    false_abstention_rate: float = 0.0
    #: Share of unanswerable questions the analyzer correctly declined.
    true_abstention_rate: float = 0.0


class RepairMetrics(BaseModel):
    """How often malformed model output could be recovered.

    The interesting number is not the parse rate but which repair each case needed, since
    that says what the model is actually doing wrong.
    """

    model_config = ConfigDict(extra="forbid")

    n_cases: int = Field(ge=0)
    n_parsed_directly: int = 0
    n_parsed_after_repair: int = 0
    n_failed: int = 0
    repairs_used: dict[str, int] = Field(default_factory=dict)
    failures: list[str] = Field(default_factory=list)

    @property
    def recovery_rate(self) -> float | None:
        """Share of all cases that ended up parsed, or ``None`` when there were none."""
        if self.n_cases == 0:
            return None
        return (self.n_parsed_directly + self.n_parsed_after_repair) / self.n_cases
