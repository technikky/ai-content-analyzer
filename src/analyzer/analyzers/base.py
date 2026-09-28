"""Analyzer protocols.

Every analysis has two implementations behind one protocol: a deterministic local baseline
and an LLM-backed one. That is not symmetry for its own sake. Without a baseline there is
nothing to compare an LLM against, so "the model does summarisation" is unfalsifiable and
the API bill is unjustifiable. With one, the question becomes how much the model adds over
free, which is answerable.

The baselines are also what make the test suite hermetic: CI exercises the whole pipeline
with no key, and the LLM paths are exercised against scripted responses.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from analyzer.types import Document, Entity, QAAnswer, SentimentResult, Summary


class AnalyzerError(RuntimeError):
    """Raised when an analyzer cannot produce a result at all.

    Distinct from an analyzer that legitimately returns nothing: no entities in a document
    is a result, while a model that returned unparseable output is a failure.
    """


@runtime_checkable
class Summarizer(Protocol):
    @property
    def name(self) -> str: ...

    def summarize(self, document: Document) -> Summary: ...


@runtime_checkable
class SentimentAnalyzer(Protocol):
    @property
    def name(self) -> str: ...

    def analyze(self, text: str) -> SentimentResult: ...


@runtime_checkable
class EntityExtractor(Protocol):
    @property
    def name(self) -> str: ...

    def extract(self, text: str) -> list[Entity]:
        """Entities in document order, non-overlapping."""
        ...


@runtime_checkable
class QuestionAnswerer(Protocol):
    @property
    def name(self) -> str: ...

    def answer(self, question: str, document: Document) -> QAAnswer: ...


def resolve_overlaps(entities: list[Entity]) -> list[Entity]:
    """Keep the longest entity where spans overlap, then order by position.

    Overlaps are normal with layered extraction: an email address contains a plausible
    person name, and a company name contains a location. Longest-match-wins is the
    conventional resolution and it keeps the output usable for highlighting, which needs
    disjoint spans.
    """
    ordered = sorted(entities, key=lambda e: (e.start, -(e.end - e.start)))
    kept: list[Entity] = []
    for entity in ordered:
        if any(entity.overlaps(existing) for existing in kept):
            continue
        kept.append(entity)
    return sorted(kept, key=lambda e: e.start)
