"""Shared fixtures.

Every fixture is offline: the default analyzers need no model download and no API key, so
the suite exercises the real pipeline instead of mocking it. The LLM paths are exercised
against :class:`ScriptedLLM`, including deliberately malformed responses.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from analyzer.evaluation import (
    EntityExample,
    QAExample,
    RepairCase,
    SentimentExample,
    SummaryExample,
    load_jsonl,
)
from analyzer.library import TfidfLibrary
from analyzer.pipeline import AnalysisCache, ContentAnalyzer
from analyzer.types import Document

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
EVAL_DIR = DATA_DIR / "eval"
CORPUS_FILE = DATA_DIR / "corpus.jsonl"

SAMPLE_TEXT = (
    "Rania Haddad joined Northwind Systems Ltd in Penang on 14 March 2024. "
    "The onboarding was excellent and the documentation is genuinely helpful. "
    "Revenue rose 18% to $4.2 million in the first quarter. "
    "Questions go to rania@northwind.example."
)


@pytest.fixture
def sample_document() -> Document:
    """A document containing at least one entity of most supported types."""
    return Document(id="sample", text=SAMPLE_TEXT, title="Northwind update")


@pytest.fixture
def analyzer() -> ContentAnalyzer:
    """The default pipeline, with caching off so tests never share state through disk."""
    return ContentAnalyzer(cache=AnalysisCache(enabled=False))


@pytest.fixture
def library() -> TfidfLibrary:
    documents = [
        Document(
            id="kb/refunds",
            title="Refund policy",
            text="Refunds are issued to the original payment method within five working days.",
        ),
        Document(
            id="kb/tokens",
            title="Token lifetimes",
            text="Access tokens expire after one hour and refresh tokens after thirty days.",
        ),
        Document(
            id="kb/limits",
            title="Rate limits",
            text="Rate limits apply per API key rather than per account, returning 429.",
        ),
    ]
    store = TfidfLibrary(dim=512)
    store.add(documents)
    return store


# --- the shipped evaluation sets ---------------------------------------------------


@pytest.fixture(scope="session")
def summary_examples() -> list[SummaryExample]:
    return load_jsonl(EVAL_DIR / "summaries.jsonl", SummaryExample)


@pytest.fixture(scope="session")
def sentiment_examples() -> list[SentimentExample]:
    return load_jsonl(EVAL_DIR / "sentiment.jsonl", SentimentExample)


@pytest.fixture(scope="session")
def entity_examples() -> list[EntityExample]:
    return load_jsonl(EVAL_DIR / "entities.jsonl", EntityExample)


@pytest.fixture(scope="session")
def qa_examples() -> list[QAExample]:
    return load_jsonl(EVAL_DIR / "qa.jsonl", QAExample)


@pytest.fixture(scope="session")
def repair_cases() -> list[RepairCase]:
    return load_jsonl(EVAL_DIR / "repair_cases.jsonl", RepairCase)
