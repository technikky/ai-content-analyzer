"""The analysis pipeline: run the requested analyzers, time each, cache the result.

Two things this module is careful about.

**One analyzer failing does not lose the others.** A document that breaks entity
extraction should still come back with its summary and sentiment. Each stage is isolated
and its failure is recorded on the result rather than raised.

**Timing is per stage.** A single total tells you the request was slow; per-stage timing
tells you which analyzer to fix, and they have completely different fixes — a slow LLM
summariser is a token-budget problem, slow rule-based entity extraction is a regex problem.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypeVar

from analyzer.analyzers.base import (
    AnalyzerError,
    EntityExtractor,
    QuestionAnswerer,
    SentimentAnalyzer,
    Summarizer,
)
from analyzer.analyzers.entities import RuleEntityExtractor
from analyzer.analyzers.qa import ExtractiveQA
from analyzer.analyzers.sentiment import LexiconSentiment
from analyzer.analyzers.summarize import LeadSummarizer
from analyzer.text import content_hash
from analyzer.types import Analysis, AnalysisKind, Document

ALL_ANALYSES: tuple[AnalysisKind, ...] = ("summary", "sentiment", "entities", "qa")

T = TypeVar("T")


class AnalysisCache:
    """Content-addressed cache of completed analyses.

    Keyed on the document text, the analyses requested, the method names and the question,
    so changing any of them correctly misses. Re-analysing an unchanged document with the
    same configuration is pure waste, and with an LLM in the pipeline it is paid waste.
    """

    def __init__(self, directory: str | Path | None = None, *, enabled: bool = True) -> None:
        self.directory = Path(directory) if directory else Path(".analysis-cache")
        self.enabled = enabled
        self.hits = 0
        self.misses = 0

    def key(
        self,
        document: Document,
        analyses: Sequence[str],
        methods: dict[str, str],
        question: str | None,
    ) -> str:
        return content_hash(
            document.text,
            ",".join(sorted(analyses)),
            json.dumps(methods, sort_keys=True),
            question or "",
        )

    def _path_for(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> Analysis | None:
        if not self.enabled:
            return None
        path = self._path_for(key)
        if not path.is_file():
            self.misses += 1
            return None
        try:
            analysis = Analysis.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # A corrupt entry must never fail a request; treat it as a miss.
            self.misses += 1
            return None
        self.hits += 1
        return analysis.model_copy(update={"cached": True})

    def set(self, key: str, analysis: Analysis) -> None:
        if not self.enabled:
            return
        path = self._path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(analysis.model_dump(mode="json"), sort_keys=True), encoding="utf-8"
        )


class ContentAnalyzer:
    """Runs summarisation, sentiment, entity extraction and QA over a document."""

    def __init__(
        self,
        *,
        summarizer: Summarizer | None = None,
        sentiment: SentimentAnalyzer | None = None,
        entities: EntityExtractor | None = None,
        question_answerer: QuestionAnswerer | None = None,
        cache: AnalysisCache | None = None,
    ) -> None:
        self.summarizer = summarizer or LeadSummarizer()
        self.sentiment = sentiment or LexiconSentiment()
        self.entities = entities or RuleEntityExtractor()
        self.question_answerer = question_answerer or ExtractiveQA()
        self.cache = cache if cache is not None else AnalysisCache(enabled=False)
        #: Populated per call with "<stage>: <reason>" for any stage that failed.
        self.last_errors: dict[str, str] = {}

    @property
    def methods(self) -> dict[str, str]:
        return {
            "summary": self.summarizer.name,
            "sentiment": self.sentiment.name,
            "entities": self.entities.name,
            "qa": self.question_answerer.name,
        }

    def analyze(
        self,
        document: Document,
        *,
        analyses: Sequence[AnalysisKind] | None = None,
        question: str | None = None,
    ) -> Analysis:
        requested = tuple(analyses) if analyses is not None else ALL_ANALYSES
        unknown = sorted(set(requested) - set(ALL_ANALYSES))
        if unknown:
            raise ValueError(
                f"unknown analyses: {', '.join(unknown)}. Expected any of {', '.join(ALL_ANALYSES)}"
            )
        if "qa" in requested and not (question or "").strip():
            raise ValueError("the 'qa' analysis needs a question")

        key = self.cache.key(document, requested, self.methods, question)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        self.last_errors = {}
        result = Analysis(
            document_id=document.id,
            title=document.title,
            n_chars=document.n_chars,
            methods={kind: self.methods[kind] for kind in requested},
        )

        if "summary" in requested:
            result.summary = self._timed(
                result, "summary", lambda: self.summarizer.summarize(document)
            )
        if "sentiment" in requested:
            result.sentiment = self._timed(
                result, "sentiment", lambda: self.sentiment.analyze(document.text)
            )
        if "entities" in requested:
            found = self._timed(result, "entities", lambda: self.entities.extract(document.text))
            result.entities = found or []
        if "qa" in requested:
            question_text = str(question)
            result.qa = self._timed(
                result,
                "qa",
                lambda: self.question_answerer.answer(question_text, document),
            )

        self.cache.set(key, result)
        return result

    def _timed(self, result: Analysis, stage: str, run: Callable[[], T]) -> T | None:
        """Run one stage, recording its duration and isolating its failure.

        Returns ``None`` when the stage failed, so the caller's field stays unset while
        the reason is recorded on ``last_errors``.
        """
        started = time.perf_counter()
        try:
            return run()
        except (AnalyzerError, ValueError, RuntimeError) as exc:
            # Recorded on the analyzer, not raised: the other stages still have value.
            self.last_errors[stage] = f"{type(exc).__name__}: {exc}"
            return None
        finally:
            result.timings_ms[stage] = (time.perf_counter() - started) * 1000.0
