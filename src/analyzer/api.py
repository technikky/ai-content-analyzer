"""FastAPI service.

The app is built around injected components rather than constructing them at import time,
so the test suite exercises every endpoint against an in-memory library with no fixture
files and no environment variables.

Stage failures are surfaced, not swallowed. A response carries an ``errors`` map naming any
analyzer that failed, with a 200 status, because the other analyses succeeded and are worth
returning. A caller that needs all-or-nothing can check the map.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

from analyzer.library import DocumentLibrary, TfidfLibrary
from analyzer.pipeline import ALL_ANALYSES, ContentAnalyzer
from analyzer.types import (
    AnalysisKind,
    Document,
    Entity,
    QAAnswer,
    SearchMatch,
    SentimentResult,
    Summary,
)

LIBRARY_DIR_ENV = "ANALYZER_LIBRARY_DIR"
MAX_TEXT_CHARS = 200_000


class AnalyzeRequest(BaseModel):
    """Text to analyse, and which analyses to run."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
    title: str = Field(default="", max_length=500)
    document_id: str | None = Field(default=None, max_length=200)
    analyses: list[AnalysisKind] | None = None
    question: str | None = Field(default=None, max_length=2000)
    #: Add the document to the searchable library as well as analysing it.
    store: bool = False


class AnalyzeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    title: str
    n_chars: int
    summary: Summary | None = None
    sentiment: SentimentResult | None = None
    entities: list[Entity] = Field(default_factory=list)
    qa: QAAnswer | None = None
    timings_ms: dict[str, float] = Field(default_factory=dict)
    methods: dict[str, str] = Field(default_factory=dict)
    total_ms: float = 0.0
    cached: bool = False
    #: Stage name to reason, for any analyzer that failed. Empty when all succeeded.
    errors: dict[str, str] = Field(default_factory=dict)


class DocumentsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    documents: list[Document] = Field(min_length=1, max_length=500)


class DocumentSummaryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    n_chars: int


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=50)


class QuestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2000)


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    documents: int


def get_analyzer(request: Request) -> ContentAnalyzer:
    analyzer = getattr(request.app.state, "analyzer", None)
    if not isinstance(analyzer, ContentAnalyzer):  # pragma: no cover - guarded by startup
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="analyzer not initialised"
        )
    return analyzer


def get_library(request: Request) -> DocumentLibrary:
    library: DocumentLibrary | None = getattr(request.app.state, "library", None)
    if library is None:  # pragma: no cover - guarded by startup
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="library not initialised"
        )
    return library


def create_app(
    analyzer: ContentAnalyzer | None = None,
    library: DocumentLibrary | None = None,
    *,
    library_dir: str | Path | None = None,
) -> FastAPI:
    """Build the app. A library directory, if given or set in the environment, is loaded once."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.analyzer = analyzer or ContentAnalyzer()
        if library is not None:
            app.state.library = library
        else:
            target = library_dir or os.environ.get(LIBRARY_DIR_ENV)
            app.state.library = TfidfLibrary.load(target) if target else TfidfLibrary()
        yield
        app.state.analyzer = None
        app.state.library = None

    app = FastAPI(
        title="ai-content-analyzer",
        summary="Summarisation, sentiment, entity extraction and document Q&A.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/healthz", response_model=HealthResponse, tags=["ops"])
    def healthz(lib: DocumentLibrary = Depends(get_library)) -> HealthResponse:
        return HealthResponse(status="ok", documents=len(lib))

    @app.get("/stats", tags=["ops"])
    def stats(
        engine: ContentAnalyzer = Depends(get_analyzer),
        lib: DocumentLibrary = Depends(get_library),
    ) -> dict[str, Any]:
        return {
            "documents": len(lib),
            "library": lib.name,
            "methods": engine.methods,
            "analyses": list(ALL_ANALYSES),
            "cache_enabled": engine.cache.enabled,
            "cache_hits": engine.cache.hits,
            "cache_misses": engine.cache.misses,
        }

    @app.post("/analyze", response_model=AnalyzeResponse, tags=["analysis"])
    def analyze(
        payload: AnalyzeRequest,
        engine: ContentAnalyzer = Depends(get_analyzer),
        lib: DocumentLibrary = Depends(get_library),
    ) -> AnalyzeResponse:
        document = Document(
            id=payload.document_id or f"doc-{len(lib) + 1}",
            text=payload.text,
            title=payload.title,
        )
        try:
            result = engine.analyze(document, analyses=payload.analyses, question=payload.question)
        except ValueError as exc:
            # A request the pipeline cannot serve, e.g. "qa" with no question.
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
            ) from exc

        if payload.store and lib.get(document.id) is None:
            lib.add([document])

        return AnalyzeResponse(
            document_id=result.document_id,
            title=result.title,
            n_chars=result.n_chars,
            summary=result.summary,
            sentiment=result.sentiment,
            entities=result.entities,
            qa=result.qa,
            timings_ms={k: round(v, 3) for k, v in result.timings_ms.items()},
            methods=result.methods,
            total_ms=round(result.total_ms, 3),
            cached=result.cached,
            errors=dict(engine.last_errors),
        )

    @app.post("/documents", tags=["library"], status_code=status.HTTP_201_CREATED)
    def add_documents(
        payload: DocumentsRequest, lib: DocumentLibrary = Depends(get_library)
    ) -> dict[str, int]:
        try:
            total = lib.add(payload.documents)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        return {"added": len(payload.documents), "documents": total}

    @app.get("/documents", tags=["library"])
    def list_documents(
        lib: DocumentLibrary = Depends(get_library),
    ) -> list[DocumentSummaryResponse]:
        return [
            DocumentSummaryResponse(id=d.id, title=d.title, n_chars=d.n_chars)
            for d in lib.all_documents()
        ]

    # `:path` because document ids contain slashes, as in "kb/refunds". A plain path
    # parameter matches a single segment and would 404 on every namespaced id.
    @app.get("/documents/{document_id:path}", response_model=Document, tags=["library"])
    def get_document(document_id: str, lib: DocumentLibrary = Depends(get_library)) -> Document:
        document = lib.get(document_id)
        if document is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"no document {document_id!r}"
            )
        return document

    @app.post("/search", tags=["library"])
    def search(
        payload: SearchRequest, lib: DocumentLibrary = Depends(get_library)
    ) -> list[SearchMatch]:
        return lib.search(payload.query, payload.top_k)

    @app.post("/documents/{document_id:path}/qa", response_model=QAAnswer, tags=["analysis"])
    def ask(
        document_id: str,
        payload: QuestionRequest,
        engine: ContentAnalyzer = Depends(get_analyzer),
        lib: DocumentLibrary = Depends(get_library),
    ) -> QAAnswer:
        document = lib.get(document_id)
        if document is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"no document {document_id!r}"
            )
        answer = engine.question_answerer.answer(payload.question, document)
        if not isinstance(answer, QAAnswer):  # pragma: no cover - defensive
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="analyzer failed"
            )
        return answer

    return app
