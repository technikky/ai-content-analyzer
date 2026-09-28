"""The document library and semantic search over it.

Scope note, stated plainly: retrieval here is deliberately minimal. The default library is
TF-IDF cosine over whole documents — enough to find the right document in a library of
hundreds, which is what a content analyzer needs. Chunking, hybrid lexical+dense
retrieval, reciprocal-rank fusion, reranking and the measurement of all of it live in a
separate project, `production-rag-system`, and duplicating that here would be padding.

A Chroma-backed library is available behind the ``chroma`` extra, which is what makes the
search semantic rather than lexical. It is not covered by the test suite and no retrieval
metrics are published for it, because a suite that silently skips is not a gate.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from analyzer.text import content_tokens, snippet, stable_bucket
from analyzer.types import Document, SearchMatch

DOCUMENTS_FILE = "documents.jsonl"
LIBRARY_META_FILE = "library.json"
DEFAULT_DIM = 4096


class LibraryError(RuntimeError):
    """Raised for library configuration, persistence and dependency failures."""


@runtime_checkable
class DocumentLibrary(Protocol):
    """Stores documents and finds the ones matching a query."""

    @property
    def name(self) -> str: ...

    def add(self, documents: list[Document]) -> int: ...

    def search(self, query: str, top_k: int = 5) -> list[SearchMatch]: ...

    def get(self, document_id: str) -> Document | None: ...

    def all_documents(self) -> list[Document]: ...

    def __len__(self) -> int: ...


class TfidfLibrary:
    """TF-IDF cosine search over whole documents.

    Vectors are sparse dictionaries rather than dense arrays. With a few hundred documents
    and a few thousand terms, a dense matrix would be almost entirely zeros, and the sparse
    form keeps this module free of a numpy dependency for what is a small amount of
    arithmetic.
    """

    def __init__(self, dim: int = DEFAULT_DIM) -> None:
        if dim < 64:
            raise ValueError(f"dim must be >= 64, got {dim}")
        self._dim = dim
        self._documents: list[Document] = []
        self._index: dict[str, int] = {}
        self._vectors: list[dict[int, float]] = []
        self._idf: dict[int, float] = {}

    @property
    def name(self) -> str:
        return f"tfidf-{self._dim}"

    def add(self, documents: list[Document]) -> int:
        """Add documents and rebuild the index. Returns the total document count.

        A rebuild because IDF is fitted over the library: adding a document changes the
        correct weight of every term in every existing document. Appending without
        refitting would leave old vectors weighted against different statistics from new
        ones, which degrades ranking quietly.
        """
        if not documents:
            raise ValueError("cannot add zero documents")

        clashes = sorted({d.id for d in documents if d.id in self._index})
        if clashes:
            raise LibraryError(f"documents already in the library: {', '.join(clashes)}")

        for document in documents:
            self._index[document.id] = len(self._documents)
            self._documents.append(document)

        self._rebuild()
        return len(self._documents)

    def _rebuild(self) -> None:
        counts = [Counter(content_tokens(d.text)) for d in self._documents]
        document_frequency: Counter[int] = Counter()
        for count in counts:
            document_frequency.update({stable_bucket(t, self._dim) for t in count})

        n = len(self._documents)
        self._idf = {
            bucket: math.log((1.0 + n) / (1.0 + df)) + 1.0
            for bucket, df in document_frequency.items()
        }

        self._vectors = []
        for count in counts:
            vector: dict[int, float] = {}
            for term, frequency in count.items():
                bucket = stable_bucket(term, self._dim)
                # Sublinear term frequency: the tenth occurrence says much less than the
                # second, and raw counts let long documents dominate.
                vector[bucket] = vector.get(bucket, 0.0) + (
                    1.0 + math.log(frequency)
                ) * self._idf.get(bucket, 1.0)
            self._vectors.append(_normalize(vector))

    def _encode_query(self, query: str) -> dict[int, float]:
        count = Counter(content_tokens(query))
        vector: dict[int, float] = {}
        for term, frequency in count.items():
            bucket = stable_bucket(term, self._dim)
            vector[bucket] = vector.get(bucket, 0.0) + (1.0 + math.log(frequency)) * self._idf.get(
                bucket, 1.0
            )
        return _normalize(vector)

    def search(self, query: str, top_k: int = 5) -> list[SearchMatch]:
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if not query.strip() or not self._documents:
            return []

        query_vector = self._encode_query(query)
        if not query_vector:
            return []

        scored: list[tuple[float, int]] = []
        for position, vector in enumerate(self._vectors):
            # Iterate the shorter side: a query has a handful of terms, a document has
            # hundreds.
            score = sum(weight * vector.get(bucket, 0.0) for bucket, weight in query_vector.items())
            if score > 0.0:
                scored.append((score, position))

        scored.sort(key=lambda pair: (-pair[0], self._documents[pair[1]].id))
        matches: list[SearchMatch] = []
        for rank, (score, position) in enumerate(scored[:top_k], start=1):
            document = self._documents[position]
            matches.append(
                SearchMatch(
                    document_id=document.id,
                    title=document.title,
                    score=score,
                    snippet=snippet(document.text),
                    rank=rank,
                )
            )
        return matches

    def get(self, document_id: str) -> Document | None:
        position = self._index.get(document_id)
        return self._documents[position] if position is not None else None

    def all_documents(self) -> list[Document]:
        return list(self._documents)

    def __len__(self) -> int:
        return len(self._documents)

    # --- persistence ---------------------------------------------------------------

    def save(self, directory: str | Path) -> Path:
        """Persist the documents. Vectors are not saved: they are cheap to recompute and
        serialising them would add a format to keep in step with the document schema."""
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        with (target / DOCUMENTS_FILE).open("w", encoding="utf-8", newline="\n") as handle:
            for document in self._documents:
                handle.write(json.dumps(document.model_dump(mode="json"), sort_keys=True) + "\n")
        (target / LIBRARY_META_FILE).write_text(
            json.dumps(
                {"name": self.name, "dim": self._dim, "n_documents": len(self._documents)}, indent=2
            )
            + "\n",
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, directory: str | Path) -> TfidfLibrary:
        source = Path(directory)
        meta_path = source / LIBRARY_META_FILE
        if not meta_path.is_file():
            raise LibraryError(f"no library at {source} (missing {LIBRARY_META_FILE})")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        library = cls(int(meta.get("dim", DEFAULT_DIM)))
        documents = [
            Document.model_validate_json(line)
            for line in (source / DOCUMENTS_FILE).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if documents:
            library.add(documents)
        return library


def _normalize(vector: dict[int, float]) -> dict[int, float]:
    norm = math.sqrt(sum(weight * weight for weight in vector.values()))
    if norm == 0.0:
        return {}
    return {bucket: weight / norm for bucket, weight in vector.items()}


class ChromaLibrary:
    """Chroma-backed library, for genuinely semantic search.

    Chroma's default embedding function is a sentence-transformers model, so this is the
    path that satisfies "semantic search" rather than lexical matching. It needs the
    ``chroma`` extra and a model download, is not exercised by the test suite, and has no
    published retrieval metrics.
    """

    def __init__(self, *, collection: str = "documents", path: str | None = None) -> None:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise LibraryError(
                "chromadb is required. Install with: pip install 'ai-content-analyzer[chroma]'"
            ) from exc

        self._client = chromadb.PersistentClient(path=path) if path else chromadb.EphemeralClient()
        self._collection = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )
        self._documents: dict[str, Document] = {}

    @property
    def name(self) -> str:
        return "chroma"

    def add(self, documents: list[Document]) -> int:  # pragma: no cover - needs the extra
        if not documents:
            raise ValueError("cannot add zero documents")
        self._collection.add(
            ids=[d.id for d in documents],
            documents=[d.text for d in documents],
            metadatas=[{"title": d.title} for d in documents],
        )
        for document in documents:
            self._documents[document.id] = document
        return len(self._documents)

    def search(self, query: str, top_k: int = 5) -> list[SearchMatch]:  # pragma: no cover
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if not query.strip() or not self._documents:
            return []

        result = self._collection.query(query_texts=[query], n_results=top_k)
        matches: list[SearchMatch] = []
        pairs = zip(result["ids"][0], result["distances"][0], strict=False)
        for rank, (document_id, distance) in enumerate(pairs, start=1):
            document = self._documents.get(document_id)
            if document is None:
                continue
            matches.append(
                SearchMatch(
                    document_id=document_id,
                    title=document.title,
                    # Chroma reports cosine distance; the rest of the code uses similarity.
                    score=1.0 - float(distance),
                    snippet=snippet(document.text),
                    rank=rank,
                )
            )
        return matches

    def get(self, document_id: str) -> Document | None:  # pragma: no cover
        return self._documents.get(document_id)

    def all_documents(self) -> list[Document]:  # pragma: no cover
        return list(self._documents.values())

    def __len__(self) -> int:  # pragma: no cover
        return len(self._documents)


LIBRARIES = ("tfidf", "chroma")


def build_library(name: str = "tfidf", **kwargs: Any) -> DocumentLibrary:
    key = name.strip().lower()
    if key == "tfidf":
        return TfidfLibrary(**kwargs)
    if key == "chroma":
        return ChromaLibrary(**kwargs)
    raise LibraryError(f"unknown library {name!r}; expected one of {', '.join(LIBRARIES)}")
