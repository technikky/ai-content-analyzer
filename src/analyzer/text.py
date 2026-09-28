"""Text primitives: normalisation, tokenisation, sentence spans, stable hashing.

Spans rather than strings wherever a position matters. An entity extractor that returned
only the matched text would leave the caller to find it again, and `text.find(name)` picks
the first occurrence, which is the wrong one often enough to matter.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

_TOKEN = re.compile(r"[^\W_]+(?:'[^\W_]+)*", re.UNICODE)
_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+")
_WHITESPACE = re.compile(r"\s+")

#: Words with no discriminative content, removed before overlap scoring. Deliberately
#: short: an aggressive stop list discards terms that carry real signal in technical prose.
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "our",
        "she",
        "should",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
    ]
)


def normalize(text: str) -> str:
    """NFKC-normalise and collapse whitespace."""
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", text)).strip()


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens."""
    return _TOKEN.findall(unicodedata.normalize("NFKC", text).lower())


def content_tokens(text: str) -> list[str]:
    """Tokens with stopwords removed, for overlap and coverage scoring."""
    return [token for token in tokenize(text) if token not in STOPWORDS]


def token_spans(text: str) -> list[tuple[str, int, int]]:
    """Each token with its character span, for offset-accurate matching."""
    return [
        (match.group(0).lower(), match.start(), match.end())
        for match in _TOKEN.finditer(unicodedata.normalize("NFKC", text))
    ]


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of each sentence, whitespace trimmed.

    A deliberately simple splitter. It mishandles abbreviations such as "e.g." and will
    split them; that is a known limitation, documented rather than papered over, and the
    alternative is a dependency this package does not need.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    for match in _SENTENCE_END.finditer(text):
        if match.start() > cursor:
            spans.append((cursor, match.start()))
        cursor = match.end()
    if cursor < len(text):
        spans.append((cursor, len(text)))

    trimmed: list[tuple[int, int]] = []
    for start, end in spans:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            trimmed.append((start, end))
    return trimmed


def sentences(text: str) -> list[str]:
    """Sentence strings, whitespace-normalised for display."""
    return [normalize(text[start:end]) for start, end in sentence_spans(text)]


def stable_bucket(token: str, buckets: int) -> int:
    """Map a token to a bucket reproducibly across processes.

    ``hash()`` is randomised per interpreter unless ``PYTHONHASHSEED`` is set, so a
    hashing vectoriser built on it would place the same token in different buckets after a
    restart and silently invalidate a persisted index.
    """
    if buckets < 1:
        raise ValueError(f"buckets must be >= 1, got {buckets}")
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % buckets


def content_hash(*parts: str) -> str:
    """Digest over the given parts, used to key the analysis cache."""
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def snippet(text: str, *, limit: int = 200) -> str:
    """A short, whitespace-normalised excerpt, cut at a word boundary."""
    flat = normalize(text)
    if len(flat) <= limit:
        return flat
    cut = flat[:limit].rsplit(" ", 1)[0]
    return f"{cut}..."


def token_f1(predicted: str, reference: str) -> float:
    """Unigram F1 with clipped counts, so repeating a token cannot inflate the score."""
    predicted_tokens = tokenize(predicted)
    reference_tokens = tokenize(reference)
    if not predicted_tokens or not reference_tokens:
        return 0.0

    remaining: dict[str, int] = {}
    for token in predicted_tokens:
        remaining[token] = remaining.get(token, 0) + 1

    overlap = 0
    for token in reference_tokens:
        if remaining.get(token, 0) > 0:
            remaining[token] -= 1
            overlap += 1
    if overlap == 0:
        return 0.0

    precision = overlap / len(predicted_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)
