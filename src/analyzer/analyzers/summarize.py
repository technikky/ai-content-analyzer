"""Summarisation: two extractive baselines and an LLM summariser.

``LeadSummarizer`` is the standard weak baseline for news-like prose, and it is
surprisingly hard to beat, because the first sentences of a well-written document usually
are the summary. ``CentroidSummarizer`` scores sentences against the document's own
TF-IDF centroid and is the honest "we tried something" comparison. Reporting both is what
makes an LLM's ROUGE number interpretable.
"""

from __future__ import annotations

import math
from collections import Counter

from analyzer.analyzers.base import AnalyzerError, Summarizer
from analyzer.providers import LLM
from analyzer.schemas import SummaryResponse, parse_model_output
from analyzer.text import content_tokens, normalize, sentences
from analyzer.types import Document, Summary

SUMMARY_PROMPT = """Summarise the document below in at most {max_sentences} sentences.

Rules:
- Use only information present in the document. Do not add anything.
- Keep every figure, name and date exactly as written.
- Reply with JSON only, in this shape:
  {{"summary": "<the summary>", "key_points": ["<point>", "..."]}}

DOCUMENT
{document}

JSON"""


def _compression(summary: str, document: str) -> float | None:
    """Summary length as a fraction of the source, in characters."""
    if not document:
        return None
    return len(summary) / len(document)


class LeadSummarizer:
    """The first ``max_sentences`` sentences. The baseline everything else must beat."""

    def __init__(self, max_sentences: int = 3) -> None:
        if max_sentences < 1:
            raise ValueError(f"max_sentences must be >= 1, got {max_sentences}")
        self.max_sentences = max_sentences

    @property
    def name(self) -> str:
        return f"lead-{self.max_sentences}"

    def summarize(self, document: Document) -> Summary:
        selected = sentences(document.text)[: self.max_sentences]
        text = " ".join(selected)
        return Summary(
            text=text,
            sentences=selected,
            method=self.name,
            compression_ratio=_compression(text, document.text),
        )


class CentroidSummarizer:
    """Selects the sentences closest to the document's TF-IDF centroid.

    Sentences are then emitted in **document order**, not score order. A summary whose
    sentences are shuffled by relevance reads badly and loses the discourse that made the
    original coherent, which costs more than the ranking gains.
    """

    def __init__(self, max_sentences: int = 3) -> None:
        if max_sentences < 1:
            raise ValueError(f"max_sentences must be >= 1, got {max_sentences}")
        self.max_sentences = max_sentences

    @property
    def name(self) -> str:
        return f"centroid-{self.max_sentences}"

    def summarize(self, document: Document) -> Summary:
        all_sentences = sentences(document.text)
        if not all_sentences:
            return Summary(text="", sentences=[], method=self.name, compression_ratio=None)

        tokenised = [content_tokens(sentence) for sentence in all_sentences]
        document_frequency: Counter[str] = Counter()
        for tokens in tokenised:
            document_frequency.update(set(tokens))

        n = len(all_sentences)
        idf = {
            term: math.log((1.0 + n) / (1.0 + df)) + 1.0 for term, df in document_frequency.items()
        }

        scores: list[tuple[int, float]] = []
        for index, tokens in enumerate(tokenised):
            if not tokens:
                scores.append((index, 0.0))
                continue
            # Averaged rather than summed, so a long sentence does not win on length alone.
            weight = sum(idf.get(term, 0.0) for term in set(tokens)) / math.sqrt(len(tokens))
            scores.append((index, weight))

        best = sorted(scores, key=lambda pair: (-pair[1], pair[0]))[: self.max_sentences]
        chosen = sorted(index for index, _ in best)
        selected = [all_sentences[index] for index in chosen]
        text = " ".join(selected)
        return Summary(
            text=text,
            sentences=selected,
            method=self.name,
            compression_ratio=_compression(text, document.text),
        )


class LLMSummarizer:
    """Asks a model for a summary as JSON, and validates what comes back."""

    def __init__(self, llm: LLM, *, max_sentences: int = 3, max_tokens: int = 600) -> None:
        if max_sentences < 1:
            raise ValueError(f"max_sentences must be >= 1, got {max_sentences}")
        self.llm = llm
        self.max_sentences = max_sentences
        self.max_tokens = max_tokens
        self.last_repairs: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return f"llm-summary:{self.llm.name}"

    def summarize(self, document: Document) -> Summary:
        prompt = SUMMARY_PROMPT.format(max_sentences=self.max_sentences, document=document.text)
        raw = self.llm.complete(prompt, max_tokens=self.max_tokens)

        outcome = parse_model_output(raw, SummaryResponse)
        self.last_repairs = outcome.repairs
        if not outcome.ok or outcome.value is None:
            raise AnalyzerError(f"summariser response unusable: {outcome.error}")

        response = outcome.value
        text = normalize(response.summary)
        return Summary(
            text=text,
            sentences=sentences(text) or [text],
            method=self.name,
            compression_ratio=_compression(text, document.text),
        )


def build_summarizer(
    method: str = "lead", *, llm: LLM | None = None, max_sentences: int = 3
) -> Summarizer:
    key = method.strip().lower()
    if key == "lead":
        return LeadSummarizer(max_sentences)
    if key == "centroid":
        return CentroidSummarizer(max_sentences)
    if key == "llm":
        if llm is None:
            raise ValueError("the 'llm' summariser needs a provider; pass llm=...")
        return LLMSummarizer(llm, max_sentences=max_sentences)
    raise ValueError(f"unknown summariser {method!r}; expected lead, centroid or llm")
