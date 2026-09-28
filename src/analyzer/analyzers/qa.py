"""Document question answering: an extractive baseline and an LLM answerer.

Both can decline. That is the point of the ``supported`` flag: a document that does not
answer a question should produce "not supported", not a confident sentence that happens to
share vocabulary with the question. An analyzer that always answers cannot be distinguished
from one that answers well, and on unanswerable questions it is always wrong.
"""

from __future__ import annotations

import math
from collections import Counter

from analyzer.analyzers.base import AnalyzerError, QuestionAnswerer
from analyzer.providers import LLM
from analyzer.schemas import QAResponse, parse_model_output
from analyzer.text import content_tokens, sentences
from analyzer.types import Document, QAAnswer

QA_PROMPT = """Answer the question using only the document below.

Rules:
- If the document does not contain the answer, set supported to false and answer with:
  The document does not answer this question.
- Quote the sentences you relied on in evidence, copied exactly from the document.
- Keep the answer to one or two sentences.
- Reply with JSON only:
  {{"answer": "<answer>", "supported": <true|false>, "evidence": ["<sentence>"]}}

DOCUMENT
{document}

QUESTION
{question}

JSON"""

NO_ANSWER = "The document does not answer this question."

#: Share of the question's information-bearing terms a sentence must cover to count as an
#: answer. Chosen by reasoning rather than tuned on the evaluation set: below roughly a
#: third, the overlap is usually incidental vocabulary rather than an answer.
SUPPORT_THRESHOLD = 0.34


class ExtractiveQA:
    """Returns the sentence that best covers the question's terms, or declines.

    Coverage is weighted by inverse sentence frequency within the document, so a term that
    appears in every sentence contributes little and a distinctive term contributes a lot.
    Matching on raw overlap makes the longest sentence win almost every time.
    """

    def __init__(self, *, max_sentences: int = 2, threshold: float = SUPPORT_THRESHOLD) -> None:
        if max_sentences < 1:
            raise ValueError(f"max_sentences must be >= 1, got {max_sentences}")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        self.max_sentences = max_sentences
        self.threshold = threshold

    @property
    def name(self) -> str:
        return "extractive"

    def answer(self, question: str, document: Document) -> QAAnswer:
        question_terms = content_tokens(question)
        document_sentences = sentences(document.text)

        if not question_terms or not document_sentences:
            return QAAnswer(
                question=question,
                answer=NO_ANSWER,
                supported=False,
                confidence=0.0,
                method=self.name,
            )

        tokenised = [content_tokens(sentence) for sentence in document_sentences]
        sentence_frequency: Counter[str] = Counter()
        for tokens in tokenised:
            sentence_frequency.update(set(tokens))

        n = len(document_sentences)
        weights = {
            term: math.log((1.0 + n) / (1.0 + sentence_frequency.get(term, 0))) + 1.0
            for term in set(question_terms)
        }
        total_weight = sum(weights[term] for term in set(question_terms)) or 1.0

        scored: list[tuple[float, int]] = []
        for index, tokens in enumerate(tokenised):
            present = set(tokens)
            covered = sum(weights[term] for term in set(question_terms) if term in present)
            scored.append((covered / total_weight, index))

        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        best_score, best_index = scored[0]

        if best_score < self.threshold:
            return QAAnswer(
                question=question,
                answer=NO_ANSWER,
                supported=False,
                confidence=best_score,
                # The near-miss is still reported, so a caller can show it as a weak match
                # instead of showing nothing.
                evidence=[document_sentences[best_index]],
                method=self.name,
            )

        # Every sentence clearing the threshold, up to the budget, in document order so
        # the answer reads as prose rather than as a relevance ranking.
        chosen = [index for score, index in scored[: self.max_sentences] if score >= self.threshold]
        if best_index not in chosen:
            chosen = [best_index, *chosen][: self.max_sentences]
        evidence = [document_sentences[index] for index in sorted(set(chosen))]

        return QAAnswer(
            question=question,
            answer=" ".join(evidence),
            supported=True,
            confidence=min(1.0, best_score),
            evidence=evidence,
            method=self.name,
        )


class LLMQA:
    """Asks a model, and verifies its evidence against the document.

    Evidence that does not occur in the document is dropped, and an answer whose evidence
    all disappears is downgraded to unsupported. A model that cites text the document does
    not contain has not answered the question, whatever it claims.
    """

    def __init__(self, llm: LLM, *, max_tokens: int = 500) -> None:
        self.llm = llm
        self.max_tokens = max_tokens
        self.last_repairs: tuple[str, ...] = ()
        self.last_dropped_evidence: list[str] = []

    @property
    def name(self) -> str:
        return f"llm-qa:{self.llm.name}"

    def answer(self, question: str, document: Document) -> QAAnswer:
        prompt = QA_PROMPT.format(document=document.text, question=question)
        raw = self.llm.complete(prompt, max_tokens=self.max_tokens)
        outcome = parse_model_output(raw, QAResponse)
        self.last_repairs = outcome.repairs
        if not outcome.ok or outcome.value is None:
            raise AnalyzerError(f"QA response unusable: {outcome.error}")

        response = outcome.value
        haystack = " ".join(document.text.split()).casefold()
        kept: list[str] = []
        dropped: list[str] = []
        for quote in response.evidence:
            flattened = " ".join(quote.split()).casefold()
            if flattened and flattened in haystack:
                kept.append(quote.strip())
            else:
                dropped.append(quote.strip())
        self.last_dropped_evidence = dropped

        supported = response.supported and (bool(kept) or not response.evidence)
        return QAAnswer(
            question=question,
            answer=response.answer if supported else NO_ANSWER,
            supported=supported,
            confidence=1.0 if supported else 0.0,
            evidence=kept,
            method=self.name,
        )


def build_question_answerer(
    method: str = "extractive", *, llm: LLM | None = None
) -> QuestionAnswerer:
    key = method.strip().lower()
    if key == "extractive":
        return ExtractiveQA()
    if key == "llm":
        if llm is None:
            raise ValueError("the 'llm' question answerer needs a provider; pass llm=...")
        return LLMQA(llm)
    raise ValueError(f"unknown question answerer {method!r}; expected extractive or llm")
