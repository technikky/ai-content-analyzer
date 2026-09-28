"""Evaluation for every analyzer, plus the JSON-repair harness.

The point of this module is that "the analyzer works" is not a claim anyone should accept
without a number, and that the number needs a baseline beside it to mean anything.

Each evaluation set is split into **dev** and **test**. The lexicon, the gazetteers and the
thresholds were developed while looking at the dev split; the test split was not consulted.
Both are reported separately, because an in-sample accuracy is optimistic by an unknown
amount and quoting only the combined figure would overstate the baselines. That split is
the difference between a measurement and a demonstration.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from analyzer.analyzers.base import (
    AnalyzerError,
    EntityExtractor,
    QuestionAnswerer,
    SentimentAnalyzer,
    Summarizer,
)
from analyzer.schemas import (
    EntitiesResponse,
    QAResponse,
    SentimentResponse,
    SummaryResponse,
    parse_model_output,
)
from analyzer.text import normalize, token_f1, tokenize
from analyzer.types import (
    Document,
    Entity,
    EntityMetrics,
    EntityType,
    QAMetrics,
    RepairMetrics,
    SentimentLabel,
    SentimentMetrics,
    SummaryMetrics,
)

Split = Literal["dev", "test"]
#: Any example schema in this module.
E = TypeVar("E", bound=BaseModel)
SCHEMAS: dict[str, type[BaseModel]] = {
    "summary": SummaryResponse,
    "sentiment": SentimentResponse,
    "entities": EntitiesResponse,
    "qa": QAResponse,
}


# --- example schemas --------------------------------------------------------------


class SummaryExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    reference: str = Field(min_length=1)
    split: Split = "test"


class SentimentExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    label: SentimentLabel
    split: Split = "test"
    note: str = ""


class GoldEntity(BaseModel):
    """A labelled entity, given as text rather than offsets.

    Offsets are derived by locating the text in the document, first unclaimed occurrence
    winning. Hand-written offsets in a JSONL file are unmaintainable — one edit to the text
    invalidates every span after it — and deriving them keeps the labels readable.
    """

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    type: EntityType


class EntityExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    entities: list[GoldEntity] = Field(default_factory=list)
    split: Split = "test"


class QAExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    document: str = Field(min_length=1)
    question: str = Field(min_length=1)
    #: ``None`` when the document does not answer the question.
    answer: str | None = None
    split: Split = "test"

    @property
    def answerable(self) -> bool:
        return self.answer is not None


class RepairCase(BaseModel):
    """A malformed model response, and whether it should be recoverable."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    schema_name: Literal["summary", "sentiment", "entities", "qa"]
    raw: str
    should_parse: bool
    note: str = ""


# --- loading ----------------------------------------------------------------------


def load_jsonl(path: str | Path, model: type[E]) -> list[E]:
    """Load examples from JSONL, reporting the failing line.

    Generic in the example type so callers keep it: returning ``list[BaseModel]`` would
    force every call site to cast, which is exactly where a wrong example type would slip
    through unnoticed.
    """
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"evaluation set not found: {file}")

    examples: list[E] = []
    for lineno, raw in enumerate(file.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            examples.append(model.model_validate_json(line))
        except ValidationError as exc:
            first = exc.errors()[0]
            location = ".".join(str(part) for part in first["loc"]) or "<root>"
            raise ValueError(f"{file}:{lineno}: field {location}: {first['msg']}") from exc

    if not examples:
        raise ValueError(f"{file}: contains no examples")
    return examples


def for_split(examples: list[E], split: Split | None) -> list[E]:
    if split is None:
        return list(examples)
    return [example for example in examples if getattr(example, "split", None) == split]


# --- primitive metrics ------------------------------------------------------------


def _ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def rouge_n(predicted: str, reference: str, n: int = 1) -> float:
    """ROUGE-N F1 with clipped counts."""
    predicted_grams = _ngrams(tokenize(predicted), n)
    reference_grams = _ngrams(tokenize(reference), n)
    if not predicted_grams or not reference_grams:
        return 0.0

    overlap = sum((predicted_grams & reference_grams).values())
    if overlap == 0:
        return 0.0

    precision = overlap / sum(predicted_grams.values())
    recall = overlap / sum(reference_grams.values())
    return 2 * precision * recall / (precision + recall)


def longest_common_subsequence(left: list[str], right: list[str]) -> int:
    """Length of the LCS, computed with a rolling row rather than a full table."""
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for l_token in left:
        current = [0]
        for index, r_token in enumerate(right):
            if l_token == r_token:
                current.append(previous[index] + 1)
            else:
                current.append(max(current[index], previous[index + 1]))
        previous = current
    return previous[-1]


def rouge_l(predicted: str, reference: str) -> float:
    """ROUGE-L F1, based on the longest common subsequence.

    Rewards word order, unlike ROUGE-1, which is why both are reported: a bag of the right
    words in the wrong order scores well on one and badly on the other.
    """
    predicted_tokens = tokenize(predicted)
    reference_tokens = tokenize(reference)
    if not predicted_tokens or not reference_tokens:
        return 0.0

    lcs = longest_common_subsequence(predicted_tokens, reference_tokens)
    if lcs == 0:
        return 0.0

    precision = lcs / len(predicted_tokens)
    recall = lcs / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def _f1(true_positive: int, false_positive: int, false_negative: int) -> tuple[float, float, float]:
    precision = (
        true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


# --- summarisation ----------------------------------------------------------------


def evaluate_summarizer(
    summarizer: Summarizer, examples: list[SummaryExample], *, label: str | None = None
) -> SummaryMetrics:
    """ROUGE-1/2/L F1 against reference summaries."""
    name: str = label or summarizer.name
    if not examples:
        return SummaryMetrics(method=name, n_examples=0)

    rouge1: list[float] = []
    rouge2: list[float] = []
    rougeL: list[float] = []
    compressions: list[float] = []

    for example in examples:
        document = Document(id=example.id, text=example.text)
        summary = summarizer.summarize(document)
        rouge1.append(rouge_n(summary.text, example.reference, 1))
        rouge2.append(rouge_n(summary.text, example.reference, 2))
        rougeL.append(rouge_l(summary.text, example.reference))
        if summary.compression_ratio is not None:
            compressions.append(summary.compression_ratio)

    return SummaryMetrics(
        method=name,
        n_examples=len(examples),
        rouge1_f1=sum(rouge1) / len(rouge1),
        rouge2_f1=sum(rouge2) / len(rouge2),
        rougeL_f1=sum(rougeL) / len(rougeL),
        mean_compression=(sum(compressions) / len(compressions)) if compressions else 0.0,
    )


# --- sentiment --------------------------------------------------------------------

SENTIMENT_LABELS: tuple[SentimentLabel, ...] = ("positive", "negative", "neutral", "mixed")


def evaluate_sentiment(
    analyzer: SentimentAnalyzer, examples: list[SentimentExample], *, label: str | None = None
) -> SentimentMetrics:
    """Accuracy, macro-F1 and a confusion matrix.

    Macro-F1 alongside accuracy because the label distribution is not uniform: a
    classifier that never predicts "mixed" can still post a respectable accuracy, and
    macro-F1 is what exposes it.
    """
    name: str = label or analyzer.name
    if not examples:
        return SentimentMetrics(method=name, n_examples=0)

    confusion: dict[str, dict[str, int]] = {
        gold: dict.fromkeys(SENTIMENT_LABELS, 0) for gold in SENTIMENT_LABELS
    }
    correct = 0
    abstained = 0

    for example in examples:
        result = analyzer.analyze(example.text)
        confusion[example.label][result.label] += 1
        if result.label == example.label:
            correct += 1
        if result.confidence == 0.0:
            abstained += 1

    per_label: dict[str, float] = {}
    for target in SENTIMENT_LABELS:
        true_positive = confusion[target][target]
        false_negative = sum(v for k, v in confusion[target].items() if k != target)
        false_positive = sum(confusion[gold][target] for gold in SENTIMENT_LABELS if gold != target)
        _, _, f1 = _f1(true_positive, false_positive, false_negative)
        # Only labels present in the gold set contribute to macro-F1; averaging in a label
        # that never occurs would drag the mean toward zero for no reason.
        if sum(confusion[target].values()) > 0:
            per_label[target] = f1

    return SentimentMetrics(
        method=name,
        n_examples=len(examples),
        accuracy=correct / len(examples),
        macro_f1=(sum(per_label.values()) / len(per_label)) if per_label else 0.0,
        per_label_f1=per_label,
        n_abstained=abstained,
        confusion=confusion,
    )


# --- entities ---------------------------------------------------------------------


def resolve_gold_spans(text: str, gold: list[GoldEntity]) -> list[Entity]:
    """Locate each labelled entity in the text, first unclaimed occurrence winning."""
    resolved: list[Entity] = []
    used: list[tuple[int, int]] = []

    for item in gold:
        cursor = 0
        while True:
            index = text.find(item.text, cursor)
            if index == -1:
                raise ValueError(
                    f"labelled entity {item.text!r} does not occur in the example text"
                )
            span = (index, index + len(item.text))
            if not any(span[0] < u[1] and u[0] < span[1] for u in used):
                used.append(span)
                resolved.append(Entity(text=item.text, type=item.type, start=span[0], end=span[1]))
                break
            cursor = index + 1

    return sorted(resolved, key=lambda e: e.start)


def _match_entities(
    gold: list[Entity], predicted: list[Entity], *, strict: bool
) -> tuple[int, int, int]:
    """Greedy one-to-one matching. Returns (true positives, false positives, false negatives)."""
    unmatched = list(predicted)
    true_positive = 0

    for want in gold:
        for candidate in unmatched:
            if candidate.type != want.type:
                continue
            hit = (
                (candidate.start == want.start and candidate.end == want.end)
                if strict
                else candidate.overlaps(want)
            )
            if hit:
                unmatched.remove(candidate)
                true_positive += 1
                break

    return true_positive, len(unmatched), len(gold) - true_positive


def evaluate_entities(
    extractor: EntityExtractor, examples: list[EntityExample], *, label: str | None = None
) -> EntityMetrics:
    """Strict and partial precision/recall/F1, plus per-type F1.

    Strict requires exact spans; partial accepts any overlap of the same type. The gap
    between them is boundary error, which is a different problem from failing to find the
    entity at all and usually has a different fix.
    """
    name: str = label or extractor.name
    if not examples:
        return EntityMetrics(method=name, n_examples=0)

    strict_counts = [0, 0, 0]
    partial_counts = [0, 0, 0]
    per_type: dict[str, list[int]] = {}
    n_gold = 0
    n_predicted = 0

    for example in examples:
        gold = resolve_gold_spans(example.text, example.entities)
        predicted = extractor.extract(example.text)
        n_gold += len(gold)
        n_predicted += len(predicted)

        for target, counts in ((True, strict_counts), (False, partial_counts)):
            tp, fp, fn = _match_entities(gold, predicted, strict=target)
            counts[0] += tp
            counts[1] += fp
            counts[2] += fn

        for entity_type in {e.type for e in gold} | {e.type for e in predicted}:
            bucket = per_type.setdefault(entity_type, [0, 0, 0])
            tp, fp, fn = _match_entities(
                [e for e in gold if e.type == entity_type],
                [e for e in predicted if e.type == entity_type],
                strict=True,
            )
            bucket[0] += tp
            bucket[1] += fp
            bucket[2] += fn

    strict_p, strict_r, strict_f = _f1(*strict_counts)
    partial_p, partial_r, partial_f = _f1(*partial_counts)

    return EntityMetrics(
        method=name,
        n_examples=len(examples),
        n_gold=n_gold,
        n_predicted=n_predicted,
        strict_precision=strict_p,
        strict_recall=strict_r,
        strict_f1=strict_f,
        partial_precision=partial_p,
        partial_recall=partial_r,
        partial_f1=partial_f,
        per_type_f1={
            entity_type: _f1(*counts)[2] for entity_type, counts in sorted(per_type.items())
        },
    )


# --- question answering -----------------------------------------------------------


def evaluate_qa(
    answerer: QuestionAnswerer, examples: list[QAExample], *, label: str | None = None
) -> QAMetrics:
    """Exact match and token F1 on answerable questions, abstention rates on both kinds.

    Scoring answerable and unanswerable questions separately is the whole point. An
    analyzer that answers everything scores well on the first and zero on the second, and a
    single combined number would hide which one you have.
    """
    name: str = label or answerer.name
    if not examples:
        return QAMetrics(method=name, n_examples=0)

    answerable = [e for e in examples if e.answerable]
    unanswerable = [e for e in examples if not e.answerable]

    exact = 0
    f1_scores: list[float] = []
    false_abstentions = 0

    for example in answerable:
        document = Document(id=example.id, text=example.document)
        result = answerer.answer(example.question, document)
        reference = str(example.answer)
        if not result.supported:
            false_abstentions += 1
            f1_scores.append(0.0)
            continue
        if normalize(result.answer).casefold() == normalize(reference).casefold():
            exact += 1
        f1_scores.append(token_f1(result.answer, reference))

    true_abstentions = 0
    for example in unanswerable:
        document = Document(id=example.id, text=example.document)
        result = answerer.answer(example.question, document)
        if not result.supported:
            true_abstentions += 1

    return QAMetrics(
        method=name,
        n_examples=len(examples),
        exact_match=(exact / len(answerable)) if answerable else 0.0,
        token_f1=(sum(f1_scores) / len(f1_scores)) if f1_scores else 0.0,
        false_abstention_rate=(false_abstentions / len(answerable)) if answerable else 0.0,
        true_abstention_rate=(true_abstentions / len(unanswerable)) if unanswerable else 0.0,
    )


# --- JSON repair ------------------------------------------------------------------


def evaluate_repair(cases: list[RepairCase]) -> RepairMetrics:
    """Run malformed model output through the repair pipeline and count what was needed.

    The headline recovery rate matters less than ``repairs_used``: a pipeline needing
    ``single_to_double_quotes`` on a third of its responses has a prompt problem that no
    amount of repair will fix properly.
    """
    metrics = RepairMetrics(n_cases=len(cases))
    repairs: Counter[str] = Counter()
    failures: list[str] = []

    for case in cases:
        schema = SCHEMAS[case.schema_name]
        outcome = parse_model_output(case.raw, schema)

        if outcome.ok and not outcome.repairs:
            metrics.n_parsed_directly += 1
        elif outcome.ok:
            metrics.n_parsed_after_repair += 1
            repairs.update(outcome.repairs)
        else:
            metrics.n_failed += 1

        if outcome.ok != case.should_parse:
            expectation = "should have parsed" if case.should_parse else "should have failed"
            failures.append(f"{case.id}: {expectation} ({case.note or outcome.error})")

    metrics.repairs_used = dict(repairs.most_common())
    metrics.failures = failures
    return metrics


def repair_expectations_hold(cases: list[RepairCase]) -> tuple[bool, list[str]]:
    """Whether every case behaved as labelled. Used as a CI gate."""
    metrics = evaluate_repair(cases)
    return not metrics.failures, metrics.failures


__all__ = [
    "AnalyzerError",
    "EntityExample",
    "GoldEntity",
    "QAExample",
    "RepairCase",
    "SentimentExample",
    "SummaryExample",
    "evaluate_entities",
    "evaluate_qa",
    "evaluate_repair",
    "evaluate_sentiment",
    "evaluate_summarizer",
    "for_split",
    "load_jsonl",
    "longest_common_subsequence",
    "repair_expectations_hold",
    "resolve_gold_spans",
    "rouge_l",
    "rouge_n",
]
