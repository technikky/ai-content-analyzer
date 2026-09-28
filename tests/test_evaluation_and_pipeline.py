"""Evaluation metrics, the pipeline, the library, and the measured floors CI enforces."""

from __future__ import annotations

from pathlib import Path

import pytest

from analyzer.analyzers.entities import RuleEntityExtractor
from analyzer.analyzers.qa import ExtractiveQA
from analyzer.analyzers.sentiment import LexiconSentiment
from analyzer.analyzers.summarize import CentroidSummarizer, LeadSummarizer
from analyzer.evaluation import (
    EntityExample,
    GoldEntity,
    QAExample,
    RepairCase,
    SentimentExample,
    SummaryExample,
    evaluate_entities,
    evaluate_qa,
    evaluate_repair,
    evaluate_sentiment,
    evaluate_summarizer,
    for_split,
    longest_common_subsequence,
    resolve_gold_spans,
    rouge_l,
    rouge_n,
)
from analyzer.library import LibraryError, TfidfLibrary
from analyzer.pipeline import AnalysisCache, ContentAnalyzer
from analyzer.providers import ScriptedLLM
from analyzer.types import Document

# --- ROUGE ------------------------------------------------------------------------


def test_rouge_is_one_for_identical_text() -> None:
    assert rouge_n("the cat sat", "the cat sat", 1) == pytest.approx(1.0)
    assert rouge_l("the cat sat", "the cat sat") == pytest.approx(1.0)


def test_rouge_is_zero_with_no_overlap() -> None:
    assert rouge_n("alpha beta", "gamma delta", 1) == 0.0
    assert rouge_l("alpha beta", "gamma delta") == 0.0


def test_rouge1_ignores_word_order_but_rouge_l_does_not() -> None:
    """Why both are reported: a bag of the right words scores well on one, badly on the other."""
    shuffled = rouge_n("sat cat the", "the cat sat", 1)
    ordered = rouge_l("sat cat the", "the cat sat")
    assert shuffled == pytest.approx(1.0)
    assert ordered < 1.0


def test_rouge2_requires_adjacent_pairs() -> None:
    assert rouge_n("the cat sat", "the cat sat", 2) == pytest.approx(1.0)
    assert rouge_n("the sat cat", "the cat sat", 2) < 1.0


def test_rouge_clips_repeated_ngrams() -> None:
    # Repeating a word must not inflate the score.
    assert rouge_n("cat cat cat", "the cat sat", 1) < 0.5


def test_longest_common_subsequence_basics() -> None:
    assert longest_common_subsequence(list("abcde"), list("ace")) == 3
    assert longest_common_subsequence([], ["a"]) == 0


def test_rouge_rejects_a_nonsense_n() -> None:
    with pytest.raises(ValueError, match="n must be >= 1"):
        rouge_n("a", "a", 0)


# --- gold span resolution ---------------------------------------------------------


def test_gold_spans_are_derived_from_the_text() -> None:
    resolved = resolve_gold_spans(
        "Acme Ltd is in Berlin.", [GoldEntity(text="Berlin", type="LOCATION")]
    )
    assert resolved[0].start == 15
    assert resolved[0].end == 21


def test_repeated_gold_entities_get_distinct_spans() -> None:
    resolved = resolve_gold_spans(
        "Acme met Acme.",
        [GoldEntity(text="Acme", type="ORG"), GoldEntity(text="Acme", type="ORG")],
    )
    assert [e.start for e in resolved] == [0, 9]


def test_a_label_absent_from_the_text_is_an_error() -> None:
    """A typo in the labels would otherwise silently depress every recall figure."""
    with pytest.raises(ValueError, match="does not occur"):
        resolve_gold_spans("Acme Ltd.", [GoldEntity(text="Globex", type="ORG")])


# --- metric behaviour -------------------------------------------------------------


def test_sentiment_macro_f1_ignores_labels_absent_from_the_gold_set() -> None:
    examples = [
        SentimentExample(id="a", text="An excellent release.", label="positive"),
        SentimentExample(id="b", text="A terrible release.", label="negative"),
    ]
    metrics = evaluate_sentiment(LexiconSentiment(), examples)
    assert set(metrics.per_label_f1) == {"positive", "negative"}
    assert metrics.accuracy == pytest.approx(1.0)


def test_sentiment_confusion_rows_sum_to_the_gold_counts() -> None:
    examples = [
        SentimentExample(id="a", text="An excellent release.", label="positive"),
        SentimentExample(id="b", text="A terrible release.", label="negative"),
    ]
    metrics = evaluate_sentiment(LexiconSentiment(), examples)
    assert sum(metrics.confusion["positive"].values()) == 1
    assert sum(metrics.confusion["mixed"].values()) == 0


def test_entity_strict_and_partial_differ_on_boundary_errors() -> None:
    """The gap between them is boundary error, a different problem from a miss."""

    class OffByOne:
        name = "off-by-one"

        def extract(self, text: str) -> list:
            from analyzer.types import Entity

            index = text.index("Berlin")
            # One character too long: overlapping but not exact.
            return [
                Entity(text=text[index : index + 7], type="LOCATION", start=index, end=index + 7)
            ]

    examples = [
        EntityExample(
            id="a",
            text="The office is in Berlin.",
            entities=[GoldEntity(text="Berlin", type="LOCATION")],
        )
    ]
    metrics = evaluate_entities(OffByOne(), examples)
    assert metrics.strict_f1 == 0.0
    assert metrics.partial_f1 == pytest.approx(1.0)


def test_qa_separates_answerable_from_unanswerable() -> None:
    document_text = "The window runs from 02:00 to 04:00 UTC every second Sunday."
    examples = [
        QAExample(
            id="a",
            document=document_text,
            question="When does the window run?",
            answer="The window runs from 02:00 to 04:00 UTC every second Sunday.",
        ),
        QAExample(
            id="b", document=document_text, question="What is the phone number?", answer=None
        ),
    ]
    metrics = evaluate_qa(ExtractiveQA(), examples)
    assert metrics.exact_match == pytest.approx(1.0)
    assert metrics.true_abstention_rate == pytest.approx(1.0)
    assert metrics.false_abstention_rate == 0.0


def test_an_analyzer_that_answers_everything_scores_zero_on_abstention() -> None:
    class AlwaysAnswers:
        name = "always"

        def answer(self, question: str, document: Document):
            from analyzer.types import QAAnswer

            return QAAnswer(question=question, answer="Something.", supported=True, confidence=1.0)

    examples = [QAExample(id="b", document="A document.", question="Unanswerable?", answer=None)]
    assert evaluate_qa(AlwaysAnswers(), examples).true_abstention_rate == 0.0


def test_empty_example_sets_return_zeroed_metrics() -> None:
    assert evaluate_summarizer(LeadSummarizer(), []).n_examples == 0
    assert evaluate_sentiment(LexiconSentiment(), []).n_examples == 0
    assert evaluate_entities(RuleEntityExtractor(), []).n_examples == 0
    assert evaluate_qa(ExtractiveQA(), []).n_examples == 0


# --- the shipped evaluation sets --------------------------------------------------


def test_every_evaluation_set_has_both_splits(
    summary_examples: list[SummaryExample],
    sentiment_examples: list[SentimentExample],
    entity_examples: list[EntityExample],
    qa_examples: list[QAExample],
) -> None:
    for examples in (summary_examples, sentiment_examples, entity_examples, qa_examples):
        assert for_split(examples, "dev")
        assert for_split(examples, "test")


def test_every_entity_label_occurs_in_its_text(entity_examples: list[EntityExample]) -> None:
    for example in entity_examples:
        resolve_gold_spans(example.text, example.entities)


def test_the_sentiment_set_covers_every_label(
    sentiment_examples: list[SentimentExample],
) -> None:
    assert {e.label for e in sentiment_examples} == {"positive", "negative", "neutral", "mixed"}


def test_the_qa_set_contains_unanswerable_questions(qa_examples: list[QAExample]) -> None:
    # Without them, an analyzer that always answers looks perfect.
    assert any(not e.answerable for e in qa_examples)
    assert any(e.answerable for e in qa_examples)


def test_example_ids_are_unique(
    summary_examples: list[SummaryExample],
    sentiment_examples: list[SentimentExample],
    entity_examples: list[EntityExample],
    qa_examples: list[QAExample],
) -> None:
    for examples in (summary_examples, sentiment_examples, entity_examples, qa_examples):
        ids = [e.id for e in examples]
        assert len(set(ids)) == len(ids)


# --- measured floors, as a regression gate ----------------------------------------
#
# Set below the figures reported in the README so ordinary tuning does not fail CI, while
# a change that genuinely breaks an analyzer does.


def test_sentiment_accuracy_holds_on_the_held_out_split(
    sentiment_examples: list[SentimentExample],
) -> None:
    metrics = evaluate_sentiment(LexiconSentiment(), for_split(sentiment_examples, "test"))
    assert metrics.accuracy >= 0.85, metrics
    assert metrics.macro_f1 >= 0.80, metrics


def test_entity_f1_holds_on_the_held_out_split(entity_examples: list[EntityExample]) -> None:
    metrics = evaluate_entities(RuleEntityExtractor(), for_split(entity_examples, "test"))
    assert metrics.strict_f1 >= 0.85, metrics


def test_qa_declines_unanswerable_questions_on_the_held_out_split(
    qa_examples: list[QAExample],
) -> None:
    metrics = evaluate_qa(ExtractiveQA(), for_split(qa_examples, "test"))
    assert metrics.true_abstention_rate >= 0.8, metrics
    assert metrics.token_f1 >= 0.6, metrics


def test_centroid_beats_lead_on_the_held_out_split(
    summary_examples: list[SummaryExample],
) -> None:
    """The comparison the README reports; if it inverts, the claim needs rewriting."""
    held_out = for_split(summary_examples, "test")
    lead = evaluate_summarizer(LeadSummarizer(3), held_out)
    centroid = evaluate_summarizer(CentroidSummarizer(3), held_out)
    assert centroid.rougeL_f1 > lead.rougeL_f1


def test_every_repair_case_behaves_as_labelled(repair_cases: list[RepairCase]) -> None:
    metrics = evaluate_repair(repair_cases)
    assert metrics.failures == []
    assert metrics.recovery_rate is not None
    assert metrics.n_parsed_after_repair > 0


# --- pipeline ---------------------------------------------------------------------


def test_the_pipeline_runs_every_requested_analysis(
    analyzer: ContentAnalyzer, sample_document: Document
) -> None:
    result = analyzer.analyze(sample_document, analyses=["summary", "sentiment", "entities"])
    assert result.summary is not None
    assert result.sentiment is not None
    assert result.entities
    assert result.qa is None
    assert set(result.timings_ms) == {"summary", "sentiment", "entities"}


def test_qa_requires_a_question(analyzer: ContentAnalyzer, sample_document: Document) -> None:
    with pytest.raises(ValueError, match="needs a question"):
        analyzer.analyze(sample_document, analyses=["qa"])


def test_an_unknown_analysis_is_rejected(
    analyzer: ContentAnalyzer, sample_document: Document
) -> None:
    with pytest.raises(ValueError, match="unknown analyses"):
        analyzer.analyze(sample_document, analyses=["astrology"])  # type: ignore[list-item]


def test_one_failing_stage_does_not_lose_the_others(sample_document: Document) -> None:
    """A document that breaks one analyzer should still return the rest."""
    broken = ContentAnalyzer(
        summarizer=__import__(
            "analyzer.analyzers.summarize", fromlist=["LLMSummarizer"]
        ).LLMSummarizer(ScriptedLLM(["not json at all"])),
        cache=AnalysisCache(enabled=False),
    )
    result = broken.analyze(sample_document, analyses=["summary", "sentiment"])

    assert result.summary is None
    assert result.sentiment is not None
    assert "summary" in broken.last_errors


def test_timings_are_recorded_per_stage(
    analyzer: ContentAnalyzer, sample_document: Document
) -> None:
    result = analyzer.analyze(sample_document, analyses=["summary", "entities"])
    assert result.timings_ms["summary"] >= 0.0
    assert result.total_ms == pytest.approx(sum(result.timings_ms.values()))


def test_the_cache_returns_a_previous_analysis(tmp_path: Path, sample_document: Document) -> None:
    engine = ContentAnalyzer(cache=AnalysisCache(tmp_path))
    first = engine.analyze(sample_document, analyses=["summary"])
    second = engine.analyze(sample_document, analyses=["summary"])

    assert not first.cached
    assert second.cached
    assert engine.cache.hits == 1


def test_changing_the_request_misses_the_cache(tmp_path: Path, sample_document: Document) -> None:
    engine = ContentAnalyzer(cache=AnalysisCache(tmp_path))
    engine.analyze(sample_document, analyses=["summary"])
    second = engine.analyze(sample_document, analyses=["summary", "sentiment"])
    assert not second.cached


def test_a_corrupt_cache_entry_is_treated_as_a_miss(
    tmp_path: Path, sample_document: Document
) -> None:
    engine = ContentAnalyzer(cache=AnalysisCache(tmp_path))
    engine.analyze(sample_document, analyses=["summary"])
    for path in tmp_path.rglob("*.json"):
        path.write_text("{ not json", encoding="utf-8")

    assert not engine.analyze(sample_document, analyses=["summary"]).cached


# --- library ----------------------------------------------------------------------


def test_search_finds_the_right_document(library: TfidfLibrary) -> None:
    matches = library.search("how long do refunds take", top_k=2)
    assert matches[0].document_id == "kb/refunds"
    assert matches[0].rank == 1


def test_search_returns_nothing_for_unrelated_vocabulary(library: TfidfLibrary) -> None:
    assert library.search("kangaroo migration patterns") == []


def test_search_on_an_empty_query_returns_nothing(library: TfidfLibrary) -> None:
    assert library.search("   ") == []


def test_matches_carry_a_snippet(library: TfidfLibrary) -> None:
    assert library.search("access tokens", top_k=1)[0].snippet


def test_a_duplicate_document_is_refused(library: TfidfLibrary) -> None:
    with pytest.raises(LibraryError, match="already in the library"):
        library.add([Document(id="kb/refunds", text="A different body.")])


def test_adding_nothing_is_an_error(library: TfidfLibrary) -> None:
    with pytest.raises(ValueError, match="cannot add zero documents"):
        library.add([])


def test_the_library_round_trips_through_disk(library: TfidfLibrary, tmp_path: Path) -> None:
    before = library.search("refund timing", top_k=3)
    library.save(tmp_path / "lib")

    restored = TfidfLibrary.load(tmp_path / "lib")
    after = restored.search("refund timing", top_k=3)

    assert len(restored) == len(library)
    assert [m.document_id for m in before] == [m.document_id for m in after]


def test_loading_from_a_directory_with_no_library_raises(tmp_path: Path) -> None:
    with pytest.raises(LibraryError, match="no library at"):
        TfidfLibrary.load(tmp_path)


def test_the_shipped_corpus_loads_and_is_searchable() -> None:
    from tests.conftest import CORPUS_FILE

    documents = [
        Document.model_validate_json(line)
        for line in CORPUS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert len(documents) >= 10

    store = TfidfLibrary()
    store.add(documents)
    assert store.search("how long are logs retained", top_k=1)[0].document_id == "kb/data-retention"
