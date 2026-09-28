from __future__ import annotations

from itertools import pairwise

import pytest

from analyzer.analyzers.base import AnalyzerError, resolve_overlaps
from analyzer.analyzers.entities import LLMEntityExtractor, RuleEntityExtractor
from analyzer.analyzers.qa import LLMQA, NO_ANSWER, ExtractiveQA
from analyzer.analyzers.sentiment import LexiconSentiment, LLMSentiment, classify
from analyzer.analyzers.summarize import (
    CentroidSummarizer,
    LeadSummarizer,
    LLMSummarizer,
    build_summarizer,
)
from analyzer.providers import ProviderError, ScriptedLLM
from analyzer.types import Document, Entity

# --- summarisation ----------------------------------------------------------------


def document(text: str) -> Document:
    return Document(id="d", text=text)


FOUR_SENTENCES = (
    "The council approved the plan on Monday. Work begins in March. "
    "Four million was allocated. Two hundred spaces will be added."
)


def test_lead_takes_the_first_sentences_in_order() -> None:
    summary = LeadSummarizer(2).summarize(document(FOUR_SENTENCES))
    assert summary.sentences == [
        "The council approved the plan on Monday.",
        "Work begins in March.",
    ]


def test_lead_reports_its_compression_ratio() -> None:
    summary = LeadSummarizer(2).summarize(document(FOUR_SENTENCES))
    assert summary.compression_ratio is not None
    assert 0.0 < summary.compression_ratio < 1.0


def test_centroid_emits_sentences_in_document_order() -> None:
    """Not score order: a summary shuffled by relevance loses the original discourse."""
    text = (
        "Unrelated opening remark. The migration moved every record to the new schema. "
        "The schema migration completed overnight. A closing aside."
    )
    summary = CentroidSummarizer(2).summarize(document(text))
    positions = [text.index(sentence.rstrip(".")) for sentence in summary.sentences]
    assert positions == sorted(positions)


def test_centroid_prefers_content_bearing_sentences() -> None:
    text = "Yes. The quarterly revenue figures were restated after the audit. No."
    summary = CentroidSummarizer(1).summarize(document(text))
    assert "revenue" in summary.text


def test_summarising_empty_prose_yields_an_empty_summary() -> None:
    summary = CentroidSummarizer(2).summarize(Document(id="d", text="   ..."))
    assert summary.sentences in ([], ["..."])


def test_a_nonsense_sentence_budget_is_rejected() -> None:
    for factory in (LeadSummarizer, CentroidSummarizer):
        with pytest.raises(ValueError, match="max_sentences must be >= 1"):
            factory(0)


def test_llm_summariser_parses_a_valid_response() -> None:
    llm = ScriptedLLM(['{"summary": "The council approved the plan."}'])
    summary = LLMSummarizer(llm).summarize(document(FOUR_SENTENCES))
    assert summary.text == "The council approved the plan."
    assert summary.method.startswith("llm-summary")


def test_llm_summariser_repairs_a_fenced_response() -> None:
    llm = ScriptedLLM(['```json\n{"summary": "Fenced but fine.",}\n```'])
    summariser = LLMSummarizer(llm)
    assert summariser.summarize(document(FOUR_SENTENCES)).text == "Fenced but fine."
    assert summariser.last_repairs == ("strip_code_fence", "remove_trailing_commas")


def test_llm_summariser_raises_on_unusable_output() -> None:
    llm = ScriptedLLM(["I would rather not."])
    with pytest.raises(AnalyzerError, match="unusable"):
        LLMSummarizer(llm).summarize(document(FOUR_SENTENCES))


def test_build_summarizer_by_name() -> None:
    assert build_summarizer("lead").name == "lead-3"
    assert build_summarizer("centroid", max_sentences=2).name == "centroid-2"
    with pytest.raises(ValueError, match="unknown summariser"):
        build_summarizer("abstractive")


def test_build_summarizer_requires_a_provider_for_llm() -> None:
    with pytest.raises(ValueError, match="needs a provider"):
        build_summarizer("llm")


# --- sentiment --------------------------------------------------------------------


def test_clear_positive_and_negative_text() -> None:
    analyzer = LexiconSentiment()
    assert analyzer.analyze("An excellent, reliable release.").label == "positive"
    assert analyzer.analyze("A terrible, unreliable release.").label == "negative"


def test_text_with_no_sentiment_words_is_neutral_with_no_confidence() -> None:
    """Zero confidence means "no signal", which is different from "balanced signal"."""
    result = LexiconSentiment().analyze("The endpoint accepts a cursor parameter.")
    assert result.label == "neutral"
    assert result.confidence == 0.0
    assert result.score == 0.0


def test_negation_flips_polarity() -> None:
    result = LexiconSentiment().analyze("This is not good.")
    assert result.label == "negative"
    assert result.negations == 1


def test_a_negated_negative_reads_as_positive() -> None:
    assert LexiconSentiment().analyze("Not disappointing at all.").label == "positive"


def test_negation_only_reaches_back_a_few_tokens() -> None:
    # "not" is too far from "excellent" to negate it.
    result = LexiconSentiment().analyze("This is not the release we planned, but it is excellent.")
    assert "excellent" in result.positive_terms


def test_an_intensifier_increases_magnitude() -> None:
    analyzer = LexiconSentiment()
    plain = analyzer.analyze("The service is slow.")
    intense = analyzer.analyze("The service is extremely slow.")
    assert intense.confidence > plain.confidence


def test_a_diminisher_reduces_magnitude() -> None:
    analyzer = LexiconSentiment()
    plain = analyzer.analyze("The service is slow.")
    mild = analyzer.analyze("The service is slightly slow.")
    assert mild.confidence < plain.confidence


def test_balanced_sentiment_is_mixed() -> None:
    result = LexiconSentiment().analyze("Great documentation, terrible pricing.")
    assert result.label == "mixed"
    assert result.positive_terms and result.negative_terms


def test_the_terms_that_drove_the_decision_are_reported() -> None:
    result = LexiconSentiment().analyze("The docs are helpful but the app is buggy.")
    assert "helpful" in result.positive_terms
    assert "buggy" in result.negative_terms


def test_classify_is_testable_on_its_own() -> None:
    assert classify(0.0, 0.0, 0.0) == "neutral"
    assert classify(1.0, 3.0, 0.0) == "positive"
    assert classify(-1.0, 0.0, 3.0) == "negative"
    assert classify(0.0, 2.0, 2.0) == "mixed"
    # Below the polarity threshold but not balanced enough to be mixed.
    assert classify(0.1, 1.1, 0.9) == "mixed"


def test_llm_sentiment_uses_the_models_label() -> None:
    llm = ScriptedLLM(['{"label": "mixed", "score": 0.1, "rationale": "both"}'])
    result = LLMSentiment(llm).analyze("Anything.")
    assert result.label == "mixed"
    assert result.method.startswith("llm-sentiment")


def test_llm_sentiment_rejects_an_invalid_label() -> None:
    llm = ScriptedLLM(['{"label": "great", "score": 0.9}'])
    with pytest.raises(AnalyzerError, match="unusable"):
        LLMSentiment(llm).analyze("Anything.")


# --- entities ---------------------------------------------------------------------

ENTITY_TEXT = (
    "Dr Amara Okonkwo joined Meridian Technologies GmbH in Berlin on 14 March 2024, "
    "reporting £1,250,000 and 18% growth. Contact press@meridian.example or "
    "https://meridian.example/news."
)


def test_each_pattern_category_is_found() -> None:
    entities = RuleEntityExtractor().extract(ENTITY_TEXT)
    by_type = {entity.type: entity.text for entity in entities}
    assert by_type["PERSON"] == "Dr Amara Okonkwo"
    assert by_type["ORG"] == "Meridian Technologies GmbH"
    assert by_type["LOCATION"] == "Berlin"
    assert by_type["DATE"] == "14 March 2024"
    assert by_type["MONEY"] == "£1,250,000"
    assert by_type["PERCENT"] == "18%"
    assert by_type["EMAIL"] == "press@meridian.example"
    assert by_type["URL"] == "https://meridian.example/news"


def test_spans_index_the_source_text() -> None:
    for entity in RuleEntityExtractor().extract(ENTITY_TEXT):
        assert ENTITY_TEXT[entity.start : entity.end] == entity.text


def test_entities_are_returned_in_document_order() -> None:
    entities = RuleEntityExtractor().extract(ENTITY_TEXT)
    assert [e.start for e in entities] == sorted(e.start for e in entities)


def test_entities_do_not_overlap() -> None:
    entities = RuleEntityExtractor().extract(ENTITY_TEXT)
    for earlier, later in pairwise(entities):
        assert earlier.end <= later.start


def test_a_trailing_full_stop_is_not_part_of_a_url() -> None:
    """Found with the wrong boundaries is still wrong, and the evaluation counts it so."""
    entities = RuleEntityExtractor().extract("Docs live at https://example.com/docs.")
    assert [e.text for e in entities if e.type == "URL"] == ["https://example.com/docs"]


def test_neighbouring_locations_are_not_merged_into_one_name() -> None:
    # "Oslo to Copenhagen" was a single run, and then a single wrong PERSON.
    entities = RuleEntityExtractor().extract("She moved from Oslo to Copenhagen last year.")
    assert sorted(e.text for e in entities if e.type == "LOCATION") == ["Copenhagen", "Oslo"]


def test_an_x_of_y_organisation_is_recognised() -> None:
    entities = RuleEntityExtractor().extract("The Bank of England raised rates.")
    assert [(e.type, e.text) for e in entities] == [("ORG", "Bank of England")]


def test_a_single_capitalised_word_is_not_a_person() -> None:
    # Otherwise every sentence opening becomes a false positive.
    entities = RuleEntityExtractor().extract("Approved the request without comment.")
    assert [e for e in entities if e.type == "PERSON"] == []


def test_text_with_nothing_to_find_yields_nothing() -> None:
    assert RuleEntityExtractor().extract("the cat sat on the mat") == []


def test_resolve_overlaps_keeps_the_longest() -> None:
    short = Entity(text="Acme", type="ORG", start=0, end=4)
    long = Entity(text="Acme Corp", type="ORG", start=0, end=9)
    assert resolve_overlaps([short, long]) == [long]


def test_llm_entities_are_located_in_the_source() -> None:
    llm = ScriptedLLM(['{"entities": [{"text": "Berlin", "type": "LOCATION"}]}'])
    entities = LLMEntityExtractor(llm).extract("The office is in Berlin.")
    assert entities[0].start == 17
    assert entities[0].end == 23


def test_llm_entities_not_present_in_the_text_are_dropped() -> None:
    """The cheapest available hallucination filter."""
    llm = ScriptedLLM(
        [
            '{"entities": [{"text": "Berlin", "type": "LOCATION"},'
            ' {"text": "Atlantis", "type": "LOCATION"}]}'
        ]
    )
    extractor = LLMEntityExtractor(llm)
    entities = extractor.extract("The office is in Berlin.")
    assert [e.text for e in entities] == ["Berlin"]
    assert extractor.last_dropped == ["Atlantis"]


def test_repeated_llm_entities_get_distinct_spans() -> None:
    llm = ScriptedLLM(
        ['{"entities": [{"text": "Acme", "type": "ORG"}, {"text": "Acme", "type": "ORG"}]}']
    )
    entities = LLMEntityExtractor(llm).extract("Acme bought Acme.")
    assert [e.start for e in entities] == [0, 12]


# --- question answering -----------------------------------------------------------

QA_DOC = Document(
    id="d",
    text=(
        "The maintenance window runs from 02:00 to 04:00 UTC every second Sunday. "
        "Scheduled jobs are paused and resume automatically afterwards. "
        "Enterprise customers can request an alternative window."
    ),
)


def test_an_answerable_question_is_answered_with_evidence() -> None:
    answer = ExtractiveQA().answer("When does the maintenance window run?", QA_DOC)
    assert answer.supported
    assert "02:00 to 04:00 UTC" in answer.answer
    assert answer.evidence


def test_an_unanswerable_question_is_declined() -> None:
    answer = ExtractiveQA().answer("What is the support phone number?", QA_DOC)
    assert not answer.supported
    assert answer.answer == NO_ANSWER


def test_a_declined_answer_still_reports_its_near_miss() -> None:
    # So a caller can show a weak match rather than showing nothing.
    answer = ExtractiveQA().answer("What is the support phone number?", QA_DOC)
    assert answer.evidence


def test_an_empty_question_is_declined() -> None:
    assert not ExtractiveQA().answer("   ", QA_DOC).supported


def test_a_nonsense_threshold_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"threshold must be in \[0, 1\]"):
        ExtractiveQA(threshold=1.5)


def test_llm_qa_keeps_evidence_present_in_the_document() -> None:
    llm = ScriptedLLM(
        [
            '{"answer": "Every second Sunday.", "supported": true,'
            ' "evidence": ["The maintenance window runs from 02:00 to 04:00'
            ' UTC every second Sunday."]}'
        ]
    )
    answer = LLMQA(llm).answer("When?", QA_DOC)
    assert answer.supported
    assert len(answer.evidence) == 1


def test_llm_qa_drops_evidence_absent_from_the_document() -> None:
    llm = ScriptedLLM(
        [
            '{"answer": "Every third Tuesday.", "supported": true,'
            ' "evidence": ["The window runs every third Tuesday."]}'
        ]
    )
    answerer = LLMQA(llm)
    answer = answerer.answer("When?", QA_DOC)

    # All evidence disappeared, so the answer is downgraded rather than trusted.
    assert not answer.supported
    assert answer.answer == NO_ANSWER
    assert answerer.last_dropped_evidence


def test_llm_qa_honours_an_explicit_refusal() -> None:
    llm = ScriptedLLM(['{"answer": "No idea.", "supported": false, "evidence": []}'])
    assert not LLMQA(llm).answer("When?", QA_DOC).supported


# --- providers --------------------------------------------------------------------


def test_scripted_provider_records_prompts() -> None:
    llm = ScriptedLLM(['{"summary": "x"}'])
    LLMSummarizer(llm).summarize(document(FOUR_SENTENCES))
    assert len(llm.prompts) == 1
    assert "DOCUMENT" in llm.prompts[0]


def test_scripted_provider_raises_when_exhausted() -> None:
    llm = ScriptedLLM([])
    with pytest.raises(ProviderError, match="no response left"):
        llm.complete("anything")
