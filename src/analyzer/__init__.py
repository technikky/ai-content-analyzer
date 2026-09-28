"""Content analysis: summarisation, sentiment, entity extraction and document Q&A.

Every analysis has a deterministic local baseline and an optional LLM implementation behind
the same interface, so the value the model adds over free is measurable rather than assumed.
"""

from __future__ import annotations

from analyzer.analyzers import (
    CentroidSummarizer,
    ExtractiveQA,
    LeadSummarizer,
    LexiconSentiment,
    RuleEntityExtractor,
    build_entity_extractor,
    build_question_answerer,
    build_sentiment_analyzer,
    build_summarizer,
)
from analyzer.evaluation import (
    evaluate_entities,
    evaluate_qa,
    evaluate_repair,
    evaluate_sentiment,
    evaluate_summarizer,
    rouge_l,
    rouge_n,
)
from analyzer.library import TfidfLibrary, build_library
from analyzer.pipeline import ALL_ANALYSES, AnalysisCache, ContentAnalyzer
from analyzer.providers import ScriptedLLM, build_llm
from analyzer.schemas import parse_json, parse_model_output
from analyzer.types import (
    Analysis,
    Document,
    Entity,
    QAAnswer,
    SearchMatch,
    SentimentResult,
    Summary,
)

__version__ = "0.1.0"

__all__ = [
    "ALL_ANALYSES",
    "Analysis",
    "AnalysisCache",
    "CentroidSummarizer",
    "ContentAnalyzer",
    "Document",
    "Entity",
    "ExtractiveQA",
    "LeadSummarizer",
    "LexiconSentiment",
    "QAAnswer",
    "RuleEntityExtractor",
    "ScriptedLLM",
    "SearchMatch",
    "SentimentResult",
    "Summary",
    "TfidfLibrary",
    "__version__",
    "build_entity_extractor",
    "build_library",
    "build_llm",
    "build_question_answerer",
    "build_sentiment_analyzer",
    "build_summarizer",
    "evaluate_entities",
    "evaluate_qa",
    "evaluate_repair",
    "evaluate_sentiment",
    "evaluate_summarizer",
    "parse_json",
    "parse_model_output",
    "rouge_l",
    "rouge_n",
]
