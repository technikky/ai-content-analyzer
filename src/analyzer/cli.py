"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from analyzer.analyzers.base import AnalyzerError
from analyzer.analyzers.entities import RuleEntityExtractor, build_entity_extractor
from analyzer.analyzers.qa import ExtractiveQA, build_question_answerer
from analyzer.analyzers.sentiment import LexiconSentiment, build_sentiment_analyzer
from analyzer.analyzers.summarize import (
    CentroidSummarizer,
    LeadSummarizer,
    build_summarizer,
)
from analyzer.evaluation import (
    EntityExample,
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
    load_jsonl,
)
from analyzer.library import LibraryError, TfidfLibrary
from analyzer.pipeline import ALL_ANALYSES, AnalysisCache, ContentAnalyzer
from analyzer.providers import ProviderError, build_llm
from analyzer.report import (
    confusion_table,
    entity_table,
    qa_table,
    repair_table,
    sentiment_table,
    summary_table,
    write_json,
    write_text,
)
from analyzer.types import Document

PROVIDERS = ("openai", "anthropic")
DEFAULT_DATA = Path("data/eval")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analyzer",
        description="Summarise, classify sentiment, extract entities and answer questions.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="analyse a text file or stdin")
    analyze.add_argument("path", nargs="?", default="-", help="file to analyse, or - for stdin")
    analyze.add_argument(
        "--analyses",
        default="summary,sentiment,entities",
        help=f"comma-separated subset of: {','.join(ALL_ANALYSES)}",
    )
    analyze.add_argument("--question", default=None, help="question for the qa analysis")
    analyze.add_argument("--summarizer", default="lead", choices=("lead", "centroid", "llm"))
    analyze.add_argument("--provider", default=None, choices=PROVIDERS, help="LLM provider")
    analyze.add_argument("--model", default=None, help="model id for the provider")
    analyze.add_argument(
        "--llm-analyses", default="", help="comma-separated analyses to run via LLM"
    )
    analyze.add_argument("--json", action="store_true", help="emit the full analysis as JSON")

    index = sub.add_parser("index", help="build a searchable library from a JSONL corpus")
    index.add_argument("corpus", help="JSONL with id, text and optional title")
    index.add_argument("--out", required=True, help="directory to write the library to")

    search = sub.add_parser("search", help="search a library")
    search.add_argument("query")
    search.add_argument("--library", required=True)
    search.add_argument("--top-k", type=int, default=5)

    evaluate = sub.add_parser("evaluate", help="measure the analyzers against labelled data")
    evaluate.add_argument("--data", default=str(DEFAULT_DATA), help="directory of evaluation sets")
    evaluate.add_argument(
        "--split",
        default="both",
        choices=("dev", "test", "both", "all"),
        help="which split to report; 'all' pools dev and test",
    )
    evaluate.add_argument("--out", default=None, help="directory for metrics.json and .md")

    repair = sub.add_parser("repair", help="run the malformed-output corpus through the parser")
    repair.add_argument("--cases", default=str(DEFAULT_DATA / "repair_cases.jsonl"))
    repair.add_argument("--out", default=None)

    serve = sub.add_parser("serve", help="run the HTTP API")
    serve.add_argument("--library", default=None, help="library directory to load at startup")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    return parser


def _read_text(path: str) -> str:
    if path == "-":
        text = sys.stdin.read()
        if not text.strip():
            raise ValueError("no text on stdin")
        return text
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"file not found: {file}")
    text = file.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"{file}: file is empty")
    return text


def _build_analyzer(args: argparse.Namespace) -> ContentAnalyzer:
    llm_kinds = {kind.strip() for kind in (args.llm_analyses or "").split(",") if kind.strip()}
    llm = None
    if llm_kinds or args.summarizer == "llm":
        if not args.provider:
            raise ValueError("--provider is required when any analysis runs via an LLM")
        llm = build_llm(args.provider, args.model)

    summarizer_method = "llm" if "summary" in llm_kinds else args.summarizer
    return ContentAnalyzer(
        summarizer=build_summarizer(summarizer_method, llm=llm),
        sentiment=build_sentiment_analyzer(
            "llm" if "sentiment" in llm_kinds else "lexicon", llm=llm
        ),
        entities=build_entity_extractor("llm" if "entities" in llm_kinds else "rules", llm=llm),
        question_answerer=build_question_answerer(
            "llm" if "qa" in llm_kinds else "extractive", llm=llm
        ),
        cache=AnalysisCache(enabled=False),
    )


def _cmd_analyze(args: argparse.Namespace) -> int:
    text = _read_text(args.path)
    requested = [kind.strip() for kind in args.analyses.split(",") if kind.strip()]
    engine = _build_analyzer(args)

    document = Document(
        id=Path(args.path).stem if args.path != "-" else "stdin",
        text=text,
        title=Path(args.path).stem if args.path != "-" else "",
    )
    result = engine.analyze(document, analyses=requested, question=args.question)

    if args.json:
        payload = result.model_dump(mode="json")
        payload["errors"] = engine.last_errors
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1 if engine.last_errors else 0

    print(f"{document.id}  ({result.n_chars} chars)")
    if result.summary:
        print(f"\nSUMMARY ({result.summary.method})\n{result.summary.text}")
    if result.sentiment:
        s = result.sentiment
        print(f"\nSENTIMENT ({s.method})")
        print(f"{s.label}  score {s.score:+.2f}  confidence {s.confidence:.2f}")
        if s.positive_terms or s.negative_terms:
            print(f"  positive: {', '.join(s.positive_terms) or '-'}")
            print(f"  negative: {', '.join(s.negative_terms) or '-'}")
    if result.entities:
        print(f"\nENTITIES ({result.methods.get('entities', '')})")
        for entity in result.entities:
            print(f"  {entity.type:9} {entity.text}  [{entity.start}:{entity.end}]")
    if result.qa:
        print(f"\nQA ({result.qa.method})")
        print(f"  supported: {result.qa.supported}  confidence {result.qa.confidence:.2f}")
        print(f"  {result.qa.answer}")

    timings = "  ".join(f"{stage} {ms:.1f}ms" for stage, ms in result.timings_ms.items())
    print(f"\n{timings}")

    if engine.last_errors:
        for stage, reason in engine.last_errors.items():
            print(f"error: {stage}: {reason}", file=sys.stderr)
        return 1
    return 0


def _cmd_index(args: argparse.Namespace) -> int:
    documents = [
        Document.model_validate_json(line)
        for line in Path(args.corpus).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not documents:
        raise ValueError(f"{args.corpus}: contains no documents")

    library = TfidfLibrary()
    total = library.add(documents)
    library.save(args.out)
    print(f"indexed {total} documents -> {args.out}")
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    library = TfidfLibrary.load(args.library)
    matches = library.search(args.query, args.top_k)
    if not matches:
        print("no matches")
        return 0
    for match in matches:
        print(f"[{match.rank}] {match.document_id}  ({match.score:.4f})  {match.title}")
        print(f"      {match.snippet}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    data = Path(args.data)
    summaries = load_jsonl(data / "summaries.jsonl", SummaryExample)
    sentiments = load_jsonl(data / "sentiment.jsonl", SentimentExample)
    entities = load_jsonl(data / "entities.jsonl", EntityExample)
    questions = load_jsonl(data / "qa.jsonl", QAExample)

    splits: list[tuple[str, Any]] = (
        [("dev", "dev"), ("test", "test")]
        if args.split == "both"
        else [("all", None)]
        if args.split == "all"
        else [(args.split, args.split)]
    )

    sections: list[str] = ["# Analyzer evaluation", ""]
    payload: dict[str, Any] = {}

    for label, split in splits:
        summary_rows = [
            evaluate_summarizer(LeadSummarizer(3), for_split(summaries, split)),
            evaluate_summarizer(CentroidSummarizer(3), for_split(summaries, split)),
        ]
        sentiment_rows = [evaluate_sentiment(LexiconSentiment(), for_split(sentiments, split))]
        entity_rows = [evaluate_entities(RuleEntityExtractor(), for_split(entities, split))]
        qa_rows = [evaluate_qa(ExtractiveQA(), for_split(questions, split))]

        sections.append(f"## Split: {label}")
        sections.append("")
        sections.append(summary_table(summary_rows, split=label))
        sections.append(sentiment_table(sentiment_rows, split=label))
        sections.append(confusion_table(sentiment_rows[0]))
        sections.append(entity_table(entity_rows, split=label))
        sections.append(qa_table(qa_rows, split=label))

        payload[label] = {
            "summarisation": [row.model_dump(mode="json") for row in summary_rows],
            "sentiment": [row.model_dump(mode="json") for row in sentiment_rows],
            "entities": [row.model_dump(mode="json") for row in entity_rows],
            "qa": [row.model_dump(mode="json") for row in qa_rows],
        }

    markdown = "\n".join(sections)
    print(markdown)
    if args.out:
        print(f"wrote {write_json(payload, Path(args.out) / 'metrics.json')}")
        print(f"wrote {write_text(markdown, Path(args.out) / 'metrics.md')}")
    return 0


def _cmd_repair(args: argparse.Namespace) -> int:
    cases = load_jsonl(args.cases, RepairCase)
    metrics = evaluate_repair(cases)
    markdown = repair_table(metrics)
    print(markdown)

    if args.out:
        print(f"wrote {write_json(metrics, Path(args.out) / 'repair.json')}")
        print(f"wrote {write_text(markdown, Path(args.out) / 'repair.md')}")

    # A case that did not behave as labelled means the repair pipeline changed behaviour.
    return 1 if metrics.failures else 0


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - depends on install extras
        print(f"error: uvicorn is required to serve: {exc}", file=sys.stderr)
        return 2

    from analyzer.api import create_app

    uvicorn.run(create_app(library_dir=args.library), host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    handlers = {
        "analyze": _cmd_analyze,
        "index": _cmd_index,
        "search": _cmd_search,
        "evaluate": _cmd_evaluate,
        "repair": _cmd_repair,
        "serve": _cmd_serve,
    }
    try:
        return handlers[args.command](args)
    except (
        FileNotFoundError,
        ValueError,
        AnalyzerError,
        LibraryError,
        ProviderError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
