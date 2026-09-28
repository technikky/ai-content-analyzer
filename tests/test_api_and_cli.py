from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from analyzer.api import create_app
from analyzer.cli import main
from analyzer.library import TfidfLibrary
from analyzer.pipeline import AnalysisCache, ContentAnalyzer
from analyzer.providers import ScriptedLLM
from tests.conftest import CORPUS_FILE, EVAL_DIR, SAMPLE_TEXT

# --- API --------------------------------------------------------------------------


@pytest.fixture
def client(analyzer: ContentAnalyzer, library: TfidfLibrary) -> Iterator[TestClient]:
    with TestClient(create_app(analyzer, library)) as test_client:
        yield test_client


def test_healthz_reports_the_library_size(client: TestClient) -> None:
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["documents"] == 3


def test_stats_describes_the_configured_methods(client: TestClient) -> None:
    body = client.get("/stats").json()
    assert body["library"].startswith("tfidf")
    assert body["methods"]["sentiment"] == "lexicon"
    assert "summary" in body["analyses"]


def test_analyze_returns_every_requested_analysis(client: TestClient) -> None:
    response = client.post(
        "/analyze",
        json={
            "text": SAMPLE_TEXT,
            "title": "Update",
            "analyses": ["summary", "sentiment", "entities"],
        },
    )
    assert response.status_code == 200

    body = response.json()
    assert body["summary"]["text"]
    assert body["sentiment"]["label"] == "positive"
    assert {e["type"] for e in body["entities"]} >= {"PERSON", "ORG", "MONEY"}
    assert body["errors"] == {}
    assert body["total_ms"] >= 0.0


def test_analyze_entity_spans_index_the_submitted_text(client: TestClient) -> None:
    body = client.post("/analyze", json={"text": SAMPLE_TEXT, "analyses": ["entities"]}).json()
    for entity in body["entities"]:
        assert SAMPLE_TEXT[entity["start"] : entity["end"]] == entity["text"]


def test_analyze_runs_qa_when_given_a_question(client: TestClient) -> None:
    body = client.post(
        "/analyze",
        json={
            "text": "The maintenance window runs from 02:00 to 04:00 UTC every second Sunday.",
            "analyses": ["qa"],
            "question": "When does the maintenance window run?",
        },
    ).json()
    assert body["qa"]["supported"] is True


def test_qa_without_a_question_is_a_422(client: TestClient) -> None:
    response = client.post("/analyze", json={"text": "Some text.", "analyses": ["qa"]})
    assert response.status_code == 422


def test_analyze_can_store_the_document(client: TestClient) -> None:
    before = client.get("/healthz").json()["documents"]
    client.post(
        "/analyze",
        json={"text": SAMPLE_TEXT, "document_id": "kb/new", "analyses": ["summary"], "store": True},
    )
    assert client.get("/healthz").json()["documents"] == before + 1


def test_an_empty_text_is_rejected_by_validation(client: TestClient) -> None:
    assert client.post("/analyze", json={"text": ""}).status_code == 422


def test_an_unknown_field_is_rejected_by_validation(client: TestClient) -> None:
    assert client.post("/analyze", json={"text": "x", "analysis": ["summary"]}).status_code == 422


def test_an_unknown_analysis_is_rejected_by_validation(client: TestClient) -> None:
    assert client.post("/analyze", json={"text": "x", "analyses": ["astrology"]}).status_code == 422


def test_documents_can_be_added_and_listed(client: TestClient) -> None:
    response = client.post(
        "/documents",
        json={
            "documents": [
                {"id": "kb/extra", "text": "Exports are available as CSV.", "title": "Exports"}
            ]
        },
    )
    assert response.status_code == 201
    assert response.json()["documents"] == 4

    listed = client.get("/documents").json()
    assert any(item["id"] == "kb/extra" for item in listed)


def test_adding_a_duplicate_document_is_a_409(client: TestClient) -> None:
    payload = {"documents": [{"id": "kb/refunds", "text": "Something else."}]}
    assert client.post("/documents", json=payload).status_code == 409


def test_a_document_can_be_fetched_by_id(client: TestClient) -> None:
    body = client.get("/documents/kb/refunds").json()
    assert body["id"] == "kb/refunds"
    assert "Refunds" in body["text"]


def test_an_unknown_document_is_a_404(client: TestClient) -> None:
    assert client.get("/documents/kb/nope").status_code == 404


def test_search_ranks_the_right_document_first(client: TestClient) -> None:
    body = client.post("/search", json={"query": "how long do refunds take", "top_k": 2}).json()
    assert body[0]["document_id"] == "kb/refunds"
    assert body[0]["rank"] == 1


def test_asking_a_stored_document_a_question(client: TestClient) -> None:
    body = client.post(
        "/documents/kb/tokens/qa", json={"question": "How long do access tokens last?"}
    ).json()
    assert body["supported"] is True
    assert "one hour" in body["answer"]


def test_asking_an_unknown_document_is_a_404(client: TestClient) -> None:
    response = client.post("/documents/kb/nope/qa", json={"question": "anything?"})
    assert response.status_code == 404


def test_a_failing_stage_is_reported_in_errors(library: TfidfLibrary) -> None:
    """A 200 with an errors map, because the other analyses succeeded and are worth returning."""
    from analyzer.analyzers.summarize import LLMSummarizer

    broken = ContentAnalyzer(
        summarizer=LLMSummarizer(ScriptedLLM(["not json"])), cache=AnalysisCache(enabled=False)
    )
    with TestClient(create_app(broken, library)) as client:
        body = client.post(
            "/analyze", json={"text": SAMPLE_TEXT, "analyses": ["summary", "sentiment"]}
        ).json()

    assert body["summary"] is None
    assert body["sentiment"] is not None
    assert "summary" in body["errors"]


def test_the_openapi_schema_is_served(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/analyze", "/search", "/documents", "/healthz"} <= set(paths)


def test_the_app_can_load_a_library_from_disk(library: TfidfLibrary, tmp_path: Path) -> None:
    library.save(tmp_path / "lib")
    with TestClient(create_app(library_dir=tmp_path / "lib")) as client:
        assert client.get("/healthz").json()["documents"] == 3


# --- CLI --------------------------------------------------------------------------


def test_analyze_reads_a_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "input.txt"
    path.write_text(SAMPLE_TEXT, encoding="utf-8")

    assert main(["analyze", str(path)]) == 0
    out = capsys.readouterr().out
    assert "SUMMARY" in out
    assert "SENTIMENT" in out
    assert "ENTITIES" in out


def test_analyze_can_emit_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "input.txt"
    path.write_text(SAMPLE_TEXT, encoding="utf-8")

    assert main(["analyze", str(path), "--json", "--analyses", "sentiment"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["sentiment"]["label"] == "positive"


def test_analyze_honours_the_summariser_choice(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "input.txt"
    path.write_text(SAMPLE_TEXT, encoding="utf-8")

    assert main(["analyze", str(path), "--summarizer", "centroid", "--analyses", "summary"]) == 0
    assert "centroid" in capsys.readouterr().out


def test_analyze_on_a_missing_file_exits_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["analyze", str(tmp_path / "nope.txt")]) == 2
    assert "not found" in capsys.readouterr().err


def test_an_llm_analysis_without_a_provider_exits_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "input.txt"
    path.write_text(SAMPLE_TEXT, encoding="utf-8")

    assert main(["analyze", str(path), "--llm-analyses", "summary"]) == 2
    assert "--provider is required" in capsys.readouterr().err


def test_index_and_search_round_trip(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    library_dir = tmp_path / "lib"
    assert main(["index", str(CORPUS_FILE), "--out", str(library_dir)]) == 0
    capsys.readouterr()

    assert main(["search", "how long are logs retained", "--library", str(library_dir)]) == 0
    assert "kb/data-retention" in capsys.readouterr().out


def test_evaluate_writes_its_artifacts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out_dir = tmp_path / "reports"
    assert main(["evaluate", "--data", str(EVAL_DIR), "--out", str(out_dir)]) == 0

    out = capsys.readouterr().out
    assert "Split: dev" in out
    assert "Split: test" in out
    assert (out_dir / "metrics.json").is_file()

    payload = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert set(payload) == {"dev", "test"}
    assert payload["test"]["sentiment"][0]["accuracy"] > 0.8


def test_evaluate_can_report_a_single_split(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["evaluate", "--data", str(EVAL_DIR), "--split", "test"]) == 0
    out = capsys.readouterr().out
    assert "Split: test" in out
    assert "Split: dev" not in out


def test_repair_passes_on_the_shipped_cases(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["repair", "--cases", str(EVAL_DIR / "repair_cases.jsonl")]) == 0
    out = capsys.readouterr().out
    assert "Recovery rate" in out
    assert "Every case behaved as labelled." in out


def test_no_subcommand_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2
