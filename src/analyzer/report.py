"""Rendering evaluation results.

Every figure is printed with its example count. An F1 of 0.82 over 20 examples and the same
figure over 2000 are different claims, and a table that omits the count invites the reader
to assume the second.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from analyzer.types import (
    EntityMetrics,
    QAMetrics,
    RepairMetrics,
    SentimentMetrics,
    SummaryMetrics,
)


def _num(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def summary_table(rows: list[SummaryMetrics], *, split: str) -> str:
    lines = [
        f"### Summarisation ({split}, {rows[0].n_examples if rows else 0} examples)",
        "",
        "| Method | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | Mean compression |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row.method}` | {_num(row.rouge1_f1)} | {_num(row.rouge2_f1)} "
            f"| {_num(row.rougeL_f1)} | {_num(row.mean_compression)} |"
        )
    lines.append("")
    return "\n".join(lines)


def sentiment_table(rows: list[SentimentMetrics], *, split: str) -> str:
    lines = [
        f"### Sentiment ({split}, {rows[0].n_examples if rows else 0} examples)",
        "",
        "| Method | Accuracy | Macro F1 | Per-label F1 | No signal |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        per_label = ", ".join(
            f"{label} {_num(value, 2)}" for label, value in sorted(row.per_label_f1.items())
        )
        lines.append(
            f"| `{row.method}` | {_pct(row.accuracy)} | {_num(row.macro_f1)} "
            f"| {per_label or '-'} | {row.n_abstained} |"
        )
    lines.append("")
    return "\n".join(lines)


def confusion_table(metrics: SentimentMetrics) -> str:
    labels = sorted(metrics.confusion)
    lines = [
        f"#### Confusion, `{metrics.method}` (rows: gold, columns: predicted)",
        "",
        "| gold \\ predicted | " + " | ".join(labels) + " |",
        "| --- | " + " | ".join("---" for _ in labels) + " |",
    ]
    for gold in labels:
        cells = " | ".join(str(metrics.confusion[gold].get(predicted, 0)) for predicted in labels)
        lines.append(f"| **{gold}** | {cells} |")
    lines.append("")
    return "\n".join(lines)


def entity_table(rows: list[EntityMetrics], *, split: str) -> str:
    lines = [
        f"### Entity extraction ({split}, {rows[0].n_examples if rows else 0} examples, "
        f"{rows[0].n_gold if rows else 0} labelled entities)",
        "",
        "| Method | Strict P | Strict R | Strict F1 | Partial F1 | Predicted |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row.method}` | {_num(row.strict_precision)} | {_num(row.strict_recall)} "
            f"| {_num(row.strict_f1)} | {_num(row.partial_f1)} | {row.n_predicted} |"
        )
    lines.append("")
    if rows and rows[0].per_type_f1:
        lines.append(
            "Per-type strict F1: "
            + ", ".join(
                f"`{entity_type}` {_num(value, 2)}"
                for entity_type, value in sorted(rows[0].per_type_f1.items())
            )
        )
        lines.append("")
    return "\n".join(lines)


def qa_table(rows: list[QAMetrics], *, split: str) -> str:
    lines = [
        f"### Question answering ({split}, {rows[0].n_examples if rows else 0} examples)",
        "",
        "| Method | Exact match | Token F1 | False abstention | True abstention |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row.method}` | {_pct(row.exact_match)} | {_num(row.token_f1)} "
            f"| {_pct(row.false_abstention_rate)} | {_pct(row.true_abstention_rate)} |"
        )
    lines.append("")
    lines.append(
        "False abstention: answerable questions wrongly declined. True abstention: "
        "unanswerable questions correctly declined -- higher is better for both columns' "
        "intent, so read the first as an error rate and the second as a success rate."
    )
    lines.append("")
    return "\n".join(lines)


def repair_table(metrics: RepairMetrics) -> str:
    lines = [
        "### JSON repair",
        "",
        f"- Cases: {metrics.n_cases}",
        f"- Parsed with no repair: {metrics.n_parsed_directly}",
        f"- Parsed after repair: {metrics.n_parsed_after_repair}",
        f"- Unrecoverable: {metrics.n_failed}",
        f"- Recovery rate: **{_pct(metrics.recovery_rate)}**",
        "",
    ]
    if metrics.repairs_used:
        lines.append("| Repair | Times needed |")
        lines.append("| --- | --- |")
        for name, count in metrics.repairs_used.items():
            lines.append(f"| `{name}` | {count} |")
        lines.append("")
    if metrics.failures:
        lines.append("#### Cases that did not behave as labelled")
        lines.append("")
        lines.extend(f"- {failure}" for failure in metrics.failures)
        lines.append("")
    else:
        lines.append("Every case behaved as labelled.")
        lines.append("")
    return "\n".join(lines)


def write_json(payload: Any, path: str | Path) -> Path:
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    data = payload.model_dump(mode="json") if hasattr(payload, "model_dump") else payload
    file.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return file


def write_text(text: str, path: str | Path) -> Path:
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(text, encoding="utf-8")
    return file
