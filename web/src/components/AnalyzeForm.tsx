"use client";

import { useState } from "react";

import { EntityHighlighter } from "@/components/EntityHighlighter";
import { QAPanel } from "@/components/QAPanel";
import { SentimentBadge } from "@/components/SentimentBadge";
import { SummaryCard } from "@/components/SummaryCard";
import { ApiError, analyze } from "@/lib/api";
import { ALL_ANALYSES, type AnalysisKind, type AnalyzeResponse } from "@/lib/types";

const SAMPLE =
  "Rania Haddad joined Northwind Systems Ltd in Penang on 14 March 2024. " +
  "The onboarding was excellent and the documentation is genuinely helpful, " +
  "though the export is still broken. Revenue rose 18% to $4.2 million in Q1 2026. " +
  "Questions go to rania@northwind.example.";

export function AnalyzeForm() {
  const [text, setText] = useState(SAMPLE);
  const [question, setQuestion] = useState("");
  const [selected, setSelected] = useState<ReadonlySet<AnalysisKind>>(
    new Set<AnalysisKind>(["summary", "sentiment", "entities"]),
  );
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  // The text the displayed result came from. Held separately from the textarea so that
  // entity offsets are always applied to the text they were computed against.
  const [analysedText, setAnalysedText] = useState("");

  function toggle(kind: AnalysisKind) {
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(kind)) {
        next.delete(kind);
      } else {
        next.add(kind);
      }
      return next;
    });
  }

  async function onSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    setPending(true);

    const analyses = ALL_ANALYSES.filter((kind) => selected.has(kind));
    const submitted = text;

    try {
      const response = await analyze({
        text: submitted,
        ...(question.trim() ? { question: question.trim() } : {}),
        analyses: [...analyses],
      });
      setResult(response);
      setAnalysedText(submitted);
    } catch (cause) {
      setResult(null);
      setError(
        cause instanceof ApiError ? cause.message : `unexpected failure: ${String(cause)}`,
      );
    } finally {
      setPending(false);
    }
  }

  const qaSelected = selected.has("qa");

  return (
    <div className="stack">
      <form onSubmit={onSubmit} className="card">
        <label htmlFor="text">Text to analyse</label>
        <textarea
          id="text"
          name="text"
          rows={8}
          value={text}
          onChange={(event) => setText(event.target.value)}
          required
        />

        <fieldset>
          <legend>Analyses</legend>
          {ALL_ANALYSES.map((kind) => (
            <label key={kind} className="inline">
              <input
                type="checkbox"
                name="analyses"
                value={kind}
                checked={selected.has(kind)}
                onChange={() => toggle(kind)}
              />
              {kind}
            </label>
          ))}
        </fieldset>

        {qaSelected && (
          <>
            <label htmlFor="question">Question</label>
            <input
              id="question"
              name="question"
              type="text"
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="What does the document say about the export?"
              required
            />
          </>
        )}

        <button type="submit" disabled={pending || selected.size === 0}>
          {pending ? "Analysing" : "Analyse"}
        </button>
      </form>

      {error && (
        <p className="warning" role="alert" data-testid="error">
          {error}
        </p>
      )}

      {result && (
        <div className="stack">
          {Object.keys(result.errors).length > 0 && (
            <div className="warning" role="alert" data-testid="stage-errors">
              <p>Some analyses failed; the rest are shown below.</p>
              <ul>
                {Object.entries(result.errors).map(([stage, reason]) => (
                  <li key={stage}>
                    <strong>{stage}</strong>: {reason}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {result.summary && <SummaryCard summary={result.summary} />}
          {result.sentiment && <SentimentBadge sentiment={result.sentiment} />}
          {selected.has("entities") && (
            <EntityHighlighter text={analysedText} entities={result.entities} />
          )}
          {result.qa && <QAPanel qa={result.qa} />}

          <p className="muted" data-testid="timings">
            {Object.entries(result.timings_ms)
              .map(([stage, ms]) => `${stage} ${ms.toFixed(1)}ms`)
              .join(" | ")}
            {result.cached ? " | cached" : ""}
          </p>
        </div>
      )}
    </div>
  );
}
