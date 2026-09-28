import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { EntityHighlighter } from "@/components/EntityHighlighter";
import { QAPanel } from "@/components/QAPanel";
import { SentimentBadge } from "@/components/SentimentBadge";
import { SummaryCard } from "@/components/SummaryCard";
import type { Entity, QAAnswer, SentimentResult, Summary } from "@/lib/types";

function sentiment(overrides: Partial<SentimentResult> = {}): SentimentResult {
  return {
    label: "positive",
    score: 0.8,
    confidence: 0.75,
    positive_terms: ["excellent"],
    negative_terms: [],
    negations: 0,
    method: "lexicon",
    ...overrides,
  };
}

describe("SentimentBadge", () => {
  it("shows the label, score and driving terms", () => {
    render(<SentimentBadge sentiment={sentiment()} />);

    expect(screen.getByTestId("sentiment-label")).toHaveTextContent("positive");
    expect(screen.getByText(/\+0\.80/)).toBeInTheDocument();
    expect(screen.getByTestId("positive-terms")).toHaveTextContent("excellent");
  });

  it("distinguishes no signal from a balanced judgement", () => {
    render(
      <SentimentBadge
        sentiment={sentiment({ label: "neutral", score: 0, confidence: 0, positive_terms: [] })}
      />,
    );

    expect(screen.getByTestId("sentiment-no-signal")).toBeInTheDocument();
    expect(screen.queryByTestId("positive-terms")).not.toBeInTheDocument();
  });

  it("reports negations when there were any", () => {
    render(<SentimentBadge sentiment={sentiment({ label: "negative", score: -0.6, negations: 2 })} />);
    expect(screen.getByText("Negations")).toBeInTheDocument();
  });
});

describe("SummaryCard", () => {
  const summary: Summary = {
    text: "The council approved the plan.",
    sentences: ["The council approved the plan."],
    method: "centroid-3",
    compression_ratio: 0.42,
  };

  it("shows the summary, its method and its compression", () => {
    render(<SummaryCard summary={summary} />);

    expect(screen.getByTestId("summary-text")).toHaveTextContent("The council approved the plan.");
    expect(screen.getByTestId("summary-method")).toHaveTextContent("centroid-3");
    expect(screen.getByText(/42% of the source length/)).toBeInTheDocument();
  });

  it("handles an empty summary without rendering a blank card", () => {
    render(<SummaryCard summary={{ ...summary, text: "", compression_ratio: null }} />);
    expect(screen.getByTestId("summary-text")).toHaveTextContent("No summary produced.");
  });
});

describe("QAPanel", () => {
  const answer: QAAnswer = {
    question: "When does the window run?",
    answer: "Every second Sunday.",
    supported: true,
    confidence: 0.9,
    evidence: ["The window runs every second Sunday."],
    method: "extractive",
  };

  it("shows a supported answer with its evidence", () => {
    render(<QAPanel qa={answer} />);

    expect(screen.getByTestId("qa-answer")).toHaveTextContent("Every second Sunday.");
    expect(screen.getByRole("heading", { name: "Evidence" })).toBeInTheDocument();
  });

  it("states plainly when the document does not answer", () => {
    render(
      <QAPanel qa={{ ...answer, supported: false, answer: "", confidence: 0.1 }} />,
    );

    expect(screen.getByTestId("qa-unsupported")).toBeInTheDocument();
    expect(screen.queryByTestId("qa-answer")).not.toBeInTheDocument();
    // The near miss is still shown, labelled as such.
    expect(screen.getByRole("heading", { name: "Closest passage" })).toBeInTheDocument();
  });
});

describe("EntityHighlighter", () => {
  const text = "Ada Lovelace joined Acme Corp in Paris.";
  const entities: Entity[] = [
    { text: "Ada Lovelace", type: "PERSON", start: 0, end: 12, confidence: 0.9 },
    { text: "Acme Corp", type: "ORG", start: 20, end: 29, confidence: 0.9 },
    { text: "Paris", type: "LOCATION", start: 33, end: 38, confidence: 0.85 },
  ];

  it("renders the full text with each entity marked", () => {
    render(<EntityHighlighter text={text} entities={entities} />);

    expect(screen.getByTestId("highlighted-text")).toHaveTextContent(
      "Ada Lovelace joined Acme Corp in Paris.",
    );
    const marks = screen.getByTestId("highlighted-text").querySelectorAll("mark");
    expect([...marks].map((mark) => mark.textContent)).toEqual([
      "Ada Lovelace",
      "Acme Corp",
      "Paris",
    ]);
  });

  it("offers a filter chip per type, with counts", () => {
    render(<EntityHighlighter text={text} entities={entities} />);

    for (const type of ["PERSON", "ORG", "LOCATION"]) {
      expect(screen.getByRole("button", { name: new RegExp(type) })).toBeInTheDocument();
    }
  });

  it("removes a type's highlights when its chip is toggled off", async () => {
    const user = userEvent.setup();
    render(<EntityHighlighter text={text} entities={entities} />);

    await user.click(screen.getByRole("button", { name: /PERSON/ }));

    const marks = screen.getByTestId("highlighted-text").querySelectorAll("mark");
    expect([...marks].map((mark) => mark.textContent)).toEqual(["Acme Corp", "Paris"]);
    // The text itself must survive unhighlighting.
    expect(screen.getByTestId("highlighted-text")).toHaveTextContent("Ada Lovelace joined");
    expect(screen.getByRole("button", { name: /PERSON/ })).toHaveAttribute(
      "aria-pressed",
      "false",
    );
  });

  it("says so when there are no entities", () => {
    render(<EntityHighlighter text={text} entities={[]} />);
    expect(screen.getByTestId("no-entities")).toBeInTheDocument();
  });

  it("warns when offsets disagree with the text", () => {
    // Otherwise this looks like a UI bug when the cause is upstream.
    render(
      <EntityHighlighter
        text={text}
        entities={[{ text: "Grace Hopper", type: "PERSON", start: 0, end: 12, confidence: 0.9 }]}
      />,
    );

    expect(screen.getByTestId("span-mismatch")).toBeInTheDocument();
  });
});
