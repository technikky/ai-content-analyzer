import type { QAAnswer } from "@/lib/types";

export interface QAPanelProps {
  qa: QAAnswer;
}

/**
 * The answer, or an explicit statement that the document does not contain one.
 *
 * An unsupported answer is rendered differently rather than hidden: the near-miss evidence
 * is often what tells a user their question was close but phrased for a different document.
 */
export function QAPanel({ qa }: QAPanelProps) {
  return (
    <section aria-labelledby="qa-heading" className="card">
      <h2 id="qa-heading">Question</h2>
      <p className="muted">{qa.question}</p>

      {qa.supported ? (
        <p data-testid="qa-answer">{qa.answer}</p>
      ) : (
        <p className="warning" data-testid="qa-unsupported">
          The document does not answer this question.
        </p>
      )}

      {qa.evidence.length > 0 && (
        <>
          <h3>{qa.supported ? "Evidence" : "Closest passage"}</h3>
          <ul data-testid="qa-evidence">
            {qa.evidence.map((quote) => (
              <li key={quote}>{quote}</li>
            ))}
          </ul>
        </>
      )}

      <p className="muted">
        {qa.method} · confidence {qa.confidence.toFixed(2)}
      </p>
    </section>
  );
}
