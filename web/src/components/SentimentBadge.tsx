import type { SentimentResult } from "@/lib/types";

const LABEL_CLASS: Record<SentimentResult["label"], string> = {
  positive: "badge badge-positive",
  negative: "badge badge-negative",
  neutral: "badge badge-neutral",
  mixed: "badge badge-mixed",
};

export interface SentimentBadgeProps {
  sentiment: SentimentResult;
}

/**
 * Shows the label, the signed score and the terms that produced it.
 *
 * The driving terms are shown because a sentiment label with no explanation is not
 * actionable: a user who disagrees with "negative" needs to see that it came from "slow"
 * and "buggy" to know whether the analyzer is wrong or they are.
 */
export function SentimentBadge({ sentiment }: SentimentBadgeProps) {
  const { label, score, confidence, positive_terms, negative_terms, negations } = sentiment;
  const noSignal = confidence === 0;

  return (
    <section aria-labelledby="sentiment-heading" className="card">
      <h2 id="sentiment-heading">Sentiment</h2>

      <p>
        <span className={LABEL_CLASS[label]} data-testid="sentiment-label">
          {label}
        </span>{" "}
        <span className="muted">
          score {score >= 0 ? "+" : ""}
          {score.toFixed(2)} · confidence {confidence.toFixed(2)}
        </span>
      </p>

      {noSignal ? (
        <p className="muted" data-testid="sentiment-no-signal">
          No sentiment-bearing words found, so this is &ldquo;no signal&rdquo; rather than a
          balanced judgement.
        </p>
      ) : (
        <dl className="terms">
          <dt>Positive</dt>
          <dd data-testid="positive-terms">
            {positive_terms.length > 0 ? positive_terms.join(", ") : "—"}
          </dd>
          <dt>Negative</dt>
          <dd data-testid="negative-terms">
            {negative_terms.length > 0 ? negative_terms.join(", ") : "—"}
          </dd>
          {negations > 0 && (
            <>
              <dt>Negations</dt>
              <dd>{negations}</dd>
            </>
          )}
        </dl>
      )}
    </section>
  );
}
