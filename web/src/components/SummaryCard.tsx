import type { Summary } from "@/lib/types";

export interface SummaryCardProps {
  summary: Summary;
}

/** The summary, with the method that produced it and how much it compressed the source. */
export function SummaryCard({ summary }: SummaryCardProps) {
  const compression =
    summary.compression_ratio === null
      ? null
      : `${Math.round(summary.compression_ratio * 100)}% of the source length`;

  return (
    <section aria-labelledby="summary-heading" className="card">
      <h2 id="summary-heading">Summary</h2>
      <p data-testid="summary-text">{summary.text || "No summary produced."}</p>
      <p className="muted">
        <span data-testid="summary-method">{summary.method}</span>
        {compression ? ` · ${compression}` : ""}
      </p>
    </section>
  );
}
