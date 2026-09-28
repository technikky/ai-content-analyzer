/**
 * Types mirroring the FastAPI response schemas.
 *
 * Hand-written rather than generated, deliberately: generating from the OpenAPI document
 * would add a build step and a code-generator dependency for a surface this small, and the
 * API contract is covered by tests on both sides. If the surface grows, generate it.
 */

export type SentimentLabel = "positive" | "negative" | "neutral" | "mixed";

export type EntityType =
  | "PERSON"
  | "ORG"
  | "LOCATION"
  | "DATE"
  | "MONEY"
  | "PERCENT"
  | "EMAIL"
  | "URL";

export type AnalysisKind = "summary" | "sentiment" | "entities" | "qa";

export interface Summary {
  text: string;
  sentences: string[];
  method: string;
  compression_ratio: number | null;
}

export interface SentimentResult {
  label: SentimentLabel;
  score: number;
  confidence: number;
  positive_terms: string[];
  negative_terms: string[];
  negations: number;
  method: string;
}

/** An entity and the half-open character span it occupies in the submitted text. */
export interface Entity {
  text: string;
  type: EntityType;
  start: number;
  end: number;
  confidence: number;
}

export interface QAAnswer {
  question: string;
  answer: string;
  supported: boolean;
  confidence: number;
  evidence: string[];
  method: string;
}

export interface AnalyzeResponse {
  document_id: string;
  title: string;
  n_chars: number;
  summary: Summary | null;
  sentiment: SentimentResult | null;
  entities: Entity[];
  qa: QAAnswer | null;
  timings_ms: Record<string, number>;
  methods: Record<string, string>;
  total_ms: number;
  cached: boolean;
  /** Stage name to reason, for any analyzer that failed. Empty when all succeeded. */
  errors: Record<string, string>;
}

export interface AnalyzeRequest {
  text: string;
  title?: string;
  document_id?: string;
  analyses?: AnalysisKind[];
  question?: string;
  store?: boolean;
}

export interface SearchMatch {
  document_id: string;
  title: string;
  score: number;
  snippet: string;
  rank: number;
}

export interface HealthResponse {
  status: string;
  documents: number;
}

export const ENTITY_TYPES: readonly EntityType[] = [
  "PERSON",
  "ORG",
  "LOCATION",
  "DATE",
  "MONEY",
  "PERCENT",
  "EMAIL",
  "URL",
];

export const ALL_ANALYSES: readonly AnalysisKind[] = [
  "summary",
  "sentiment",
  "entities",
  "qa",
];
