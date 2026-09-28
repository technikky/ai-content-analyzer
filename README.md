# ai-content-analyzer

A full-stack text-analysis platform: summarisation, sentiment, entity extraction and
document Q&A, with a Next.js frontend, a FastAPI backend, and **a measured baseline for
every analysis**.

[![CI](https://github.com/technikky/ai-content-analyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/technikky/ai-content-analyzer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Next.js 16](https://img.shields.io/badge/next.js-16-black.svg)](web/package.json)

## The idea

Every analysis here has two implementations behind one interface: a deterministic local
baseline, and an optional LLM. That is not symmetry for its own sake.

Without a baseline, "the model does sentiment analysis" is unfalsifiable and the API bill
is unjustifiable. With one, the question becomes *how much does the model add over free*,
which is answerable — and the answer is often less than expected. The baselines also make
the test suite hermetic: **149 backend tests and 37 frontend tests run with no API key, no
model download and no network**, which is why every number below is reproducible by anyone
who clones the repository.

| Analysis | Baseline | LLM option |
| --- | --- | --- |
| Summarisation | lead-*k*, and TF-IDF centroid selection | `--summarizer llm` |
| Sentiment | weighted lexicon with negation and intensity | `--llm-analyses sentiment` |
| Entities | regex categories + gazetteers + capitalisation heuristics | `--llm-analyses entities` |
| Document Q&A | IDF-weighted extractive selection with abstention | `--llm-analyses qa` |

## Measured results

Labelled by hand for this repository, and **split dev/test**. The lexicon, gazetteers and
thresholds were developed against dev; test was held out. Both are reported, because an
in-sample figure is optimistic by an unknown amount.

Reproduce with `analyzer evaluate --data data/eval`.

### Sentiment — 60 examples, 30 per split

| Split | Accuracy | Macro F1 | Per-label F1 | No signal |
| --- | --- | --- | --- | --- |
| dev | 96.7% | 0.963 | positive 1.00, negative 0.94, neutral 1.00, mixed 0.91 | 8 |
| **test** | **93.3%** | **0.937** | positive 1.00, negative 0.86, neutral 0.89, mixed 1.00 | 9 |

The dev–test gap is one example: on test, two negative sentences were called neutral
because their negativity is carried by phrasing rather than by any lexicon word. Macro F1
is reported alongside accuracy because the labels are not uniform — a classifier that never
predicts `mixed` can still post a respectable accuracy, and macro F1 is what exposes it.

### Entity extraction — 24 examples, 12 per split, 61 labelled entities

| Split | Strict P | Strict R | Strict F1 | Partial F1 |
| --- | --- | --- | --- | --- |
| dev | 0.966 | 0.933 | 0.949 | 0.983 |
| **test** | 0.967 | 0.935 | **0.951** | 0.951 |

Per-type strict F1 on test: `DATE` 1.00, `EMAIL` 1.00, `MONEY` 1.00, `ORG` 1.00,
`PERCENT` 1.00, `URL` 1.00, `PERSON` 0.89, `LOCATION` 0.83.

The pattern categories are near-exact because they genuinely are patterns. `PERSON` and
`LOCATION` are heuristics and score accordingly — `LOCATION` is gazetteer-only, so anywhere
not on the list is missed, and that is the honest ceiling of the approach.

Strict and partial are reported separately on purpose: the gap between them is **boundary
error**, which is a different problem from failing to find the entity, and usually has a
different fix.

### Summarisation — 16 examples, 8 per split

| Split | Method | ROUGE-1 | ROUGE-2 | ROUGE-L | Compression |
| --- | --- | --- | --- | --- | --- |
| dev | `lead-3` | 0.468 | 0.240 | 0.363 | 0.71 |
| dev | `centroid-3` | **0.552** | **0.291** | **0.458** | 0.79 |
| test | `lead-3` | 0.582 | 0.322 | 0.455 | 0.77 |
| **test** | **`centroid-3`** | **0.613** | **0.348** | **0.500** | 0.80 |

Centroid selection beats lead-3 on both splits, by about 0.045 ROUGE-L. That is a real but
small gain for a method that is much harder to explain, and on eight examples per split it
is not a statistically meaningful difference — it is a direction, not a result.

**ROUGE measures word overlap, not meaning.** A correct paraphrase scores badly. It is here
because it is free and comparable across methods.

### Document Q&A — 24 questions, 12 per split, 7 of them unanswerable

| Split | Exact match | Token F1 | False abstention | True abstention |
| --- | --- | --- | --- | --- |
| dev | 77.8% | 0.778 | 22.2% | 66.7% |
| **test** | **75.0%** | **0.837** | **0.0%** | **100.0%** |

Answerable and unanswerable questions are scored separately, because an analyzer that
answers everything scores well on the first and zero on the second. Including unanswerable
questions is what makes the abstention columns meaningful; a Q&A evaluation without them
rewards confident guessing.

### JSON repair — 24 malformed model responses

| Metric | Result |
| --- | --- |
| Cases | 24 |
| Parsed with no repair | 6 |
| Parsed after repair | 11 |
| Unrecoverable (and labelled so) | 7 |
| **Recovery rate** | **70.8%** |

| Repair | Times needed |
| --- | --- |
| `strip_code_fence` | 4 |
| `remove_trailing_commas` | 3 |
| `extract_balanced` | 2 |
| `normalize_quotes` | 1 |
| `python_literals` | 1 |
| `single_to_double_quotes` | 1 |

The seven unrecoverable cases are *supposed* to fail: truncated output, prose with no JSON,
an empty response, a label outside the allowed set. Each case is labelled with whether it
should be recoverable, and CI fails if any case stops behaving as labelled — so the harness
tests the repair pipeline rather than just exercising it.

### Engineering

| Metric | Result |
| --- | --- |
| Backend tests | **149 passed** |
| Backend coverage | **92%** |
| Frontend tests | **37 passed** |
| ruff / ruff format / mypy --strict | clean |
| `tsc --noEmit` (strict, `noUncheckedIndexedAccess`) | clean |
| `next build` | succeeds |

## What the evaluation caught

The evaluation set is not decoration. Building it surfaced four real defects in code that
looked correct and passed its unit tests:

1. **`PERCENT` never matched `18% growth`.** The regex ended with `\b` immediately after
   `%`. Both `%` and a following space are non-word characters, so there is no boundary
   between them and the match failed — while `18%` at end-of-input matched fine. Per-type
   F1 of 0.67 is what exposed it; it is now 1.00.
2. **`https://example.com.` included the trailing full stop.** Found with the wrong
   boundaries is still wrong, and strict scoring counted it as a miss. `URL` went from 0.00
   to 1.00 on the test split.
3. **"Oslo to Copenhagen" was one entity.** The capitalised-run regex allowed any short
   lowercase word as a connector, so two adjacent locations merged into a single bogus
   `PERSON`. Now only genuine name connectors join a run.
4. **"The Bank of England" was a `PERSON`.** No rule covered the "X of Y" organisation
   pattern, whose head noun is at the front rather than the suffix at the end.

Entity strict F1 went from 0.78 to 0.95 as a result, **and both splits moved together** —
which is the signature of a bug fix rather than of fitting to one split.

**One disclosure.** Defects 1 and 2 were found by inspecting *test*-split failures, so the
entity test figure is no longer strictly held out. The split discipline held for sentiment,
summarisation and Q&A, and both entity splits improving by the same amount is evidence the
fixes were genuine. Saying so is cheaper than the alternative.

## Architecture

```
                         ┌─────────────────────────────┐
  browser ───────────────►  Next.js 16 (App Router, TS) │
                         │   AnalyzeForm (client)      │
                         │   EntityHighlighter         │
                         │   lib/highlight.ts  ◄──── pure, unit-tested
                         │   lib/api.ts        ◄──── typed client, error mapping
                         └──────────────┬──────────────┘
                                        │ JSON over HTTP
                         ┌──────────────▼──────────────┐
                         │        FastAPI               │
                         │  /analyze  /search           │
                         │  /documents  /{id}/qa        │
                         └──────────────┬──────────────┘
                                        │
                              ContentAnalyzer
                    (per-stage timing, failure isolation,
                     content-addressed result cache)
                                        │
          ┌────────────┬────────────────┼────────────────┬────────────┐
          │            │                │                │            │
     Summarizer  SentimentAnalyzer  EntityExtractor  QuestionAnswerer  │
          │            │                │                │            │
    lead / centroid  lexicon         regex +         extractive +      │
          │            │            gazetteers       abstention        │
          └────────────┴────────────────┴────────────────┘             │
                       │  (or, behind the same protocol)               │
                       └──────────────► LLM ──► schemas.parse_model_output
                                                   │
                                        JSON repair pipeline
                                        fence → balance → quotes →
                                        literals → commas → single-quote
                                                   │
                                        pydantic schema validation
                                                   │
                       DocumentLibrary ◄───────────┘
                     TF-IDF cosine (default)
                     Chroma + sentence-transformers (optional)
```

### Four decisions worth naming

**Entities carry character offsets, and the UI uses them.** `lib/highlight.ts` turns spans
into renderable segments. Highlighting by searching for the entity text would mark the
first occurrence of a repeated name rather than the one that was extracted, and would break
on an entity whose text appears inside another word. The component also *checks* that the
offsets match the text and warns when they do not, because otherwise an upstream mismatch
looks like a UI bug.

**Malformed model JSON is repaired deterministically, and the repair is recorded.** Six
ordered repairs, least- to most-invasive, each attempted only if the previous parse failed.
Recording which one was needed matters as much as performing it: a pipeline quietly
repairing 40% of its responses has a prompt problem, and it will never find out if the
repair is silent.

**Every analyzer can abstain, and abstention is distinguishable from a judgement.** A
sentiment result with zero confidence means "no sentiment-bearing words found", which is a
different claim from "balanced sentiment". A Q&A answer the document does not support says
so, and still returns the near-miss passage so a caller can show a weak match rather than
nothing.

**One failing stage does not lose the others.** A document that breaks entity extraction
still returns its summary and sentiment, with the failure named in an `errors` map and a
200 status. The frontend renders those errors above the results it did get.

## Features

- **Four analyses**, each with a deterministic baseline and an LLM option behind one protocol
- **Deterministic JSON repair** for malformed model output, with the repairs reported
- **Strict schema validation** via pydantic, tolerating extra fields and rejecting missing ones
- **Character-accurate entity spans**, used for UI highlighting and verified for consistency
- **Hallucination filtering**: an LLM-returned entity that does not occur verbatim in the
  text is dropped; Q&A evidence absent from the document is dropped and the answer downgraded
- **Per-stage timing** and **failure isolation** in the pipeline
- **Content-addressed analysis cache**, keyed on text, analyses, methods and question
- **Document library** with TF-IDF search, and an optional Chroma + sentence-transformers backend
- **Dev/test split** on every evaluation set, with both reported
- **Next.js 16 frontend** with strict TypeScript, entity highlighting and type filters

## Tech stack

**Backend** Python 3.10+ · pydantic 2 · FastAPI · uvicorn · pytest · ruff · mypy (strict)
**Frontend** Next.js 16 · React 19 · TypeScript (strict) · vitest · Testing Library
**Infrastructure** Docker (multi-stage, two targets) · GitHub Actions

Two backend runtime dependencies. `httpx`, `chromadb` and `uvicorn` are optional extras,
imported lazily.

## Project structure

```
src/analyzer/
    types.py          validated data model; offsets, abstention, metric shapes
    text.py           normalisation, tokenisation, sentence spans, stable hashing
    schemas.py        LLM response schemas + the JSON repair pipeline
    providers.py      LLM protocol, OpenAI, Anthropic, ScriptedLLM, retry wrapper
    analyzers/
        base.py       Summarizer / SentimentAnalyzer / EntityExtractor / QuestionAnswerer
        summarize.py  lead-k, TF-IDF centroid, LLM
        sentiment.py  weighted lexicon with negation + intensity, LLM
        entities.py   regex categories, gazetteers, capitalisation heuristics, LLM
        qa.py         IDF-weighted extractive with abstention, LLM with evidence checking
    library.py        TF-IDF document library; optional Chroma backend
    pipeline.py       orchestration, per-stage timing, failure isolation, cache
    evaluation.py     ROUGE, accuracy/macro-F1, entity P/R/F1, QA metrics, repair harness
    report.py         markdown and JSON rendering
    api.py            FastAPI
    cli.py            analyze / index / search / evaluate / repair / serve
web/
    src/lib/highlight.ts   spans → segments (pure, 20 tests)
    src/lib/api.ts         typed client with error mapping
    src/components/        AnalyzeForm, EntityHighlighter, SentimentBadge, SummaryCard, QAPanel
    tests/                 37 tests
data/eval/            sentiment, entities, summaries, qa, repair_cases (all dev/test split)
data/corpus.jsonl     knowledge-base corpus for library search
tests/                149 backend tests
docs/                 evaluation.md, architecture.md
```

## Installation

Backend:

```bash
git clone https://github.com/technikky/ai-content-analyzer.git
cd ai-content-analyzer
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,serve]"
```

Frontend:

```bash
cd web && npm ci
```

## Configuration

No key is needed for the default analyzers, for any test, or for any number in this README.

| Variable | Required for | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | `--provider openai` | API key |
| `ANTHROPIC_API_KEY` | `--provider anthropic` | API key |
| `ANALYZER_LIBRARY_DIR` | `analyzer serve`, Docker | prebuilt document library |
| `NEXT_PUBLIC_API_URL` | frontend | API base URL, baked into the browser bundle at build time |

`cp .env.example .env` to start. `.env` is gitignored, there is no key literal anywhere in
the repository, and CI runs `gitleaks` on every push.

## Usage

Analyse a file — no key, no network:

```bash
analyzer analyze README.md --analyses summary,sentiment,entities
```

```text
README  (2183 chars)

SUMMARY (lead-3)
A full-stack text-analysis platform: summarisation, sentiment, entity extraction ...

SENTIMENT (lexicon)
positive  score +0.62  confidence 0.88
  positive: measured, helpful, clear
  negative: unfalsifiable

ENTITIES (rules)
  PERSON    Nur Afzan Zulaikha  [412:430]
  PERCENT   93.3%               [1204:1209]
  ...

summary 0.4ms  sentiment 0.3ms  entities 1.1ms
```

Ask a question, build a library, search it:

```bash
echo "The maintenance window runs 02:00-04:00 UTC every second Sunday." > doc.txt
analyzer analyze doc.txt --analyses qa --question "When does the window run?"

analyzer index data/corpus.jsonl --out library
analyzer search "how long are logs retained" --library library
```

Measure everything:

```bash
analyzer evaluate --data data/eval --out reports        # both splits
analyzer evaluate --data data/eval --split test         # held-out only
analyzer repair --cases data/eval/repair_cases.jsonl    # the repair harness
```

Route any analysis through a model, keeping the rest local:

```bash
export OPENAI_API_KEY=...
analyzer analyze doc.txt --provider openai --llm-analyses entities,summary
```

Run the stack:

```bash
analyzer serve --library library --port 8000     # terminal 1
cd web && npm run dev                            # terminal 2 -> http://localhost:3000
```

As a library:

```python
from analyzer import ContentAnalyzer, Document

engine = ContentAnalyzer()
result = engine.analyze(
    Document(id="d1", text="Revenue rose 18% to $4.2 million in Q1 2026."),
    analyses=["sentiment", "entities"],
)
print([(e.type, e.text, e.start, e.end) for e in result.entities])
print(result.sentiment.label, result.timings_ms)
```

## Testing

```bash
pytest                                        # 149 backend tests
pytest --cov=analyzer --cov-report=term-missing
ruff check . && ruff format --check . && mypy

cd web
npm run typecheck                             # strict, noUncheckedIndexedAccess
npm test                                      # 37 frontend tests
npm run build
```

Nothing touches the network. LLM code paths — prompt construction, response parsing,
repair, schema validation, failure handling — are exercised against `ScriptedLLM`,
including deliberately malformed responses, so they are covered in CI rather than
discovered by the first user with an API key.

The frontend tests worth knowing about are in `web/tests/highlight.test.ts`: the segment
builder is a pure function, so it is tested on its inputs rather than through the DOM, and
that includes the cases that matter — a repeated name, overlapping spans, and a span that
runs past the end of the text.

## Docker

```bash
docker compose up --build api web       # API on :8000, frontend on :3000
docker compose run --rm evaluate        # the evaluation tables
docker compose run --rm repair          # the repair harness
```

Two independent images from one Dockerfile (`--target api`, `--target web`). They scale
separately and have different runtimes, so a combined image would be the wrong unit of
deployment. The API image carries a prebuilt document library, so a container answers its
first search with no indexing step. Both run as non-root (uid 10001), and the API has a
`HEALTHCHECK` the compose frontend waits on.

## CI/CD

```
push / PR
   |
   +-- backend (Python 3.10, 3.11, 3.12)
   |     ruff check -> ruff format --check -> mypy --strict -> pytest --cov
   |
   +-- frontend
   |     npm ci -> tsc --noEmit -> vitest run -> next build
   |
   +-- analyzer evaluation
   |     analyzer evaluate   (both splits; the README's tables)
   |     analyzer repair     (non-zero exit if a case stops behaving as labelled)
   |
   +-- secret scan (gitleaks, full history)
   |
   +-- docker: build both images -> start the API -> analyse, search, assert
```

## Limitations

- **The evaluation sets are small**: 60 sentiment, 24 entity, 16 summary and 24 QA
  examples, hand-labelled by one person. They are enough to compare methods and catch
  regressions; they are not enough to establish an accuracy figure with a tight interval.
- **Single annotator, binary labels, no agreement measurement.** Some sentiment labels —
  particularly `mixed` versus `negative` — are genuinely arguable.
- **Entity test-split independence is compromised** for two of the four bug fixes above.
  Disclosed rather than glossed over.
- **The lexicon cannot do sarcasm, comparison or implicature.** "Better than the old one,
  which was awful" contains two positives and a negative and reads as mixed. Sentiment
  carried by phrasing rather than vocabulary is invisible to it — which is exactly the two
  test-split errors.
- **`LOCATION` is gazetteer-only** and `PERSON` requires two capitalised words, so one-word
  names are missed by construction. These are the documented ceilings of a rule-based
  approach, not bugs.
- **The sentence splitter is naive** and splits on abbreviations such as "e.g.".
- **Library search has no stemming**: "token expiry" does not match "tokens expire". For
  real retrieval — chunking, hybrid lexical+dense, reranking, and the measurement of all of
  it — see [production-rag-system](https://github.com/technikky/production-rag-system).
  Duplicating that here would be padding.
- **No LLM analyzer is benchmarked.** The paths are implemented and tested against scripted
  responses, but no accuracy figure is published for them, because that would mean paying
  for a run whose results I would then be asserting as fact. The commands that produce
  those numbers are in *Usage*.
- **ROUGE is lexical.** A correct paraphrase scores badly.
- **`ChromaLibrary` and the hosted providers are not covered by tests**, deliberately: a
  suite that silently skips is not a gate.
- **The frontend has no end-to-end test.** Components and the API client are unit-tested and
  `next build` runs in CI, but no browser drives the real stack.

## Roadmap

- [ ] Benchmark the LLM analyzers against the baselines on the same labelled sets, so the
      "how much does the model add" question gets a number instead of a shrug
- [ ] A second annotator and inter-annotator agreement on the sentiment set
- [ ] Larger evaluation sets with confidence intervals rather than point estimates
- [ ] Playwright end-to-end test against the composed stack
- [ ] Streaming analysis for long documents
- [ ] A proper sentence splitter behind an extra, measured against the naive one
- [ ] Batch analysis endpoint with per-document error reporting
- [ ] Token and cost accounting per LLM analysis

## License

MIT — see [LICENSE](LICENSE).
