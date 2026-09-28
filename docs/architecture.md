# Architecture

## Module responsibilities

| Module | Owns | Deliberately does not |
| --- | --- | --- |
| `types.py` | The validated data model and its invariants | Any I/O, scoring or configuration |
| `text.py` | Normalisation, tokenisation, sentence spans, stable hashing | Know about analyses |
| `schemas.py` | LLM response schemas and JSON repair | Call a model |
| `providers.py` | Turning a prompt into text | Parse, validate or retry the content |
| `analyzers/*` | One analysis each, baseline and LLM | Time themselves, or cache |
| `library.py` | Storing and finding documents | Analyse them |
| `pipeline.py` | Orchestration, timing, failure isolation, caching | Implement any analysis |
| `evaluation.py` | Metrics and harnesses | Know how a result was produced |
| `report.py` | Rendering | Compute anything |
| `api.py` | HTTP surface and request validation | Contain analysis logic |
| `cli.py` | Argument parsing and exit codes | Contain analysis logic |
| `web/src/lib/highlight.ts` | Spans → renderable segments | Render |
| `web/src/lib/api.ts` | Transport and error mapping | Presentation |

Four boundaries carry most of the weight.

**Analyzers do not time themselves.** `ContentAnalyzer._timed` wraps each stage, so timing is
uniform and an analyzer cannot forget to do it. It is also where failure isolation lives.

**Providers do not parse.** `LLM.complete` returns text and raises on transport failure.
Returning an error string would let it be parsed as a model answer — the worst available
outcome, because a pipeline would then score a network failure as a bad answer.

**Generation does not verify itself.** `LLMEntityExtractor` drops entities absent from the
text and `LLMQA` drops unverifiable evidence, but the *grading* of those decisions lives in
`evaluation.py`. A component that graded its own output would be checking its own homework.

**The frontend's highlighting logic is a pure function.** `segmentText` takes text and spans
and returns segments. Testing it needs no DOM, and the twenty tests it has cover the cases
that matter — a repeated name, overlapping spans, a span past the end of the text.

## The analyzer protocols

Four protocols in `analyzers/base.py`, each with two implementations:

```python
class Summarizer(Protocol):
    @property
    def name(self) -> str: ...
    def summarize(self, document: Document) -> Summary: ...
```

Structural typing, not inheritance: anything with the right shape works, and the test suite
exploits that with small fake analyzers where a real one would obscure the point.

The protocols do real work in the type checker. Before they were applied, `ContentAnalyzer`
held its analyzers as `object` and every call site needed a `type: ignore[attr-defined]`.
Those suppressions were hiding the fact that nothing checked the analyzers had the right
methods at all.

## The JSON repair pipeline

```
model output
    │
    ├─ json.loads ──────────────────────────► success, repairs = ()
    │      │ fails
    │      ▼
    ├─ strip_code_fence          ```json { … } ```
    ├─ extract_balanced          prose { … } prose      ← string-aware brace matching
    ├─ normalize_quotes          “key”: “value”
    ├─ python_literals           True / False / None    ← outside string literals only
    ├─ remove_trailing_commas    { "a": 1, }            ← outside string literals only
    ├─ single_to_double_quotes   {'a': 'b'}             ← only when no " is present
    │      │ still fails
    │      ▼
    └─ ValueError("could not parse …")
           │
           ▼  on success
    pydantic model_validate → ParseOutcome(ok, value, repairs, error)
```

Each repair is applied cumulatively and only after the previous attempt failed, so
well-formed output is never modified. Two are conservative on purpose:

**`extract_balanced` is string-aware.** It tracks whether it is inside a string and whether
the previous character was an escape, so a brace inside a prose value does not change the
depth. A regex cannot do this correctly, and the cases where it fails are exactly the
interesting ones — JSON containing prose containing punctuation.

**`_outside_strings` splits before substituting.** Rewriting `True` or a trailing comma
across the whole text is the obvious implementation, and it corrupts a value containing the
word "None" or a comma before a bracket. Splitting on string boundaries first is what makes
the substitution safe.

**`single_to_double_quotes` declines when a double quote is present.** Converting then would
rewrite an apostrophe inside a real string and turn a valid document into an invalid one —
worse than the original error.

## Failure model

| Failure | Effect | Where |
| --- | --- | --- |
| Malformed evaluation record | Raise with `path:line` and the field | `evaluation.load_jsonl` |
| Gold entity absent from its text | Raise, rather than score zero | `evaluation.resolve_gold_spans` |
| LLM output unparseable | `ParseOutcome(ok=False)`, never raises | `schemas.parse_model_output` |
| LLM output fails its schema | `ParseOutcome(ok=False)` with the field named | `schemas.parse_model_output` |
| Analyzer cannot produce a result | `AnalyzerError` | `analyzers/*` |
| One stage fails | That field stays `None`, reason in `last_errors`, others still run | `pipeline._timed` |
| Provider transport failure | Raises; `RetryingLLM` retries transport only | `providers.py` |
| LLM entity not in the text | Dropped, recorded in `last_dropped` | `entities.LLMEntityExtractor` |
| LLM evidence not in the document | Dropped; answer downgraded to unsupported | `qa.LLMQA` |
| Corrupt cache entry | Treated as a miss | `pipeline.AnalysisCache.get` |
| Duplicate document id | Raise | `library.TfidfLibrary.add` |
| Request the pipeline cannot serve | HTTP 422 with the reason | `api.py` |
| Unknown document | HTTP 404 | `api.py` |
| Entity offsets disagree with the text | Warning rendered above the highlights | `EntityHighlighter` |

The rule: **configuration and data errors raise; per-item runtime failures degrade to a
recorded result.** A batch of a thousand documents is not aborted by document 42, and nothing
is silently coerced into a successful-looking analysis.

## Why a schema-failure is not retried

`RetryingLLM` retries transport failures. It does **not** retry a response that parsed but
violated its schema.

That is deliberate: a model that returns the wrong shape will usually return the wrong shape
again, so retrying burns tokens without improving the odds. More importantly, retrying
silently hides how often it happens, and the frequency is the signal that the prompt needs
changing. The repair pipeline reports what it had to fix for the same reason.

## Extension points

### A new analysis

Add a protocol to `analyzers/base.py`, a module with a baseline and an LLM implementation, a
response schema in `schemas.py`, a branch in `ContentAnalyzer.analyze`, a field on `Analysis`,
and a metric in `evaluation.py`.

The last one is not optional. An analysis with no evaluation is an analysis nobody can tell
is working.

### A new provider

Anything with `name` and `complete(prompt, *, max_tokens) -> str`. Raise on transport
failure; do not return an error string. Register it in `build_llm` only if it needs to be
reachable from the CLI.

### A new entity type

Add it to `EntityType` in `types.py`, add the pattern or gazetteer to
`analyzers/entities.py`, and add labelled examples to `data/eval/entities.jsonl` in **both**
splits. Per-type F1 will then report it, which is the only way anyone will know whether it
works.

### A frontend component

Keep the logic in `src/lib` as a pure function and the component thin. `segmentText` and
`countByType` are both testable without rendering anything, which is why they have more
tests than the component that uses them.

## Why the frontend holds `analysedText` separately

`AnalyzeForm` stores the text the current result came from, separately from the textarea
value.

Entity offsets index the text that was submitted. If the highlighter read the live textarea,
editing it after an analysis would apply offsets to text they were never computed from, and
the highlights would drift — silently, and in a way that looks like a rendering bug.
`spansAreConsistent` is the second line of defence: it checks the offsets still resolve to
the entity text and surfaces a warning if they do not.
