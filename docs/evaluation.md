# Evaluation

How each analyzer is measured, why the conventions are what they are, and what the numbers
do and do not support.

## The dev/test split

Every evaluation set carries a `split` field. The lexicon, the gazetteers, the thresholds
and the regexes were developed while looking at **dev**; **test** was held out.

Both are reported, always. An in-sample figure is optimistic by an amount nobody can
estimate from the figure itself, and quoting only the pooled number would overstate every
baseline here. The difference between a measurement and a demonstration is whether anything
was held back.

Observed gaps:

| Analyzer | dev | test | Gap |
| --- | --- | --- | --- |
| Sentiment accuracy | 96.7% | 93.3% | −3.4pp |
| Entity strict F1 | 0.949 | 0.951 | +0.002 |
| Summary ROUGE-L (centroid) | 0.458 | 0.500 | +0.042 |
| QA token F1 | 0.778 | 0.837 | +0.059 |

Sentiment shows the expected direction. The others go the *other* way, which on 8–12
examples per split is noise rather than evidence of anything — and saying so is more useful
than pretending the ordering means something.

### Where the split discipline was broken

Two of the four entity bugs described in the README were found by inspecting **test**-split
failures: `PERCENT` scoring 0.67 and `URL` scoring 0.00 were both visible only there.

So the entity test figure is not strictly held out any more. Two things make it still worth
reporting: both splits improved by almost exactly the same amount (0.78 → 0.95 and
0.78 → 0.95), which is what a bug fix looks like and not what fitting to a split looks
like; and the fixes are mechanical corrections to regexes that were simply wrong, not
threshold tuning.

The alternative — quietly reporting 0.95 as held-out — would have been easier and worse.

## Sentiment

**Accuracy and macro F1, with a confusion matrix.**

Macro F1 alongside accuracy because the label distribution is not uniform (16 positive, 16
negative, 16 neutral, 12 mixed). A classifier that never predicts `mixed` can still post a
respectable accuracy; macro F1 is what exposes it.

Abstention is counted separately: a result with `confidence == 0` means no
sentiment-bearing word was found at all, which is a different state from a balanced
judgement and is worth knowing the rate of.

The two test-split errors are both negative sentences called neutral:

- *"The support reply was slow and did not address the problem."* — `slow` and `problem` are
  both in the lexicon, but `did not address` negates nothing the lexicon can see, and the
  sentence's negativity is mostly carried by the phrasing.
- A similar case where negativity is implied rather than stated.

That is the documented ceiling of a lexicon: it cannot do sarcasm, comparison, or sentiment
expressed without a sentiment word. The confusion matrix is printed so the failure mode is
visible rather than averaged away.

## Entity extraction

**Strict and partial precision/recall/F1, plus per-type F1.**

- **Strict** requires the span *and* the type to match exactly.
- **Partial** requires overlapping spans of the same type.

The gap between them is **boundary error**. Reporting only one conflates two different
problems: not finding an entity, and finding it with the wrong edges. They usually have
different fixes — the `URL` full-stop bug was pure boundary error, invisible to a partial
score and glaring in a strict one.

Gold entities are labelled as **text, not offsets**, and spans are derived by locating each
one in the example (first unclaimed occurrence winning). Hand-written offsets in a JSONL
file are unmaintainable: one edit to the text invalidates every span after it. A label that
does not occur in its text raises rather than silently scoring zero.

Per-type F1 on the test split:

| Type | Strict F1 | Why |
| --- | --- | --- |
| `DATE`, `EMAIL`, `MONEY`, `ORG`, `PERCENT`, `URL` | 1.00 | Genuinely patterns, so a regex is close to exact |
| `PERSON` | 0.89 | Heuristic: two or more capitalised words, no function words |
| `LOCATION` | 0.83 | Gazetteer-only, so anywhere not on the list is missed |

`PERSON` deliberately never matches a single capitalised word. Otherwise every sentence
opening becomes a candidate, and precision collapses. The cost is that one-word names are
missed, and the recall figure is where that cost shows up.

## Summarisation

**ROUGE-1, ROUGE-2 and ROUGE-L F1, with clipped n-gram counts.**

All three because they answer different questions. ROUGE-1 ignores word order entirely, so a
bag of the right words scores well. ROUGE-L is based on the longest common subsequence and
rewards order. A summary that scores high on ROUGE-1 and low on ROUGE-L has the right
content in the wrong arrangement.

Clipped counts mean repeating a word cannot inflate the score.

**What ROUGE does not measure: meaning.** A correct paraphrase sharing no vocabulary with
the reference scores near zero. It is used here because it is free, deterministic and
comparable across methods — which is exactly what is needed to compare two baselines — and
it is the wrong tool for judging whether a summary is *good*.

Compression is reported alongside, because a "summary" at 95% of the source length is not
one, however well it scores.

`centroid-3` beats `lead-3` by 0.042 ROUGE-L on test. On eight examples that is a direction,
not a result, and the README says so.

## Document Q&A

**Exact match and token F1 on answerable questions; abstention rates on both kinds.**

Seven of 24 questions are unanswerable by their document. Including them is the point: an
analyzer that answers everything scores well on exact match and **zero** on true abstention,
and a single combined metric would hide which of those you have.

- **False abstention rate** — answerable questions wrongly declined. An error rate; lower is
  better.
- **True abstention rate** — unanswerable questions correctly declined. A success rate;
  higher is better.

On the test split the extractive baseline declines 100% of unanswerable questions and 0% of
answerable ones. On dev it wrongly declines 22.2% (two questions), where the answering
sentence shares little vocabulary with the question. The support threshold (0.34 of the
question's IDF-weighted terms) was chosen by reasoning rather than tuned, so that gap is
real and not hidden by fitting.

## JSON repair

**Recovery rate, and which repair each case needed.**

The headline rate matters less than the breakdown. A pipeline needing
`single_to_double_quotes` on a third of its responses has a prompt problem that no amount of
repair will fix properly, and it will never find out if the repair is silent. Every
`ParseOutcome` carries the repairs it required, and the analyzers expose the last set on
`last_repairs`.

Each case is labelled with `should_parse`. Seven cases are labelled unrecoverable on
purpose — truncated output, prose with no JSON at all, an empty response, a label outside
the allowed set — and CI fails if any case stops behaving as labelled. That makes the
harness a test of the repair pipeline rather than just an exercise of it.

Repairs are ordered least- to most-invasive, and each is attempted only after the previous
parse failed, so well-formed output is never touched. Two are deliberately conservative:

- `single_to_double_quotes` runs **only** when the text contains no double quote at all. An
  apostrophe inside a legitimate double-quoted string would otherwise be rewritten, turning
  a valid document into an invalid one.
- `python_literals` and `remove_trailing_commas` operate **outside string literals only**,
  via a string-aware scan. Applying them to the whole text is the obvious implementation and
  it corrupts prose values containing the word "None" or a comma before a bracket.

## What is not measured

- **LLM analyzer accuracy.** Every LLM path is implemented and tested against scripted
  responses, including malformed ones. No accuracy figure is published, because that would
  mean paying for a run and asserting its results as fact.
- **Semantic retrieval.** The Chroma + sentence-transformers library is wired and
  unbenchmarked.
- **Summary factuality.** ROUGE cannot detect a fluent summary that states something the
  source does not.
- **Inter-annotator agreement.** One annotator, binary labels.
- **Statistical significance.** Point estimates on small sets. A 0.04 ROUGE-L difference
  between two methods on eight examples is not evidence.

## Reproducing

```bash
analyzer evaluate --data data/eval                    # both splits
analyzer evaluate --data data/eval --split test       # held-out only
analyzer evaluate --data data/eval --split all        # pooled
analyzer repair --cases data/eval/repair_cases.jsonl
```

CI runs the first and last on every push, and both are also available as
`docker compose run --rm evaluate` and `docker compose run --rm repair`.
