"""Sentiment: a lexicon baseline and an LLM analyzer.

The lexicon handles the two things that break naive word counting on real text:

* **Negation.** "not good" is not evidence of positivity. A negator within three tokens
  before a sentiment word flips its polarity.
* **Intensity.** "extremely poor" is stronger than "poor", and "slightly poor" is weaker.

It still cannot handle sarcasm, comparison ("better than the old one, which was awful"),
or sentiment expressed without a sentiment word. Those are real limits, they are listed in
the docs, and they are visible in the measured confusion matrix rather than hidden.
"""

from __future__ import annotations

from analyzer.analyzers.base import AnalyzerError, SentimentAnalyzer
from analyzer.providers import LLM
from analyzer.schemas import SentimentResponse, parse_model_output
from analyzer.text import token_spans
from analyzer.types import SentimentLabel, SentimentResult

SENTIMENT_PROMPT = """Classify the sentiment of the text below.

Rules:
- label must be exactly one of: positive, negative, neutral, mixed
- use "mixed" when the text expresses both positive and negative sentiment
- use "neutral" when it expresses none
- score is -1.0 (wholly negative) to 1.0 (wholly positive)
- Reply with JSON only: {{"label": "<label>", "score": <number>, "rationale": "<one sentence>"}}

TEXT
{text}

JSON"""

#: Weights are 1.0 for ordinary terms and 2.0 for strong ones. Hand-built for
#: product-and-service prose, which is what the evaluation set contains.
POSITIVE_TERMS: dict[str, float] = {
    "excellent": 2.0,
    "outstanding": 2.0,
    "superb": 2.0,
    "brilliant": 2.0,
    "flawless": 2.0,
    "exceptional": 2.0,
    "perfect": 2.0,
    "delighted": 2.0,
    "love": 2.0,
    "loved": 2.0,
    "fantastic": 2.0,
    "wonderful": 2.0,
    "impressive": 2.0,
    "remarkable": 2.0,
    "good": 1.0,
    "great": 1.0,
    "nice": 1.0,
    "solid": 1.0,
    "helpful": 1.0,
    "useful": 1.0,
    "fast": 1.0,
    "quick": 1.0,
    "responsive": 1.0,
    "reliable": 1.0,
    "stable": 1.0,
    "clean": 1.0,
    "clear": 1.0,
    "intuitive": 1.0,
    "smooth": 1.0,
    "polished": 1.0,
    "recommend": 1.0,
    "recommended": 1.0,
    "happy": 1.0,
    "pleased": 1.0,
    "satisfied": 1.0,
    "improved": 1.0,
    "improvement": 1.0,
    "better": 1.0,
    "best": 1.0,
    "efficient": 1.0,
    "affordable": 1.0,
    "valuable": 1.0,
    "worth": 1.0,
    "generous": 1.0,
    "thorough": 1.0,
    "accurate": 1.0,
    "consistent": 1.0,
    "straightforward": 1.0,
    "convenient": 1.0,
    "comprehensive": 1.0,
    "elegant": 1.0,
    "robust": 1.0,
    "welcome": 1.0,
    "enjoyed": 1.0,
    "enjoy": 1.0,
    "appreciate": 1.0,
    "appreciated": 1.0,
    "praise": 1.0,
    "positive": 1.0,
    "works": 1.0,
    "worked": 1.0,
    "easy": 1.0,
    "simple": 1.0,
    "smart": 1.0,
}

NEGATIVE_TERMS: dict[str, float] = {
    "terrible": 2.0,
    "awful": 2.0,
    "appalling": 2.0,
    "useless": 2.0,
    "unusable": 2.0,
    "disaster": 2.0,
    "horrible": 2.0,
    "unacceptable": 2.0,
    "hate": 2.0,
    "hated": 2.0,
    "atrocious": 2.0,
    "abysmal": 2.0,
    "broken": 2.0,
    "worst": 2.0,
    "furious": 2.0,
    "bad": 1.0,
    "poor": 1.0,
    "slow": 1.0,
    "sluggish": 1.0,
    "buggy": 1.0,
    "bug": 1.0,
    "bugs": 1.0,
    "crash": 1.0,
    "crashes": 1.0,
    "crashed": 1.0,
    "fails": 1.0,
    "failed": 1.0,
    "failure": 1.0,
    "error": 1.0,
    "errors": 1.0,
    "confusing": 1.0,
    "confused": 1.0,
    "clunky": 1.0,
    "awkward": 1.0,
    "difficult": 1.0,
    "hard": 1.0,
    "complicated": 1.0,
    "expensive": 1.0,
    "overpriced": 1.0,
    "disappointing": 1.0,
    "disappointed": 1.0,
    "frustrating": 1.0,
    "frustrated": 1.0,
    "annoying": 1.0,
    "annoyed": 1.0,
    "unreliable": 1.0,
    "unstable": 1.0,
    "missing": 1.0,
    "lacking": 1.0,
    "limited": 1.0,
    "outdated": 1.0,
    "unresponsive": 1.0,
    "regression": 1.0,
    "downtime": 1.0,
    "outage": 1.0,
    "complaint": 1.0,
    "complaints": 1.0,
    "worse": 1.0,
    "negative": 1.0,
    "problem": 1.0,
    "problems": 1.0,
    "issue": 1.0,
    "issues": 1.0,
    "wrong": 1.0,
    "cumbersome": 1.0,
    "tedious": 1.0,
}

#: A negator within this many tokens before a sentiment word flips its polarity.
NEGATION_WINDOW = 3
NEGATORS = frozenset(
    {
        "not",
        "no",
        "never",
        "none",
        "cannot",
        "without",
        "hardly",
        "barely",
        "neither",
        "nor",
        "isn't",
        "wasn't",
        "aren't",
        "weren't",
        "don't",
        "doesn't",
        "didn't",
        "can't",
        "couldn't",
        "won't",
        "wouldn't",
        "shouldn't",
        "lacks",
        "lacked",
    }
)
INTENSIFIERS = {
    "very": 1.5,
    "extremely": 2.0,
    "incredibly": 2.0,
    "really": 1.4,
    "highly": 1.5,
    "deeply": 1.5,
    "utterly": 2.0,
    "completely": 1.6,
    "totally": 1.6,
    "absolutely": 1.8,
    "slightly": 0.5,
    "somewhat": 0.6,
    "marginally": 0.5,
    "fairly": 0.8,
    "mildly": 0.5,
    "a": 1.0,
}

#: Above this magnitude the label is polarised; below it, neutral.
POLARITY_THRESHOLD = 0.25
#: When the weaker polarity is at least this fraction of the stronger, the text is mixed.
MIXED_RATIO = 0.5
#: Total sentiment weight at which confidence saturates.
CONFIDENCE_SATURATION = 4.0


def classify(score: float, positive: float, negative: float) -> SentimentLabel:
    """Turn polarity magnitudes into a label.

    Separate from scoring so the decision rule is testable on its own, and so the
    thresholds are in one readable place rather than buried in a scoring loop.
    """
    total = positive + negative
    if total == 0.0:
        return "neutral"

    stronger, weaker = max(positive, negative), min(positive, negative)
    if stronger > 0 and weaker / stronger >= MIXED_RATIO and total >= 2.0:
        return "mixed"
    if score >= POLARITY_THRESHOLD:
        return "positive"
    if score <= -POLARITY_THRESHOLD:
        return "negative"
    return "neutral"


class LexiconSentiment:
    """Weighted lexicon with negation and intensity handling."""

    def __init__(
        self,
        positive: dict[str, float] | None = None,
        negative: dict[str, float] | None = None,
    ) -> None:
        self.positive = dict(positive if positive is not None else POSITIVE_TERMS)
        self.negative = dict(negative if negative is not None else NEGATIVE_TERMS)

    @property
    def name(self) -> str:
        return "lexicon"

    def analyze(self, text: str) -> SentimentResult:
        tokens = [token for token, _, _ in token_spans(text)]

        positive_weight = 0.0
        negative_weight = 0.0
        positive_hits: list[str] = []
        negative_hits: list[str] = []
        negation_count = 0

        for index, token in enumerate(tokens):
            base = self.positive.get(token)
            polarity = 1
            if base is None:
                base = self.negative.get(token)
                polarity = -1
            if base is None:
                continue

            weight = base
            window = tokens[max(0, index - NEGATION_WINDOW) : index]
            for modifier in window:
                if modifier in INTENSIFIERS:
                    weight *= INTENSIFIERS[modifier]
            if any(modifier in NEGATORS for modifier in window):
                polarity *= -1
                negation_count += 1
                # A negated sentiment word is weaker evidence than a plain one: "not bad"
                # is mildly positive, not as positive as "good".
                weight *= 0.7

            if polarity > 0:
                positive_weight += weight
                positive_hits.append(token)
            else:
                negative_weight += weight
                negative_hits.append(token)

        total = positive_weight + negative_weight
        score = 0.0 if total == 0.0 else (positive_weight - negative_weight) / total
        confidence = 0.0 if total == 0.0 else min(1.0, total / CONFIDENCE_SATURATION)

        return SentimentResult(
            label=classify(score, positive_weight, negative_weight),
            score=max(-1.0, min(1.0, score)),
            confidence=confidence,
            positive_terms=positive_hits,
            negative_terms=negative_hits,
            negations=negation_count,
            method=self.name,
        )


class LLMSentiment:
    """Asks a model to classify, and validates the label against the allowed set."""

    def __init__(self, llm: LLM, *, max_tokens: int = 200) -> None:
        self.llm = llm
        self.max_tokens = max_tokens
        self.last_repairs: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return f"llm-sentiment:{self.llm.name}"

    def analyze(self, text: str) -> SentimentResult:
        raw = self.llm.complete(SENTIMENT_PROMPT.format(text=text), max_tokens=self.max_tokens)
        outcome = parse_model_output(raw, SentimentResponse)
        self.last_repairs = outcome.repairs
        if not outcome.ok or outcome.value is None:
            raise AnalyzerError(f"sentiment response unusable: {outcome.error}")

        response = outcome.value
        return SentimentResult(
            label=response.label,
            score=response.score,
            # A model's self-reported certainty is not calibrated, so the magnitude of its
            # own score is used instead of inventing a confidence it did not give.
            confidence=min(1.0, abs(response.score)),
            method=self.name,
        )


def build_sentiment_analyzer(
    method: str = "lexicon", *, llm: LLM | None = None
) -> SentimentAnalyzer:
    key = method.strip().lower()
    if key == "lexicon":
        return LexiconSentiment()
    if key == "llm":
        if llm is None:
            raise ValueError("the 'llm' sentiment analyzer needs a provider; pass llm=...")
        return LLMSentiment(llm)
    raise ValueError(f"unknown sentiment analyzer {method!r}; expected lexicon or llm")
