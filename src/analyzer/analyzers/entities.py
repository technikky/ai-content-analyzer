"""Entity extraction: a rule-based baseline and an LLM extractor.

The baseline is layered. Regular expressions handle the categories that genuinely are
patterns -- emails, URLs, money, percentages, dates -- and those are close to exact.
Capitalised-sequence heuristics plus small gazetteers handle PERSON, ORG and LOCATION, and
those are not exact at all; they trade recall for precision in specific, documented ways:

* A **single** capitalised word is never a PERSON. "Approved the request" begins a sentence
  with a capital, and admitting single tokens turns every sentence opening into a false
  positive. The cost is that one-word names are missed, which the measured recall shows.
* ORG is recognised from a suffix (``Inc``, ``GmbH``, ``Holdings``) or a gazetteer, so an
  organisation named like a person is missed.
* LOCATION is gazetteer-only, so anywhere not on the list is missed.

These are the right trade-offs for a baseline whose job is to make an LLM's numbers
interpretable, and :mod:`analyzer.evaluation` reports strict and partial scores separately
so boundary errors are distinguishable from misses.
"""

from __future__ import annotations

import re

from analyzer.analyzers.base import AnalyzerError, EntityExtractor, resolve_overlaps
from analyzer.providers import LLM
from analyzer.schemas import EntitiesResponse, parse_model_output
from analyzer.types import Entity, EntityType

ENTITY_PROMPT = """Extract named entities from the text below.

Allowed types: PERSON, ORG, LOCATION, DATE, MONEY, PERCENT, EMAIL, URL

Rules:
- Copy each entity exactly as it appears in the text, with no added or removed characters.
- Do not invent entities that are not present.
- Reply with JSON only: {{"entities": [{{"text": "<exact text>", "type": "<TYPE>"}}]}}

TEXT
{text}

JSON"""

EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
URL_RE = re.compile(r"\bhttps?://[^\s<>\"')\]]+|\bwww\.[^\s<>\"')\]]+", re.IGNORECASE)
# No trailing \b after the alternation: `%` is a non-word character, so requiring a word
# boundary after it means "18% growth" never matches, while "18% " at end of input does.
# That asymmetry is why PERCENT scored only 0.67 before this was fixed. The spelled-out
# forms keep their own boundary.
PERCENT_RE = re.compile(r"\b\d+(?:\.\d+)?\s?(?:%|per\s?cent\b|percent\b)", re.IGNORECASE)
MONEY_RE = re.compile(
    r"[$£€¥]\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:million|billion|trillion|thousand|bn|m|k))?"
    r"|\b\d[\d,]*(?:\.\d+)?\s?(?:million|billion|trillion)?\s?"
    r"(?:dollars|euros|pounds|yen|USD|EUR|GBP|JPY)\b",
    re.IGNORECASE,
)

_MONTHS = (
    r"January|February|March|April|May|June|July|August|September|October|November|December"
    r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)
DATE_RES = (
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(rf"\b(?:{_MONTHS})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b"),
    re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTHS})\.?\s+\d{{4}}\b"),
    re.compile(rf"\b(?:{_MONTHS})\.?\s+\d{{4}}\b"),
    re.compile(r"\bQ[1-4]\s+\d{4}\b"),
    re.compile(r"\b(?:19|20)\d{2}\b"),
)

ORG_SUFFIXES = frozenset(
    [
        "inc",
        "incorporated",
        "ltd",
        "limited",
        "llc",
        "llp",
        "plc",
        "corp",
        "corporation",
        "company",
        "co",
        "gmbh",
        "ag",
        "sa",
        "nv",
        "bv",
        "ab",
        "oy",
        "group",
        "holdings",
        "technologies",
        "technology",
        "systems",
        "solutions",
        "labs",
        "laboratories",
        "foundation",
        "institute",
        "university",
        "college",
        "school",
        "bank",
        "airlines",
        "partners",
        "ventures",
        "capital",
        "media",
        "studios",
        "foods",
        "motors",
        "pharmaceuticals",
        "industries",
        "enterprises",
        "associates",
        "consulting",
        "networks",
    ]
)

ORG_GAZETTEER = frozenset(
    {
        "world health organization",
        "european commission",
        "united nations",
        "nato",
        "european union",
        "world bank",
        "reserve bank",
        "federal reserve",
    }
)

LOCATION_GAZETTEER = frozenset(
    [
        "malaysia",
        "singapore",
        "indonesia",
        "thailand",
        "vietnam",
        "philippines",
        "japan",
        "china",
        "india",
        "pakistan",
        "bangladesh",
        "australia",
        "zealand",
        "britain",
        "england",
        "scotland",
        "wales",
        "ireland",
        "france",
        "germany",
        "spain",
        "portugal",
        "italy",
        "greece",
        "netherlands",
        "belgium",
        "switzerland",
        "austria",
        "sweden",
        "norway",
        "denmark",
        "finland",
        "poland",
        "ukraine",
        "russia",
        "turkey",
        "egypt",
        "nigeria",
        "kenya",
        "africa",
        "morocco",
        "brazil",
        "argentina",
        "chile",
        "colombia",
        "mexico",
        "canada",
        "states",
        "america",
        "penang",
        "kuala",
        "lumpur",
        "johor",
        "selangor",
        "sabah",
        "sarawak",
        "singapore",
        "london",
        "manchester",
        "birmingham",
        "edinburgh",
        "dublin",
        "paris",
        "berlin",
        "munich",
        "madrid",
        "barcelona",
        "rome",
        "milan",
        "amsterdam",
        "brussels",
        "zurich",
        "geneva",
        "vienna",
        "stockholm",
        "oslo",
        "copenhagen",
        "helsinki",
        "warsaw",
        "kyiv",
        "moscow",
        "istanbul",
        "cairo",
        "lagos",
        "nairobi",
        "tokyo",
        "osaka",
        "beijing",
        "shanghai",
        "shenzhen",
        "delhi",
        "mumbai",
        "bangalore",
        "karachi",
        "dhaka",
        "sydney",
        "melbourne",
        "auckland",
        "toronto",
        "vancouver",
        "montreal",
        "york",
        "francisco",
        "angeles",
        "chicago",
        "boston",
        "seattle",
        "austin",
        "denver",
        "miami",
        "asia",
        "europe",
        "pacific",
        "atlantic",
        "mediterranean",
    ]
)

HONORIFICS = frozenset({"mr", "mrs", "ms", "miss", "dr", "prof", "professor", "sir", "dame"})

#: Function words that may appear inside a capitalised name ("Bank of England").
NAME_CONNECTORS = frozenset({"of", "the", "and", "for", "de", "van", "von", "del", "da", "bin"})

#: Capitalised words that are almost never part of a name in running prose. Without this,
#: every sentence-initial adverb becomes a candidate.
NON_NAME_WORDS = frozenset(
    [
        "the",
        "a",
        "an",
        "this",
        "that",
        "these",
        "those",
        "we",
        "they",
        "he",
        "she",
        "it",
        "our",
        "their",
        "his",
        "her",
        "its",
        "there",
        "here",
        "however",
        "therefore",
        "moreover",
        "furthermore",
        "meanwhile",
        "although",
        "because",
        "while",
        "when",
        "where",
        "what",
        "which",
        "who",
        "why",
        "how",
        "if",
        "then",
        "than",
        "and",
        "or",
        "but",
        "so",
        "yet",
        "for",
        "nor",
        "after",
        "before",
        "during",
        "since",
        "until",
        "unless",
        "whereas",
        "despite",
        "though",
        "also",
        "both",
        "each",
        "every",
        "either",
        "neither",
        "all",
        "some",
        "many",
        "most",
        "few",
        "more",
        "less",
        "least",
        "best",
        "worst",
        "first",
        "second",
        "third",
        "next",
        "last",
        "other",
        "another",
        "such",
        "only",
        "just",
        "even",
        "still",
        "already",
        "again",
        "once",
        "twice",
        "always",
        "never",
        "often",
        "sometimes",
        "usually",
        "rarely",
        "perhaps",
        "maybe",
    ]
)

#: Head nouns of the "X of Y" organisation pattern: "Bank of England", "University of
#: Malaya". Without these the run is a sequence of capitalised words with no org suffix at
#: the end, and the person heuristic claims it.
ORG_HEAD_NOUNS = frozenset(
    [
        "bank",
        "university",
        "institute",
        "ministry",
        "department",
        "board",
        "council",
        "office",
        "school",
        "college",
        "academy",
        "society",
        "association",
        "federation",
        "organization",
        "organisation",
        "authority",
        "commission",
        "agency",
        "bureau",
        "centre",
        "center",
        "court",
        "parliament",
        "congress",
        "senate",
    ]
)

#: A capitalised word, then further capitalised words, optionally joined by a connector
#: that really does occur inside names ("Bank of England"). Allowing an arbitrary short
#: lowercase word here merges neighbouring entities: "Oslo to Copenhagen" becomes one run
#: and is then classified as a single, wrong PERSON.
_CAP_RUN = re.compile(
    r"\b[A-Z][\w'’\-]*"
    r"(?:\s+(?:of|the|and|for|de|van|von|del|da|bin|&)\s+[A-Z][\w'’\-]*"
    r"|\s+[A-Z][\w'’\-]*)*"
)

#: Punctuation that ends a sentence rather than a URL or an address. Untrimmed, "visit
#: https://example.com." yields a URL whose span includes the full stop: the entity is
#: found but its boundaries are wrong, which the evaluation correctly scores as a miss.
_TRAILING_PUNCTUATION = ".,;:!?)”’'\""


def _trim_trailing(text: str, start: int, end: int) -> int:
    while end > start and text[end - 1] in _TRAILING_PUNCTUATION:
        end -= 1
    return end


def _add(
    found: list[Entity], text: str, start: int, end: int, kind: EntityType, confidence: float
) -> None:
    surface = text[start:end].strip()
    if surface:
        # Re-derive the span in case stripping moved the boundary.
        offset = text[start:end].index(surface)
        found.append(
            Entity(
                text=surface,
                type=kind,
                start=start + offset,
                end=start + offset + len(surface),
                confidence=confidence,
            )
        )


class RuleEntityExtractor:
    """Regex categories plus capitalised-sequence heuristics for names."""

    @property
    def name(self) -> str:
        return "rules"

    def extract(self, text: str) -> list[Entity]:
        found: list[Entity] = []

        # Highest-confidence, most specific patterns first. Order matters only for
        # readability; overlaps are resolved by length afterwards.
        for match in EMAIL_RE.finditer(text):
            end = _trim_trailing(text, match.start(), match.end())
            _add(found, text, match.start(), end, "EMAIL", 0.99)
        for match in URL_RE.finditer(text):
            end = _trim_trailing(text, match.start(), match.end())
            _add(found, text, match.start(), end, "URL", 0.99)
        for match in PERCENT_RE.finditer(text):
            _add(found, text, match.start(), match.end(), "PERCENT", 0.95)
        for match in MONEY_RE.finditer(text):
            _add(found, text, match.start(), match.end(), "MONEY", 0.95)
        for pattern in DATE_RES:
            for match in pattern.finditer(text):
                # A bare four-digit year is a weaker signal than a full date, and often is
                # a version or a quantity rather than a date.
                confidence = 0.6 if pattern is DATE_RES[-1] else 0.9
                _add(found, text, match.start(), match.end(), "DATE", confidence)

        for match in _CAP_RUN.finditer(text):
            entity = self._classify_run(text, match.start(), match.end())
            if entity is not None:
                found.append(entity)

        return resolve_overlaps(found)

    def _classify_run(self, text: str, start: int, end: int) -> Entity | None:
        surface = text[start:end]
        words = surface.split()
        if not words:
            return None

        lowered = [word.strip(".,;:!?").lower() for word in words]

        # A leading article belongs to the sentence, not to the name: the label for
        # "The Bank of England" is "Bank of England".
        if lowered and lowered[0] == "the" and len(words) > 1:
            start += len(words[0]) + 1
            words, lowered = words[1:], lowered[1:]

        # Trailing connector, as in "Bank of" at the end of a clause.
        while lowered and lowered[-1] in NAME_CONNECTORS:
            words, lowered = words[:-1], lowered[:-1]
            end = start + len(" ".join(words))
        if not words:
            return None

        joined = " ".join(lowered)

        if joined in ORG_GAZETTEER:
            return Entity(text=text[start:end], type="ORG", start=start, end=end, confidence=0.9)
        if lowered[-1] in ORG_SUFFIXES and len(words) >= 2:
            return Entity(text=text[start:end], type="ORG", start=start, end=end, confidence=0.9)
        # The "X of Y" organisation pattern, whose suffix is at the front rather than the end.
        if lowered[0] in ORG_HEAD_NOUNS and len(words) >= 2:
            return Entity(text=text[start:end], type="ORG", start=start, end=end, confidence=0.8)
        if joined in LOCATION_GAZETTEER or (len(words) == 1 and lowered[0] in LOCATION_GAZETTEER):
            return Entity(
                text=text[start:end], type="LOCATION", start=start, end=end, confidence=0.85
            )
        if len(words) >= 2 and all(word in LOCATION_GAZETTEER for word in lowered):
            return Entity(
                text=text[start:end], type="LOCATION", start=start, end=end, confidence=0.85
            )
        if lowered[0] in HONORIFICS and len(words) >= 2:
            return Entity(text=text[start:end], type="PERSON", start=start, end=end, confidence=0.9)

        # Multi-word capitalised run with no function words: treat as a person. Single
        # capitalised words are deliberately not candidates.
        content = [word for word in lowered if word not in NAME_CONNECTORS]
        if len(content) >= 2 and not any(word in NON_NAME_WORDS for word in content):
            return Entity(text=text[start:end], type="PERSON", start=start, end=end, confidence=0.6)
        return None


class LLMEntityExtractor:
    """Asks a model for entities, then locates each one in the source text.

    A model returns entity *strings*; a caller highlighting them needs *spans*. Anything
    the model returns that does not occur in the text verbatim is **dropped**, which is
    the cheapest available hallucination filter: an entity that is not in the document is
    not an entity from the document.
    """

    def __init__(self, llm: LLM, *, max_tokens: int = 800) -> None:
        self.llm = llm
        self.max_tokens = max_tokens
        self.last_repairs: tuple[str, ...] = ()
        self.last_dropped: list[str] = []

    @property
    def name(self) -> str:
        return f"llm-entities:{self.llm.name}"

    def extract(self, text: str) -> list[Entity]:
        raw = self.llm.complete(ENTITY_PROMPT.format(text=text), max_tokens=self.max_tokens)
        outcome = parse_model_output(raw, EntitiesResponse)
        self.last_repairs = outcome.repairs
        if not outcome.ok or outcome.value is None:
            raise AnalyzerError(f"entity response unusable: {outcome.error}")

        found: list[Entity] = []
        dropped: list[str] = []
        used: list[tuple[int, int]] = []

        for item in outcome.value.entities:
            span = self._locate(text, item.text, used)
            if span is None:
                dropped.append(item.text)
                continue
            start, end = span
            used.append(span)
            found.append(
                Entity(text=text[start:end], type=item.type, start=start, end=end, confidence=0.8)
            )

        self.last_dropped = dropped
        return resolve_overlaps(found)

    @staticmethod
    def _locate(text: str, needle: str, used: list[tuple[int, int]]) -> tuple[int, int] | None:
        """First occurrence of ``needle`` not already claimed by another entity.

        Skipping claimed spans is what makes a name mentioned three times produce three
        entities rather than three copies of the first occurrence.
        """
        cleaned = needle.strip()
        if not cleaned:
            return None
        cursor = 0
        while True:
            index = text.find(cleaned, cursor)
            if index == -1:
                return None
            span = (index, index + len(cleaned))
            if not any(span[0] < u[1] and u[0] < span[1] for u in used):
                return span
            cursor = index + 1


def build_entity_extractor(method: str = "rules", *, llm: LLM | None = None) -> EntityExtractor:
    key = method.strip().lower()
    if key == "rules":
        return RuleEntityExtractor()
    if key == "llm":
        if llm is None:
            raise ValueError("the 'llm' entity extractor needs a provider; pass llm=...")
        return LLMEntityExtractor(llm)
    raise ValueError(f"unknown entity extractor {method!r}; expected rules or llm")
