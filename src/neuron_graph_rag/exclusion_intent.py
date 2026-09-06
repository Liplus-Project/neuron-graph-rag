from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from .models import SearchHit
from .retrieval import TOKEN_PATTERN, tokenize

_QUOTED_TEXT = re.compile(
    r'"(?:\\.|[^"\\])*"'
    r"|'[^']*'"
    r"|`[^`]*`"
    r"|「[^」]*」"
    r"|『[^』]*』"
)
_ENGLISH_MARKER = re.compile(
    r"(?i:\b(?:without|excluding|exclude|except|avoid)\b|\bbut\s+not\b)"
)
_JAPANESE_PREFIX_MARKER = re.compile(
    r"(?:^|(?<=[,、;；]))\s*除外\s*[:：]"
)
_JAPANESE_SUFFIX_EXCLUSION = re.compile(
    r"(?:^|[,、;；])\s*(?P<clause>[^,、;；]+?)\s*"
    r"(?:を除外|を除く|は含めない)"
    r"(?=$|[,、;；])"
)
_TRAILING_CONJUNCTION = re.compile(r"(?i:\s+(?:and|or|but))\s*$")
_TRAILING_POSITIVE_CONNECTOR = re.compile(
    r"(?i:\s+(?:and|but|while))\s*$"
)
_NEGATED_ENGLISH_PREFIX = re.compile(
    r"(?i:(?:\b(?:without|no|avoid(?:s|ed|ing)?|"
    r"exclud(?:e|es|ed|ing))\b|"
    r"\b(?:do|does|did|is|are|was|were|will)\s+not\b)"
    r"(?:[\s,:;-]+[A-Za-z0-9_]+){0,3}[\s,:;-]*$)"
)
_NEGATED_ENGLISH_SUFFIX = re.compile(
    r"(?i:^\s*(?:"
    r"-free\b|"
    r"(?:is|are|was|were)\s+not\s+"
    r"(?:used|accepted|enabled|supported|allowed|included|returned|"
    r"selected|required|present)\b|"
    r"(?:do|does|did)\s+not\s+"
    r"(?:use|run|accept|enable|support|allow|include|return|select|require)\b"
    r"))"
)
_NEGATED_JAPANESE_SUFFIX = re.compile(
    r"^\s*(?:を|は|が)?\s*(?:"
    r"使わない|使用しない|利用しない|用いない|含まない|"
    r"含めない|不要|除外する|除く|なし|"
    r"使われていない|使用されていない|利用されていない|"
    r"採用されていない|許可されていない|受け付けられていない)"
)
_TOKEN_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "document",
        "documents",
        "file",
        "files",
        "return",
        "returned",
        "returning",
        "result",
        "results",
        "the",
        "while",
    }
)
_NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}


@dataclass(frozen=True, slots=True)
class RetrievalIntent:
    original_query: str
    positive_query: str
    exclusion_clauses: tuple[str, ...]


def _clean_clause(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip(" \t\r\n,;:、。；："))


def _clean_exclusion_clause(value: str) -> str:
    previous = ""
    cleaned = _clean_clause(value)
    while cleaned != previous:
        previous = cleaned
        cleaned = _clean_clause(_TRAILING_CONJUNCTION.sub("", cleaned))
    return cleaned


def _clean_positive_clause(value: str) -> str:
    return _clean_clause(_TRAILING_POSITIVE_CONNECTOR.sub("", value))


def _mask_quoted_text(value: str) -> str:
    characters = list(value)
    for match in _QUOTED_TEXT.finditer(value):
        characters[match.start() : match.end()] = " " * (
            match.end() - match.start()
        )
    return "".join(characters)


def decompose_exclusion_intent(query: str) -> RetrievalIntent:
    """Split conservative explicit exclusion syntax from a retrieval query."""

    if not isinstance(query, str) or not query.strip():
        raise TypeError("query must be a non-empty string")

    quoted_mask = _mask_quoted_text(query)
    suffix_rows: list[tuple[int, str]] = []
    remaining = list(query)
    for match in _JAPANESE_SUFFIX_EXCLUSION.finditer(quoted_mask):
        clause = _clean_exclusion_clause(
            query[match.start("clause") : match.end("clause")]
        )
        if not clause:
            raise ValueError("each exclusion marker must have a clause")
        suffix_rows.append((match.start(), clause))
        remaining[match.start() : match.end()] = " " * (
            match.end() - match.start()
        )

    remaining_text = "".join(remaining)
    remaining_mask = _mask_quoted_text(remaining_text)
    marker_matches = sorted(
        (
            *_ENGLISH_MARKER.finditer(remaining_mask),
            *_JAPANESE_PREFIX_MARKER.finditer(remaining_mask),
        ),
        key=lambda match: match.start(),
    )
    if not marker_matches and not suffix_rows:
        return RetrievalIntent(query, query, ())

    first_marker_start = (
        marker_matches[0].start() if marker_matches else len(remaining_text)
    )
    positive_query = _clean_positive_clause(
        remaining_text[:first_marker_start]
    )
    if not positive_query:
        raise ValueError("an exclusion-only query has no positive retrieval intent")

    exclusion_rows = list(suffix_rows)
    for index, marker in enumerate(marker_matches):
        end = (
            marker_matches[index + 1].start()
            if index + 1 < len(marker_matches)
            else len(remaining_text)
        )
        clause = _clean_exclusion_clause(remaining_text[marker.end() : end])
        if not clause:
            raise ValueError("each exclusion marker must have a clause")
        exclusion_rows.append((marker.start(), clause))

    return RetrievalIntent(
        original_query=query,
        positive_query=positive_query,
        exclusion_clauses=tuple(
            clause for _, clause in sorted(exclusion_rows, key=lambda row: row[0])
        ),
    )


def _phrase_pattern(clause: str) -> re.Pattern[str]:
    escaped = re.escape(_clean_clause(clause)).replace(r"\ ", r"\s+")
    prefix = r"(?<![A-Za-z0-9_])" if clause[0].isascii() else ""
    suffix = r"(?![A-Za-z0-9_])" if clause[-1].isascii() else ""
    return re.compile(prefix + escaped + suffix, re.IGNORECASE)


def _mention_is_negated(text: str, start: int, end: int) -> bool:
    prefix = text[max(0, start - 80) : start]
    suffix = text[end : end + 32]
    if re.search(r"(?i:\bnot\s+only\s*)$", prefix):
        return False
    return bool(
        _NEGATED_ENGLISH_PREFIX.search(prefix)
        or _NEGATED_ENGLISH_SUFFIX.match(suffix)
        or _NEGATED_JAPANESE_SUFFIX.match(suffix)
    )


def _canonical_token(value: str) -> str:
    token = _NUMBER_WORDS.get(value, value)
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and len(token) > 4:
        return token[:-1]
    return token


def _significant_tokens(value: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            canonical
            for token in tokenize(value)
            if token not in _TOKEN_STOP_WORDS
            for canonical in (_canonical_token(token),)
        )
    )


def _token_occurrences(text: str, token: str) -> tuple[re.Match[str], ...]:
    return tuple(
        match
        for match in TOKEN_PATTERN.finditer(text)
        if _canonical_token(match.group(0).casefold()) == token
    )


def _token_identifier_match(text: str, clause: str) -> bool:
    clause_tokens = _significant_tokens(clause)
    if not clause_tokens:
        return False
    text_tokens = {_canonical_token(token) for token in tokenize(text)}
    if not set(clause_tokens).issubset(text_tokens):
        return False
    for token in clause_tokens:
        occurrences = _token_occurrences(text, token)
        if occurrences and all(
            _mention_is_negated(text, match.start(), match.end())
            for match in occurrences
        ):
            return False
    return True


def _decision(hit: SearchHit, intent: RetrievalIntent) -> dict[str, Any]:
    matched: list[dict[str, str]] = []
    negated_mentions: list[str] = []
    for clause in intent.exclusion_clauses:
        direct_match = False
        negated_match = False
        phrase_matches = tuple(_phrase_pattern(clause).finditer(hit.node.text))
        for match in phrase_matches:
            if _mention_is_negated(hit.node.text, match.start(), match.end()):
                negated_match = True
            else:
                direct_match = True
        if negated_match:
            negated_mentions.append(clause)
        if direct_match:
            matched.append(
                {
                    "clause": clause,
                    "reason": "direct_phrase_match",
                }
            )
        elif not phrase_matches and _token_identifier_match(
            hit.node.text, clause
        ):
            matched.append(
                {
                    "clause": clause,
                    "reason": "token_identifier_match",
                }
            )
    accepted = not matched
    return {
        "accepted": accepted,
        "reason": (
            "no_exclusion_match" if accepted else "excluded_by_query_clause"
        ),
        "exclusion_clauses": list(intent.exclusion_clauses),
        "matched_exclusions": matched,
        "negated_mentions": negated_mentions,
        "node_id": hit.node.node_id,
    }


def apply_exclusion_intent(
    ranked_hits: Sequence[SearchHit], intent: RetrievalIntent
) -> tuple[tuple[SearchHit, ...], tuple[SearchHit, ...]]:
    """Annotate ranked candidates and return the non-excluded subsequence."""

    if not intent.exclusion_clauses:
        return tuple(ranked_hits), tuple(ranked_hits)
    annotated: list[SearchHit] = []
    accepted: list[SearchHit] = []
    for hit in ranked_hits:
        decision = _decision(hit, intent)
        annotated_hit = replace(hit, exclusion_intent=decision)
        annotated.append(annotated_hit)
        if decision["accepted"]:
            accepted.append(annotated_hit)
    return tuple(annotated), tuple(accepted)
