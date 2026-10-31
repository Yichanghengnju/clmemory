"""F03 conservative relative-date annotations (screening arm W1, shortlist id F03/F10).

Renders, next to each source header, a clearly-marked derived annotation that
resolves relative date phrases in the evidence body against that source's own
timestamp. The body text is never modified; unknown source timestamps are never
resolved (no ingestion-clock substitution — sweep anti-pattern guard); only a
bounded number of longest-first, non-overlapping phrases are annotated; phrase
granularity is preserved where the phrase carries it (month phrases resolve to
YYYY-MM, year phrases to YYYY — tidemark / aml-memory-jb timex lessons;
shortlist F03). Purely deterministic; no model.

Annotation line format, inserted between header and body:
    [dates derived from source time: "last year"->2022; "two weeks ago"->2023-04-24]
"""
from __future__ import annotations

import dataclasses
import re
from datetime import UTC, datetime, timedelta

MAX_NOTES_DEFAULT = 4
_HEADER_RX = re.compile(
    r"^\[source=(?P<sid>[^\]]*); role=(?P<role>[^\]]*); timestamp=(?P<ts>[^\]]*)\]\n", re.M)

# (regex, granularity) — granularity: day | month | year
_PHRASE_SPECS = (
    (re.compile(r"\byesterday\b", re.I), "day"),
    (re.compile(r"\btoday\b", re.I), "day"),
    (re.compile(r"\btomorrow\b", re.I), "day"),
    (re.compile(r"\b(\d+)\s*days?\s+ago\b", re.I), "day"),
    (re.compile(r"\bin\s+(\d+)\s*days\b", re.I), "day"),
    (re.compile(r"\b(\d+)\s*weeks?\s+ago\b", re.I), "day"),
    (re.compile(r"\bin\s+(\d+)\s*weeks\b", re.I), "day"),
    (re.compile(r"\blast\s+week\b", re.I), "day"),
    (re.compile(r"\bnext\s+week\b", re.I), "day"),
    (re.compile(r"\b(\d+)\s*months?\s+ago\b", re.I), "month"),
    (re.compile(r"\bin\s+(\d+)\s*months\b", re.I), "month"),
    (re.compile(r"\blast\s+month\b", re.I), "month"),
    (re.compile(r"\bnext\s+month\b", re.I), "month"),
    (re.compile(r"\b(\d+)\s*years?\s+ago\b", re.I), "year"),
    (re.compile(r"\bin\s+(\d+)\s*years\b", re.I), "year"),
    (re.compile(r"\blast\s+year\b", re.I), "year"),
    (re.compile(r"\bnext\s+year\b", re.I), "year"),
    (re.compile(r"\bthis\s+year\b", re.I), "year"),
    (re.compile(r"\blast\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I), "day"),
    (re.compile(r"\bnext\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I), "day"),
)
_WEEKDAYS = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
             "friday": 4, "saturday": 5, "sunday": 6}
_WEEKDAY_RX = re.compile(r"monday|tuesday|wednesday|thursday|friday|saturday|sunday", re.I)


def _resolve_day(text, now):
    if text == "yesterday":
        return (now - timedelta(days=1)).date().isoformat()
    if text == "today":
        return now.date().isoformat()
    if text == "tomorrow":
        return (now + timedelta(days=1)).date().isoformat()
    if text == "last week":
        return (now - timedelta(days=7)).date().isoformat()
    if text == "next week":
        return (now + timedelta(days=7)).date().isoformat()
    m = re.search(r"(\d+)", text)
    n = int(m.group(1)) if m else 0
    unit_days = 7 if "week" in text else 1
    if "ago" in text or text.startswith("last "):
        return (now - timedelta(days=n * unit_days)).date().isoformat()
    return (now + timedelta(days=n * unit_days)).date().isoformat()


def _resolve_month(text, now):
    m = re.search(r"(\d+)", text)
    n = int(m.group(1)) if m else 1
    if "ago" in text or text.startswith("last "):
        sign = -1
    elif text.startswith("in ") or text.startswith("next "):
        sign = 1
    else:  # "this month" or bare form
        sign = 0
    total = now.year * 12 + (now.month - 1) + sign * n
    return f"{total // 12:04d}-{total % 12 + 1:02d}"


def _resolve_year(text, now):
    if "this" in text:
        return f"{now.year:04d}"
    m = re.search(r"(\d+)", text)
    n = int(m.group(1)) if m else 1
    sign = -1 if ("ago" in text or text.startswith("last ")) else 1
    return f"{now.year + sign * n:04d}"


def _resolve_phrase(phrase, granularity, now):
    text = phrase.lower()
    if _WEEKDAY_RX.search(text) and ("last " in text or "next " in text):
        target = _WEEKDAYS[_WEEKDAY_RX.search(text).group(0).lower()]
        if text.startswith("last "):
            delta = (now.weekday() - target) % 7 or 7
            return (now - timedelta(days=delta)).date().isoformat()
        delta = (target - now.weekday()) % 7 or 7
        return (now + timedelta(days=delta)).date().isoformat()
    if granularity == "day":
        return _resolve_day(text, now)
    if granularity == "month":
        return _resolve_month(text, now)
    return _resolve_year(text, now)


def _notes_for(body, now, max_notes):
    candidates = []
    for rx, granularity in _PHRASE_SPECS:
        for m in rx.finditer(body):
            candidates.append((m.start(), m.end(), m.group(0), granularity))
    # longest phrase first; skip overlaps; cap count
    candidates.sort(key=lambda t: (-(t[1] - t[0]), t[0]))
    notes, taken = [], []
    for start, end, phrase, granularity in candidates:
        if len(notes) >= max_notes:
            break
        if any(not (end <= s or start >= e) for s, e in taken):
            continue
        taken.append((start, end))
        notes.append(f'"{phrase}"->{_resolve_phrase(phrase, granularity, now)}')
    return notes


def _annotate_block(block, max_notes):
    m = _HEADER_RX.match(block)
    if not m or m.group("ts") in ("", "unknown"):
        return block
    try:
        stamp = datetime.fromtimestamp(float(m.group("ts")) / 1000, UTC)
    except (ValueError, OSError, OverflowError):
        return block
    header_end = m.end()
    body = block[header_end:]
    notes = _notes_for(body, stamp, max_notes)
    if not notes:
        return block
    line = "[dates derived from source time: " + "; ".join(notes) + "]\n"
    return block[:header_end] + line + body


def annotate_evidence(hits, *, max_notes=MAX_NOTES_DEFAULT):
    """Return hits whose contents gain per-source derived date annotation
    lines. Headers and bodies otherwise stay byte-identical."""
    if max_notes <= 0 or not hits:
        return hits
    out = []
    for hit in hits:
        headers = list(_HEADER_RX.finditer(hit.content))
        if not headers:
            out.append(hit)
            continue
        parts = []
        preamble = hit.content[:headers[0].start()]
        if preamble.strip():
            parts.append(preamble)
        for i, hm in enumerate(headers):
            end = headers[i + 1].start() if i + 1 < len(headers) else len(hit.content)
            parts.append(_annotate_block(hit.content[hm.start():end], max_notes).strip("\n"))
        new_content = "\n".join(parts)
        if new_content == hit.content:
            out.append(hit)
            continue
        out.append(dataclasses.replace(hit, content=new_content))
    return out
