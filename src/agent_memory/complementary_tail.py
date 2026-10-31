"""F04 normalized complementary tail selection (screening arm W1, shortlist id F04).

Replaces the anchor-count tail ordering with a same-scale greedy selection over
undelivered sources: base score (anchor-hit count, normalized) + coverage bonus
for query content words not yet covered by selected texts − redundancy penalty
(max token-Jaccard against already-selected texts). Core hits are never touched
(protected slots); the arm only orders the appended tail (sweep evidence:
normalized-MMR in agent-memory-service, coverage/MMR in aml-memory-mvp and
starter-collection 10-aml-memory-mvp; shortlist F04/F14 base).

Preserves three pillars: sources are appended verbatim, never rewritten; only
the order/membership of the tail changes. Purely deterministic; no model.
"""
from __future__ import annotations

import hashlib
import json
import re

from .anchor_channel import _render_source, extract_anchors, normalized, STOP_SINGLE
from .domain import Evidence

_WORD_RX = re.compile(r"[a-z0-9]+")


def _content_words(text):
    return {w for w in _WORD_RX.findall((text or "").lower()) if w not in STOP_SINGLE and len(w) > 1}


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


RULE_CUE_RX = re.compile(
    r"\b(i (really |absolutely |just )?(love|like|enjoy|prefer|favor|adore|hate|dislike|despise|"
    r"can'?t stand|avoid|always|never|rarely|only (eat|drink|watch|play|read))"
    r"|i'?m (a (big |huge )?(fan|lover|person)|allergic|vegan|vegetarian|lactose|gluten[- ]free|not (a )?(fan|person))"
    r"|my (favorite|favourite|go[- ]to)"
    r"|i (don'?t|do not|never|won'?t|will not) (eat|drink|like|want|watch|play|read|touch)"
    r"|allergic to|on a diet|trying to (quit|cut (down|back)|avoid)"
    r"|please (always|never|don'?t))\b", re.I)


def rule_quota_sources(sources, query, selected_texts, *, m=8):
    """Mechanical user-preference/rule sentence reservation (shortlist F13).

    user-role sources whose text matches preference/rule cue regexes, ranked by
    cue count then query-word overlap. Returns verbatim Evidence with
    kind="rule_quota"; the caller composes: reserve these slots first, let the
    complementary selector fill the rest. Purely deterministic; three pillars
    safe (tail order/membership only, sources appended verbatim).
    """
    if m <= 0 or not sources:
        return []
    qwords = _content_words(query)
    delivered = {t.lower() for t in selected_texts}
    cands = []
    for order, s in enumerate(sources):
        if (s.get("role") or "").lower() != "user":
            continue
        text = normalized(s.get("content"))
        if not text or text.lower() in delivered:
            continue
        cues = len(RULE_CUE_RX.findall(text))
        if not cues:
            continue
        overlap = len(_content_words(text) & qwords)
        cands.append((cues, overlap, order, s))
    if not cands:
        return []
    cands.sort(key=lambda c: (-c[0], -c[1], c[2]))
    out = []
    for cues, overlap, order, s in cands[:m]:
        out.append(Evidence(
            id="memmachine_rule_quota_" + hashlib.sha256(
                json.dumps([s["id"]], ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()[:24],
            content=_render_source(s), source_ids=(s["id"],),
            score=float(cues), kind="rule_quota"))
    return out


def select_tail(sources, query, selected_texts, *, limit,
                coverage_weight=0.14, redundancy_weight=0.12,
                boost_categories=None, boost_factor=1.0):
    """Greedily fill up to `limit` tail slots with complementary verbatim sources.

    sources: iterable of {id, content, role, timestamp} in stable store order.
    selected_texts: lowercase normalized texts already delivered (cores+neighbors).
    Greedy score = anchor_ratio + coverage_weight*new_word_coverage
                 − redundancy_weight*max_jaccard(selected).
    boost_categories/boost_factor: anchors from these extract_anchors categories
    count boost_factor times in the anchor-hit term (shortlist F06 variant;
    default off keeps equal-weight behavior).
    """
    if limit <= 0 or not sources:
        return []
    boost = set(boost_categories or ())
    weighted_anchors = []
    for cat, vals in extract_anchors(query).items():
        w = boost_factor if cat in boost else 1.0
        weighted_anchors.extend((a.lower(), w) for a in vals)
    query_words = _content_words(query)
    covered_words = set()
    for text in selected_texts:
        covered_words |= (_content_words(text) & query_words)

    delivered_texts = {t.lower() for t in selected_texts}
    scored = []
    for order, source in enumerate(sources):
        text = normalized(source.get("content"))
        key = text.lower()
        if not text or key in delivered_texts:
            continue
        hits = sum(w for a, w in weighted_anchors if a in key)
        words = _content_words(text)
        new_words = (words & query_words) - covered_words
        redundancy = max((_jaccard(words, _content_words(t)) for t in delivered_texts), default=0.0)
        scored.append((-hits, order, source, text, key, len(new_words), redundancy))
    if not scored:
        return []
    max_hits = max(-s[0] for s in scored) or 1
    out = []
    selected_word_sets = []
    remaining = list(scored)
    for _ in range(limit):
        best, best_key = None, None
        for item in remaining:
            neg_hits, order, source, text, key, new_words, redundancy = item
            base = (-neg_hits) / max_hits
            score = base + coverage_weight * (new_words / len(query_words) if query_words else 0.0) \
                    - redundancy_weight * redundancy
            k = (-score, order)
            if best is None or k < best_key:
                best, best_key = item, k
        if best is None:
            break
        neg_hits, order, source, text, key, new_words, redundancy = best
        remaining.remove(best)
        base = (-neg_hits) / max_hits
        delivered_texts.add(key)
        out.append(Evidence(
            id="memmachine_complementary_" + hashlib.sha256(
                json.dumps([source["id"]], ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()[:24],
            content=_render_source(source), source_ids=(source["id"],),
            score=round(base + coverage_weight * (new_words / len(query_words) if query_words else 0.0)
                        - redundancy_weight * redundancy, 6),
            kind="complementary_tail",
        ))
    return out
