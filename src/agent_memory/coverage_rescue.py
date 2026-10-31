"""F02 coverage-gap rescue append (screening arm W1, shortlist id F02).

After the baseline anchor-append channel, some query anchors (names / dates /
numbers / quoted spans) may still be uncovered by delivered evidence: the
baseline ranks candidate sources by TOTAL anchor hits, so a source covering a
rare uncovered anchor loses to sources covering many already-covered ones.
This arm appends, into slots the caller reserves, verbatim sources ranked by
UNCOVERED-anchor hits first (sweep evidence: subgoal/coverage-gap expansion in
infini-memory, evidence-grid sum/count coverage checks; shortlist F02/F15).

Same contract as anchor_channel: append-not-displace, verbatim sources only,
no model, no rewrite. Purely deterministic.
"""
from __future__ import annotations

import hashlib
import json

from .anchor_channel import _render_source, anchor_list, extract_anchors, normalized
from .domain import Evidence


def uncovered_anchors(merged, query):
    """Return query anchors not covered by any delivered evidence text."""
    anchors = anchor_list(extract_anchors(query))
    if not anchors:
        return []
    low = [a.lower() for a in anchors]
    delivered_text = "\n".join(
        hit.content for hit in merged if hit.kind != "coverage_rescue")
    delivered_low = normalized(delivered_text).lower()
    return [original for original, a in zip(anchors, low) if a not in delivered_low]


def _delivered_low(merged, by_id):
    texts = set()
    for hit in merged:
        for sid in hit.source_ids:
            source = by_id.get(sid)
            if source is not None:
                text = normalized(source.get("content"))
                if text:
                    texts.add(text.lower())
    return texts


def rescue_append(merged, sources, query, *, m, delivered_texts=None):
    """Append up to m verbatim sources ranked by uncovered-anchor hits first.

    sources: iterable of {id, content, role, timestamp} in stable store order.
    Sources already delivered (by id) or textually identical to delivered
    content are skipped, matching anchor_channel semantics.
    """
    if m <= 0 or not sources:
        return []
    anchors = anchor_list(extract_anchors(query))
    if not anchors:
        return []
    low = [a.lower() for a in anchors]
    delivered_text = "\n".join(
        hit.content for hit in merged if hit.kind != "coverage_rescue")
    delivered_low_text = normalized(delivered_text).lower()
    uncovered = [a for a in low if a not in delivered_low_text]
    if not uncovered:
        return []

    delivered = {sid for hit in merged for sid in hit.source_ids}
    by_id = {}
    for order, source in enumerate(sources):
        by_id.setdefault(source["id"], source)
    seen_text = set()
    if delivered_texts is None:
        delivered_texts = _delivered_low(merged, by_id)
    seen_text.update(delivered_texts)

    scored = []
    for order, source in enumerate(sources):
        if source["id"] in delivered:
            continue
        text = normalized(source.get("content"))
        key = text.lower()
        if not text or key in seen_text:
            continue
        total = sum(1 for a in low if a in key)
        fresh = sum(1 for a in uncovered if a in key)
        if fresh:
            seen_text.add(key)
            scored.append((-fresh, -total, order, source))
    scored.sort(key=lambda t: (t[0], t[1], t[2]))
    out = []
    for fresh, total, _, source in scored[:m]:
        out.append(Evidence(
            id="memmachine_coverage_" + hashlib.sha256(
                json.dumps([source["id"]], ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()[:24],
            content=_render_source(source), source_ids=(source["id"],),
            score=float(-fresh),
            kind="coverage_rescue",
        ))
    return out
