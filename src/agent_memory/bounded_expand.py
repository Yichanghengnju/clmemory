"""Bounded neighbor expansion for Textual submission surface A.

Keeps native MemMachine expand_context=0. After RRF core hits, appends up to
`budget` verbatim neighbor sources: session-adjacent first, then temporal
neighbors in the same session (roughly 2/3 before, 1/3 after by timestamp/seq).
Core ids are never displaced; when truncating to top_k, neighborhood is dropped
before core. Neighbor scores are capped by neighbor_score_ratio (memfusion-style)
so they cannot outrank cores under score re-sort. Zero generative LLM.
"""
from __future__ import annotations


def _order_index(sources: list[dict]) -> dict[str, int]:
    return {row["id"]: i for i, row in enumerate(sources)}


def session_adjacent(core_ids: list[str], sources: list[dict], limit: int) -> list[str]:
    """Same-session ±1 by storage order (existing hybrid neighbors semantics)."""
    if limit <= 0 or not core_ids:
        return []
    by = _order_index(sources)
    seen = set(core_ids)
    extra: list[str] = []
    for sid in core_ids:
        i = by.get(sid)
        if i is None:
            continue
        for j in (i - 1, i + 1):
            if 0 <= j < len(sources):
                cand = sources[j]
                if cand["session_id"] == sources[i]["session_id"] and cand["id"] not in seen:
                    extra.append(cand["id"])
                    seen.add(cand["id"])
                    if len(extra) >= limit:
                        return extra
    return extra


def _time_key(row: dict) -> tuple:
    # Prefer explicit timestamp; fall back to storage order via seq/position.
    ts = row.get("timestamp")
    seq = row.get("seq", row.get("position", 0))
    return (ts if ts is not None else -1, seq if seq is not None else 0)


def temporal_neighbors(core_ids: list[str], sources: list[dict], limit: int) -> list[str]:
    """Same-session temporal neighborhood; fill remaining budget after adjacency.

    Within each core hit's session, take earlier messages first (≈2/3 of the
    remaining slots for that core) then later ones (≈1/3), walking by
    timestamp/seq. Skips ids already selected.
    """
    if limit <= 0 or not core_ids:
        return []
    by_session: dict[str, list[dict]] = {}
    for row in sources:
        by_session.setdefault(row["session_id"], []).append(row)
    for rows in by_session.values():
        rows.sort(key=_time_key)

    id_to_row = {row["id"]: row for row in sources}
    seen = set(core_ids)
    extra: list[str] = []
    for sid in core_ids:
        if len(extra) >= limit:
            break
        row = id_to_row.get(sid)
        if row is None:
            continue
        session_rows = by_session.get(row["session_id"], [])
        try:
            idx = next(i for i, r in enumerate(session_rows) if r["id"] == sid)
        except StopIteration:
            continue
        remain = limit - len(extra)
        before_n = max(1, (remain * 2) // 3) if remain >= 2 else remain
        after_n = remain - before_n
        for r in reversed(session_rows[:idx]):
            if before_n <= 0 or len(extra) >= limit:
                break
            if r["id"] in seen:
                continue
            extra.append(r["id"])
            seen.add(r["id"])
            before_n -= 1
        for r in session_rows[idx + 1 :]:
            if after_n <= 0 or len(extra) >= limit:
                break
            if r["id"] in seen:
                continue
            extra.append(r["id"])
            seen.add(r["id"])
            after_n -= 1
    return extra


def expand_source_ids(core_ids: list[str], sources: list[dict], budget: int = 8) -> list[str]:
    """Return neighbor ids only (not including core), length ≤ budget."""
    if budget <= 0:
        return []
    adj = session_adjacent(core_ids, sources, budget)
    remain = budget - len(adj)
    if remain <= 0:
        return adj
    temp = temporal_neighbors(core_ids + adj, sources, remain)
    return adj + temp


def merge_core_then_expand(
    core_hits: list,
    expand_ids: list[str],
    sources_by_id: dict,
    *,
    top_k: int,
    render,
    neighbor_score_ratio: float = 0.3,
) -> list:
    """Keep core order; append expand; drop expand before core when truncating.

    `render(source) -> content str` must return verbatim evidence content.
    Neighbor scores are capped to max(core_score)*neighbor_score_ratio so they
    cannot outrank cores if a later stage re-sorts by score (memfusion-style).
    """
    from .domain import Evidence

    if not (0.0 <= float(neighbor_score_ratio) <= 1.0):
        raise ValueError("neighbor_score_ratio must be in [0, 1]")
    core = list(core_hits[:top_k])
    slots = top_k - len(core)
    if slots <= 0 or not expand_ids:
        return core
    out = list(core)
    used = {h.id for h in core}
    core_max = max((float(getattr(h, "score", 0.0) or 0.0) for h in core), default=0.0)
    for sid in expand_ids:
        if slots <= 0:
            break
        if sid in used or sid not in sources_by_id:
            continue
        source = sources_by_id[sid]
        text = render(source)
        base = core_max * float(neighbor_score_ratio)
        score = float(max(0.0, base - len(out) * 1e-6))
        out.append(
            Evidence(
                id=sid,
                content=text,
                source_ids=(sid,),
                score=score,
                kind="source",
            )
        )
        used.add(sid)
        slots -= 1
    return out


# --- P0-2 / Option B: guarded temporal soft-boost (score/order only; zero generative LLM) ---

# Conservative bilingual cues; substring match (ASCII case-insensitive).
TEMPORAL_SOFT_BOOST_CUES: tuple[str, ...] = (
    "now",
    "current",
    "currently",
    "latest",
    "recent",
    "recently",
    "updated",
    "update",
    "moved",
    "changed",
    "after",
    "today",
    "nowadays",
    "最近",
    "现在",
    "目前",
    "当前",
    "最新",
    "更新后",
    "更新",
    "搬到",
    "搬去",
    "改成",
    "变成",
)

# Mild ratios when the flag is ON. Official default keeps the flag OFF.
TEMPORAL_SOFT_BOOST_NEWER_RATIO = 1.15
TEMPORAL_SOFT_BOOST_OLDER_RATIO = 0.85


def query_has_temporal_cue(query: str, cues: tuple[str, ...] = TEMPORAL_SOFT_BOOST_CUES) -> bool:
    """True when the query carries a conservative temporal/update cue."""
    if not query:
        return False
    q = query.casefold()
    for cue in cues:
        if not cue:
            continue
        if cue.casefold() in q or cue in query:
            return True
    return False


def _normalize_tokens(text: str) -> list[str]:
    """Deterministic tokens: ASCII words + individual CJK chars."""
    import re

    if not text:
        return []
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.casefold())


def near_duplicate_contents(
    a: str,
    b: str,
    *,
    min_jaccard: float = 0.4,
    min_shared: int = 2,
) -> bool:
    """Near-duplicate / attribute-update pair via token Jaccard (e.g. Beijing→Shanghai).

    Requires shared skeleton tokens and at least one differing token on each side
    (or identical token sets). Deterministic; no LLM.
    """
    ta = set(_normalize_tokens(a))
    tb = set(_normalize_tokens(b))
    if not ta or not tb:
        return False
    inter = ta & tb
    if len(inter) < min_shared:
        return False
    union = ta | tb
    jaccard = len(inter) / len(union)
    if jaccard < float(min_jaccard):
        return False
    if ta == tb:
        return True
    return bool(ta - tb) and bool(tb - ta)


def _evidence_time_key(hit, sources_by_id: dict | None) -> tuple:
    """Prefer source timestamp/seq; fall back to stable id for determinism."""
    sid = getattr(hit, "id", None)
    row = (sources_by_id or {}).get(sid) if sid is not None else None
    if isinstance(row, dict):
        ts = row.get("timestamp")
        seq = row.get("seq", row.get("position", 0))
        return (ts if ts is not None else -1, seq if seq is not None else 0, str(sid))
    return (-1, 0, str(sid) if sid is not None else "")


def _near_dup_groups(hits: list, *, min_jaccard: float = 0.4) -> list[list[int]]:
    """Union-find groups of near-duplicate contents among hits (index lists)."""
    n = len(hits)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    contents = [getattr(h, "content", "") or "" for h in hits]
    for i in range(n):
        for j in range(i + 1, n):
            if near_duplicate_contents(contents[i], contents[j], min_jaccard=min_jaccard):
                union(i, j)
    buckets: dict[int, list[int]] = {}
    for i in range(n):
        buckets.setdefault(find(i), []).append(i)
    return [idxs for idxs in buckets.values() if len(idxs) >= 2]


def apply_temporal_soft_boost(
    hits: list,
    query: str,
    *,
    enabled: bool = False,
    newer_ratio: float = TEMPORAL_SOFT_BOOST_NEWER_RATIO,
    older_ratio: float = TEMPORAL_SOFT_BOOST_OLDER_RATIO,
    sources_by_id: dict | None = None,
    min_jaccard: float = 0.4,
) -> list:
    """Soft-rerank only: boost newer / demote older near-dups when query has cues.

    - No-op when disabled, no temporal cue, or no near-duplicate groups.
    - Never deletes hits; never rewrites Evidence.content (new Evidence, same content).
    - Re-sorts by adjusted score descending; ties broken by original order (stable).
    """
    from .domain import Evidence

    if not enabled or not hits:
        return hits
    if not (newer_ratio > 0 and older_ratio > 0):
        raise ValueError("temporal soft-boost ratios must be positive")
    if not query_has_temporal_cue(query):
        return hits
    groups = _near_dup_groups(hits, min_jaccard=min_jaccard)
    if not groups:
        return hits

    mult = [1.0] * len(hits)
    for idxs in groups:
        ranked = sorted(
            idxs,
            key=lambda i: _evidence_time_key(hits[i], sources_by_id),
            reverse=True,
        )
        newest = ranked[0]
        mult[newest] = float(newer_ratio)
        for i in ranked[1:]:
            mult[i] = float(older_ratio)

    adjusted: list = []
    for i, h in enumerate(hits):
        new_score = float(getattr(h, "score", 0.0) or 0.0) * mult[i]
        if isinstance(h, Evidence):
            adjusted.append(
                Evidence(
                    id=h.id,
                    content=h.content,
                    source_ids=h.source_ids,
                    score=new_score,
                    kind=h.kind,
                )
            )
        else:
            adjusted.append(h)
            try:
                object.__setattr__(h, "score", new_score)
            except Exception:
                pass

    decorated = list(enumerate(adjusted))
    decorated.sort(key=lambda pair: (-float(getattr(pair[1], "score", 0.0) or 0.0), pair[0]))
    return [h for _, h in decorated]


# --- P0-3: session-level RRF (ReFind de-agent; score/order only; zero generative LLM) ---

SESSION_RRF_K = 60


def _hit_session_id(hit, sources_by_id: dict | None) -> str:
    """Resolve session_id from sources; fall back to a per-hit synthetic key."""
    sid = getattr(hit, "id", None)
    row = (sources_by_id or {}).get(sid) if sid is not None else None
    if isinstance(row, dict):
        sess = row.get("session_id")
        if sess is not None and sess != "":
            return str(sess)
    return f"__nosession__:{sid}"


def apply_session_rrf(
    hits: list,
    *,
    enabled: bool = False,
    rrf_k: int = SESSION_RRF_K,
    sources_by_id: dict | None = None,
) -> list:
    """ReFind-style chunk-rank + session-sum-rank RRF on already-ranked hits.

    - No-op when disabled or empty.
    - Session score = sum of current hit scores sharing session_id.
    - fused = 1/(k + chunk_rank) + 1/(k + session_rank); rewrite score; re-sort.
    - Never deletes hits; never rewrites Evidence.content.
    - Intended placement: after native BM25+dense RRF, before bounded expand.
    """
    from .domain import Evidence

    if not enabled or not hits:
        return hits
    k = int(rrf_k)
    if k < 1:
        raise ValueError("session_rrf rrf_k must be >= 1")

    chunk_ranks = {i: rank for rank, i in enumerate(range(len(hits)), 1)}

    session_scores: dict[str, float] = {}
    session_of: list[str] = []
    for h in hits:
        sess = _hit_session_id(h, sources_by_id)
        session_of.append(sess)
        session_scores[sess] = session_scores.get(sess, 0.0) + float(
            getattr(h, "score", 0.0) or 0.0
        )

    session_order = sorted(session_scores, key=lambda s: (-session_scores[s], s))
    session_ranks = {s: rank for rank, s in enumerate(session_order, 1)}

    fused_scores: list[float] = []
    for i, _h in enumerate(hits):
        fused_scores.append(
            1.0 / (k + chunk_ranks[i]) + 1.0 / (k + session_ranks[session_of[i]])
        )

    adjusted: list = []
    for i, h in enumerate(hits):
        fused = fused_scores[i]
        if isinstance(h, Evidence):
            adjusted.append(
                Evidence(
                    id=h.id,
                    content=h.content,
                    source_ids=h.source_ids,
                    score=fused,
                    kind=h.kind,
                )
            )
        else:
            adjusted.append(h)
            try:
                object.__setattr__(h, "score", fused)
            except Exception:
                pass

    decorated = list(enumerate(adjusted))
    decorated.sort(key=lambda pair: (-fused_scores[pair[0]], pair[0]))
    return [h for _, h in decorated]
