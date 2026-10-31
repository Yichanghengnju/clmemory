"""W1 screening arms (shortlist F02/F03/F04) on the MemMachine event core.

Direct MemMachineMemory subclass implementing the R-recipe Search pipeline
(native RRF -> optional session RRF -> bounded neighbor expand -> optional
temporal soft boost -> anchor append) WITHOUT importing the grep recipe module,
so the screening lane runs in the native venv (no tiktoken dependency). The
inlined pipeline calls the same bounded_expand / anchor_channel functions with
the same default parameters as MemMachineGrepAppendMemory v6; with all W1 flags
off and the optional stages off, behavior equals that recipe.

Arms (deterministic, zero-model, append/order-only):
- coverage_rescue (F02): reserve tail slots for sources covering query anchors
  the delivered evidence does not yet cover.
- complementary_tail (F04): greedy same-scale tail selection (anchor ratio +
  new-query-word coverage - token-Jaccard redundancy); cores protected.
- date_annotations (F03): clearly-marked derived date annotations per source
  header; bodies byte-identical; unknown timestamps never resolved.

Add semantics inherited unchanged. The W1 flag block is part of identity; the
Phase B store is opened by passing the original vanilla identity as
storage_identity so MethodStore validation is unchanged.
"""
from __future__ import annotations

from pathlib import Path

from .anchor_channel import _render_source, append_evidence, normalized
from .bounded_expand import (
    SESSION_RRF_K, TEMPORAL_SOFT_BOOST_NEWER_RATIO, TEMPORAL_SOFT_BOOST_OLDER_RATIO,
    apply_session_rrf, apply_temporal_soft_boost, expand_source_ids,
    merge_core_then_expand)
from .complementary_tail import rule_quota_sources, select_tail
from .coverage_rescue import rescue_append, uncovered_anchors
from .date_annotations import annotate_evidence
from .method_memmachine import MemMachineMemory, _index_lock


def _body_texts(hits):
    texts = []
    for hit in hits:
        lines = [ln for ln in hit.content.splitlines()
                 if not ln.startswith("[source=") and not ln.startswith("[dates ")]
        text = normalized("\n".join(lines)).lower()
        if text:
            texts.append(text)
    return texts


class MemMachineW1Memory(MemMachineMemory):
    def __init__(self, db_path: str | Path, provider=None, embedder=None,
                 configdict=None, *, config=None,
                 anchor_append_m: int = 8, neighbor_budget: int = 8,
                 neighbor_score_ratio: float = 0.3,
                 session_rrf: bool = False, session_rrf_k: int = SESSION_RRF_K,
                 temporal_soft_boost: bool = False,
                 temporal_soft_boost_newer_ratio: float = TEMPORAL_SOFT_BOOST_NEWER_RATIO,
                 temporal_soft_boost_older_ratio: float = TEMPORAL_SOFT_BOOST_OLDER_RATIO,
                 coverage_rescue: bool = False, complementary_tail: bool = False,
                 combined_tail: bool = False,
                 rule_quota: bool = False, rule_quota_m: int = 8,
                 numeric_anchor_boost: bool = False, numeric_boost_factor: float = 1.5,
                 date_annotations: bool = False, date_annotation_notes: int = 4,
                 complementary_coverage_weight: float = 0.14,
                 complementary_redundancy_weight: float = 0.12,
                 storage_identity=None):
        if type(anchor_append_m) is not int or anchor_append_m < 0:
            raise ValueError("anchor_append_m must be a nonnegative integer")
        if type(neighbor_budget) is not int or neighbor_budget < 0:
            raise ValueError("neighbor_budget must be a nonnegative integer")
        if not (0.0 <= float(neighbor_score_ratio) <= 1.0):
            raise ValueError("neighbor_score_ratio must be in [0, 1]")
        self._anchor_append_m = anchor_append_m
        self._neighbor_budget = neighbor_budget
        self._neighbor_score_ratio = float(neighbor_score_ratio)
        self._session_rrf = bool(session_rrf)
        self._session_rrf_k = int(session_rrf_k)
        self._temporal_soft_boost = bool(temporal_soft_boost)
        self._temporal_soft_boost_newer_ratio = float(temporal_soft_boost_newer_ratio)
        self._temporal_soft_boost_older_ratio = float(temporal_soft_boost_older_ratio)
        self._coverage_rescue = bool(coverage_rescue)
        self._complementary_tail = bool(complementary_tail)
        self._combined_tail = bool(combined_tail)
        if self._combined_tail and (self._complementary_tail or not self._coverage_rescue):
            raise ValueError("combined_tail requires coverage_rescue and excludes complementary_tail")
        self._rule_quota = bool(rule_quota)
        if type(rule_quota_m) is not int or rule_quota_m < 0:
            raise ValueError("rule_quota_m must be a nonnegative integer")
        self._rule_quota_m = rule_quota_m
        self._numeric_anchor_boost = bool(numeric_anchor_boost)
        if float(numeric_boost_factor) < 1.0:
            raise ValueError("numeric_boost_factor must be >= 1.0")
        self._numeric_boost_factor = float(numeric_boost_factor)
        self._date_annotations = bool(date_annotations)
        if type(date_annotation_notes) is not int or date_annotation_notes < 0:
            raise ValueError("date_annotation_notes must be a nonnegative integer")
        self._date_annotation_notes = date_annotation_notes
        if not (0.0 <= float(complementary_coverage_weight) <= 1.0
                and 0.0 <= float(complementary_redundancy_weight) <= 1.0):
            raise ValueError("complementary weights must be in [0, 1]")
        self._complementary_coverage_weight = float(complementary_coverage_weight)
        self._complementary_redundancy_weight = float(complementary_redundancy_weight)
        super().__init__(db_path, provider=provider, embedder=embedder,
                         configdict=configdict, config=config,
                         storage_identity=storage_identity)

    @property
    def identity(self):
        base = MemMachineMemory.identity.fget(self)
        return {**base,
                "name": "memmachine-w1-screening", "version": 1,
                "adaptation": base.get("adaptation", "") +
                              "; R-recipe pipeline (neighbor expand + anchor append, inlined)"
                              "; W1 screening arms F02/F03/F04 (flag-gated)",
                "r_recipe": {"anchor_append_m": self._anchor_append_m,
                             "neighbor_budget": self._neighbor_budget,
                             "neighbor_score_ratio": self._neighbor_score_ratio,
                             "session_rrf": self._session_rrf,
                             "temporal_soft_boost": self._temporal_soft_boost},
                "w1_screening": {"coverage_rescue": self._coverage_rescue,
                                 "complementary_tail": self._complementary_tail,
                                 "combined_tail": self._combined_tail,
                                 "rule_quota": self._rule_quota,
                                 "rule_quota_m": self._rule_quota_m,
                                 "numeric_anchor_boost": self._numeric_anchor_boost,
                                 "numeric_boost_factor": self._numeric_boost_factor,
                                 "date_annotations": self._date_annotations,
                                 "date_annotation_notes": self._date_annotation_notes,
                                 "complementary_weights": [self._complementary_coverage_weight,
                                                           self._complementary_redundancy_weight]}}

    def _arms_on(self):
        return (self._coverage_rescue or self._complementary_tail or self._date_annotations
                or self._rule_quota or self._numeric_anchor_boost)

    def _with_expand(self, request, hits, sources):
        # _search_core hands over a dict keyed by id (what _query expects);
        # the expand/append helpers want the source rows themselves.
        source_rows = list(sources.values()) if isinstance(sources, dict) else list(sources)
        top_k = request.top_k
        if len(hits) > top_k:
            raise ValueError("Base MemMachine returned more than the requested top_k")
        by_id = {s["id"]: s for s in source_rows}
        hits = apply_session_rrf(hits, enabled=self._session_rrf,
                                 rrf_k=self._session_rrf_k, sources_by_id=by_id)
        core_ids = [h.id for h in hits]
        extras = expand_source_ids(core_ids, source_rows, budget=self._neighbor_budget)
        merged = merge_core_then_expand(hits, extras, by_id, top_k=top_k,
                                        render=_render_source,
                                        neighbor_score_ratio=self._neighbor_score_ratio)
        merged = apply_temporal_soft_boost(merged, request.query,
                                           enabled=self._temporal_soft_boost,
                                           newer_ratio=self._temporal_soft_boost_newer_ratio,
                                           older_ratio=self._temporal_soft_boost_older_ratio,
                                           sources_by_id=by_id)
        remaining = top_k - len(merged)
        if self._complementary_tail and not self._combined_tail:
            # W1 f04 attribution arm: the complementary selector replaces the
            # anchor machinery entirely (this is why W1 "combo" never stacked f02).
            reserved = (rule_quota_sources(source_rows, request.query, _body_texts(merged),
                                           m=self._rule_quota_m)
                        if self._rule_quota else [])
            tail = select_tail(source_rows, request.query, _body_texts(merged),
                               limit=max(0, remaining - len(reserved)),
                               coverage_weight=self._complementary_coverage_weight,
                               redundancy_weight=self._complementary_redundancy_weight,
                               boost_categories=("numbers", "quoted") if self._numeric_anchor_boost else None,
                               boost_factor=self._numeric_boost_factor)
            merged = merged + tail + reserved
        elif self._combined_tail:
            # Proper stack: rescue reserve first (uncovered-anchor coverage),
            # complementary selector fills the rest; no plain anchor append.
            reserve = 0
            if remaining > 0:
                uncovered = uncovered_anchors(merged, request.query)
                reserve = min(len(uncovered), remaining)
            m_rest = max(0, remaining - reserve)
            if m_rest > 0:
                merged = merged + select_tail(source_rows, request.query, _body_texts(merged),
                                              limit=m_rest,
                                              coverage_weight=self._complementary_coverage_weight,
                                              redundancy_weight=self._complementary_redundancy_weight)
            if reserve > 0:
                merged = merged + rescue_append(merged, source_rows, request.query,
                                                m=min(reserve, top_k - len(merged)))
        else:
            m = min(self._anchor_append_m, max(0, remaining))
            reserve = 0
            if self._coverage_rescue and m > 0:
                uncovered = uncovered_anchors(merged, request.query)
                reserve = min(len(uncovered), m)
                m -= reserve
            if m > 0:
                merged = merged + append_evidence(merged, source_rows, request.query, m=m)
            if reserve > 0:
                merged = merged + rescue_append(merged, source_rows, request.query,
                                                m=min(reserve, top_k - len(merged)))
        if self._date_annotations:
            merged = annotate_evidence(merged, max_notes=self._date_annotation_notes)
        return merged[:top_k] if len(merged) > top_k else merged

    def _search_core(self, request, source_ids=None):
        request = type(request).model_validate(request.model_dump())
        with _index_lock(self._user_path(request.user_id) / "index.lock"):
            if source_ids is None:
                sources = {s["id"]: s for s in self._store.all_sources(request.user_id)}
            else:
                sources = {s["id"]: s for s in self._store.all_sources(request.user_id)
                           if s["id"] in source_ids}
            if not sources:
                return [], sources
            hits = self._query_native(request, sources)
            return hits, sources

    def _query_native(self, request, sources):
        import asyncio
        self._check_identity()
        result = asyncio.run(self._query(request, sources))
        self._check_identity()
        return result

    def search(self, request):
        from .domain import SearchRequest
        request = SearchRequest.model_validate(request.model_dump())
        hits, sources = self._search_core(request)
        if not hits:
            return []
        return self._with_expand(request, hits, sources)

    def search_visible(self, request, source_ids):
        from .domain import SearchRequest
        request = SearchRequest.model_validate(request.model_dump())
        hits, sources = self._search_core(request, set(source_ids))
        if not hits:
            return []
        return self._with_expand(request, hits, sources)
