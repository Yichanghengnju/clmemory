"""Reuse the established Textual hybrid16 + neighbors8 + anchor8 recipe.

Search-only composition over the unchanged native source/index store. The
selection functions are shared with the existing frozen baseline builder.
"""
from .domain import Evidence
from .anchor_channel import append_evidence
from .r_source_packing import neighbors, pack

VERSION = "textual-hybrid-neighbors-anchor-v1"


class HybridRetrieval:
    def __init__(self, native, *, neighbor_limit=8, anchor_limit=8):
        self.native, self._store = native, native._store
        self.neighbor_limit, self.anchor_limit = neighbor_limit, anchor_limit
        self.config = native.config

    @property
    def identity(self):
        return {"name": VERSION, "native": self.native.identity, "neighbors": self.neighbor_limit,
                "anchor": self.anchor_limit, "evidence_tokens": 16000, "source_identity": "native-stable-id"}

    def add(self, request):
        return self.native.add(request)

    def _search(self, request, source_ids=None):
        sources = self._store.all_sources(request.user_id)
        if source_ids is not None:
            sources = [s for s in sources if s["id"] in source_ids]
        # The existing packer records these fields for provenance; no source
        # text is transformed. Official Add messages are whole source spans.
        sources = [{**s, "original_message_index": i, "char_span": [0,len(s["content"])]}
                   for i,s in enumerate(sources)]
        by_id = {s["id"]: s for s in sources}
        raw = (self.native.search(request) if source_ids is None
               else self.native.search_visible(request, source_ids))
        ids = list(dict.fromkeys(s for h in raw for s in h.source_ids if s in by_id))
        ids += neighbors(ids, sources, self.neighbor_limit)
        first = pack(ids, by_id)
        hits = [Evidence(**{**h,"source_ids":tuple(h["source_ids"])}) for h in first["evidence"]]
        anchors = append_evidence(hits,sources,request.query,m=self.anchor_limit)
        ids = [h.id for h in hits] + [s for h in anchors for s in h.source_ids]
        final = pack(ids,by_id,max_items=request.top_k)
        return [Evidence(**{**h,"source_ids":tuple(h["source_ids"])}) for h in final["evidence"]]

    def search(self, request):
        return self._search(request)

    def search_visible(self, request, source_ids):
        return self._search(request,source_ids)

    def close(self):
        self.native.close()
