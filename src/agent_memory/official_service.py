"""Official AML endpoint assembly (deployment-design-20261024.md).

Builds the public Add/Search service on the shared api.create_app core with
the MemMachine recipe (verbatim Add; dense candidates with BM25/RRF ranking,
optionally followed by source-bound GREP). Zero-LLM by construction: no provider is wired in the official
profile; embeddings, when enabled, must be text-embedding-v4 per the
Open-source Methods model clause (rerankers unrestricted; our rerank is
score-free ordering).

Version manifest: written at startup; the declared evaluation version must
match this manifest (official version-freeze contract).
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

from .api import create_app

SERVICE_VERSION = "official-endpoint/2026-10-25-r2"


class DrillHashEmbedder:
    """Deterministic bag-of-words hash embedder for the ZERO-NETWORK drill
    profile only. Never valid for formal evaluation: dense-channel semantics
    are meaningless; the drill exercises the HTTP contract and BM25 path."""

    dimensions = 384

    @property
    def identity(self):
        return {"implementation": "drill-hash-embedder", "dimensions": self.dimensions,
                "formal_use": False,
                "note": "zero-network drill profile; formal embedding must be text-embedding-v4"}

    @staticmethod
    def _vec(text: str):
        import hashlib
        import math
        import re as _re
        v = [0.0] * DrillHashEmbedder.dimensions
        for w in _re.findall(r"[a-z0-9]+", str(text).lower()):
            h = int(hashlib.sha256(w.encode()).hexdigest()[:8], 16)
            v[h % DrillHashEmbedder.dimensions] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, inputs):
        return [self._vec(t) for t in inputs]

    def embed_query(self, text):
        return self._vec(text)


def _code_fingerprint() -> str:
    root = Path(__file__).parent
    h = hashlib.sha256()
    for name in ("api.py", "domain.py", "memory.py", "method_memmachine.py",
                 "method_store.py", "official_service.py", "method_memmachine_grep.py",
                 "anchor_channel.py", "bailian_embeddings.py"):
        p = root / name
        if p.exists():
            h.update(name.encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def build_official_app(*, db_path: str | Path, api_key: str | None,
                       embedder=None, manifest_dir: str | Path | None = None,
                       extra_meta: dict | None = None, method: str = "memmachine",
                       method_config: dict | None = None, anchor_append_m: int = 8):
    """Resolve the explicit embedder before opening a method/index.

    None is unsupported; DrillHashEmbedder is a synthetic dense drill, never
    a lexical-only or formal v4 profile. Existing index identity is checked by
    the method store; this assembly never migrates or falls back.
    """
    from .method_memmachine import MemMachineMemory

    if method not in {"memmachine", "memmachine_grep_append"}:
        raise ValueError("Unknown official memory method")
    if embedder is None:
        raise ValueError("An explicit embedder is required; None is not lexical-only")
    resolved = dict(method_config or {})
    dimensions = getattr(embedder, "dimensions", None)
    identity = getattr(embedder, "identity", None)
    if dimensions is None and isinstance(identity, dict):
        dimensions = identity.get("dimensions")
    if type(dimensions) is not int or dimensions < 1:
        raise ValueError("embedder must declare positive integer dimensions")
    if "embedding_dimensions" in resolved and (
        type(resolved["embedding_dimensions"]) is not int
        or resolved["embedding_dimensions"] != dimensions
    ):
        raise ValueError("Configured embedding_dimensions differs from the embedder")
    resolved["embedding_dimensions"] = dimensions
    if method == "memmachine_grep_append":
        from .method_memmachine_grep import MemMachineGrepAppendMemory
        service = MemMachineGrepAppendMemory(
            db_path, embedder=embedder, config=resolved, anchor_append_m=anchor_append_m)
    else:
        service = MemMachineMemory(db_path, embedder=embedder, config=resolved)
    app = create_app(db_path, service=service, api_key=api_key,
                     enforce_search_limit=True)
    app.state.official_service_version = SERVICE_VERSION
    app.state.memory_service = service
    if manifest_dir is not None:
        manifest = {
            "service_version": SERVICE_VERSION,
            "code_fingerprint": _code_fingerprint(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "method": "memmachine-verbatim" if method == "memmachine" else method,
            "method_identity": service.identity,
            "resolved_method_config": dict(service.config),
            "embedding_dimensions": dimensions,
            "embedding_mode": "synthetic-hash-drill" if isinstance(embedder, DrillHashEmbedder) else "explicit-embedder",
            "embedding": json.dumps(getattr(embedder, "identity", None) or "unspecified",
                                    sort_keys=True) if embedder is not None else "none",
            "llm_calls_in_add_search": 0,
            "top_k_contract": "<= request top_k (formal 100)",
            "auth": "Bearer | Token | X-Api-Key",
            "extra": extra_meta or {},
        }
        mdir = Path(manifest_dir)
        mdir.mkdir(parents=True, exist_ok=True)
        (mdir / "version-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        app.state.version_manifest = manifest
    return app


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run the official AML endpoint")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--db", required=True)
    parser.add_argument("--api-key-file", required=True,
                        help="file containing one line: the API key (no trailing spaces)")
    parser.add_argument("--manifest-dir", default="run/official")
    parser.add_argument("--embedding", choices=["off", "hash-drill", "bailian-v4"], default="off",
                        help="hash-drill (legacy alias: off) = synthetic dense vectors, not lexical-only; "
                             "bailian-v4 = explicit text-embedding-v4 profile")
    parser.add_argument("--method", choices=["memmachine", "memmachine_grep_append"], default="memmachine")
    parser.add_argument("--max-results", type=int, default=8)
    parser.add_argument("--anchor-append-m", type=int, default=8)
    parser.add_argument("--dedup-overfetch", type=int, default=4)
    parser.add_argument("--expand-context", type=int, default=0)
    parser.add_argument("--embedding-profile",
                        help=" Bailian text-embedding-v4 profile JSON path (credentials external)")
    parser.add_argument("--confirm-bailian", action="store_true",
                        help="explicit runtime confirmation for Bailian usage (project rule)")
    args = parser.parse_args(argv)

    key = Path(args.api_key_file).read_text(encoding="utf-8").strip()
    if not key:
        raise SystemExit("api key file is empty")
    if args.embedding in {"off", "hash-drill"}:
        embedder = DrillHashEmbedder()
    elif args.embedding == "bailian-v4":
        if not args.embedding_profile or not args.confirm_bailian:
            raise SystemExit("bailian-v4 requires --embedding-profile and --confirm-bailian "
                             "(explicit runtime confirmation; no standing authorization)")
        from .bailian_embeddings import BailianEmbeddings, load_profile
        embedder = BailianEmbeddings(load_profile(args.embedding_profile),
                                     user_confirmed=True)
    app = build_official_app(db_path=args.db, api_key=key, embedder=embedder,
                             manifest_dir=args.manifest_dir,
                             method=args.method, anchor_append_m=args.anchor_append_m,
                             method_config={"max_results": args.max_results,
                                            "dedup_overfetch": args.dedup_overfetch,
                                            "expand_context": args.expand_context},
                             extra_meta={"argv": vars(args), "sys_argv0": sys.argv[0]})
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
