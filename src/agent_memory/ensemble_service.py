"""Local development deployment of the unified Add/Search ensemble."""
from __future__ import annotations

from contextlib import asynccontextmanager
import json
from pathlib import Path

from .api import create_app
from .official_service import build_official_app
from .deepseek_memory import EnsembleMemory, NativeHarness
from .native_chat import NativeChatProvider
from .hybrid_retrieval import HybridRetrieval


def load_skill(path):
    """Accept a plain skill or a saved candidate, never its JSON metadata."""
    text = Path(path).read_text(encoding="utf-8").strip()
    if Path(path).suffix.lower() == ".json":
        value = json.loads(text)
        if value.get("track") != "textual":
            raise ValueError("Textual skill required")
        text = value.get("skill")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("nonempty skill text required")
    return text


def build_ensemble_app(*, db_path, api_key, embedder, harness, ledger_path,
                       method_config=None, anchor_append_m=8):
    from .method_memmachine import _native
    _native()
    # Preserve the existing R implementation, storage identity, and native index.
    baseline_app = build_official_app(db_path=db_path, api_key=api_key, embedder=embedder,
        method="memmachine", method_config=method_config,
        anchor_append_m=anchor_append_m)
    baseline = HybridRetrieval(baseline_app.state.memory_service, anchor_limit=anchor_append_m)
    service = EnsembleMemory(baseline, harness, ledger_path=ledger_path)
    app = create_app(db_path, api_key=api_key, service=service)
    app.state.memory_service = service
    app.state.version_manifest = service.identity
    return app


def main(argv=None):
    import argparse
    import uvicorn
    from .bailian_embeddings import BailianEmbeddings, load_profile
    parser = argparse.ArgumentParser(description="Local Textual DeepSeek ensemble development API")
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument("--mini-key-file", type=Path, default=Path.home()/".config/agentmemory/credentials/closeai-mini-20261025.key")
    parser.add_argument("--embedding-profile", type=Path, default=Path("configs/embedding-bailian-v4.json"))
    parser.add_argument("--embedding-key-file", type=Path, default=Path.home()/".config/agentmemory/credentials/bailian-apikey-8122300.key")
    parser.add_argument("--runtime-dir", type=Path, default=Path("data/deepseek-runtime"))
    parser.add_argument("--skill-file", type=Path)
    parser.add_argument("--max-native-calls", type=int, default=64)
    parser.add_argument("--max-results", type=int, default=16)
    parser.add_argument("--expand-context", type=int, default=0)
    parser.add_argument("--port", type=int, default=18086)
    args = parser.parse_args(argv)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    profile = load_profile(args.embedding_profile)
    profile["api_key_file"] = str(args.embedding_key_file)
    if profile["dimensions"] != 1024 or profile["model"] != "text-embedding-v4":
        raise ValueError("this recipe requires text-embedding-v4/1024")
    embeddings = BailianEmbeddings(profile, user_confirmed=True)
    provider = NativeChatProvider(args.run_dir/"native-calls.sqlite3", key_file=args.mini_key_file,
        max_calls=args.max_native_calls, max_output_tokens=args.max_native_calls*1536,
        max_input_estimate=args.max_native_calls*180000, allow_dated_snapshot=True)
    options = {"skill": load_skill(args.skill_file)} if args.skill_file else {}
    harness = NativeHarness(provider=provider, runtime_dir=args.runtime_dir,
                            artifact_dir=args.run_dir/"native", **options)
    key = args.api_key_file.read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError("API credential file is empty")
    app = build_ensemble_app(db_path=args.db, api_key=key, embedder=embeddings,
        harness=harness, ledger_path=args.run_dir/"search.sqlite3",
        method_config={"max_results": args.max_results, "expand_context": args.expand_context})
    (args.run_dir/"version-manifest.json").write_text(json.dumps(app.state.version_manifest,
        ensure_ascii=False, indent=2), encoding="utf-8")
    @asynccontextmanager
    async def lifetime(app):
        try:
            yield
        finally:
            app.state.memory_service.close()
            provider.close()
            embeddings.store.close()
    app.router.lifespan_context = lifetime
    uvicorn.run(app, host="127.0.0.1", port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
