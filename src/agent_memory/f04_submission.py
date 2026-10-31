"""Textual f04-only: frozen W1 native16 plus static complementary tail."""
import argparse
from contextlib import asynccontextmanager
from pathlib import Path
import hashlib
import json

from .api import create_app
from .bailian_embeddings import BailianEmbeddings, load_profile
from .memmachine_submission import SerializedEmbeddings
from .method_memmachine import _index_lock
from .method_memmachine_w1 import MemMachineW1Memory

VERSION = 'textual-f04-only-v1.0'
CONFIG = {'embedding_dimensions': 1024, 'deriver': 'whole_text', 'segmenter': 'passthrough',
          'rrf_k': 60, 'bm25_tokenizer': 'simple', 'dedup_overfetch': 4,
          'expand_context': 0, 'max_results': 16}


class ExactQuerySingleFlight:
    """Serialize identical HTTP searches; successful embeddings use their durable cache."""
    def __init__(self, memory, directory):
        self.memory, self.directory = memory, Path(directory)

    @property
    def identity(self):
        return self.memory.identity

    def add(self, request):
        return self.memory.add(request)

    def search(self, request):
        key = hashlib.sha256(json.dumps(request.model_dump(), sort_keys=True,
            ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
        with _index_lock(self.directory / (key + '.lock')):
            return self.memory.search(request)

    def close(self):
        self.memory.close()


def build_app(*, db_path, api_key, embedder, run_dir):
    if not api_key:
        raise ValueError('A nonempty private service token is required')
    if embedder.dimensions != 1024:
        raise ValueError('The frozen native backend requires 1024 dimensions')
    run = Path(run_dir)
    run.mkdir(parents=True, exist_ok=True)
    embeddings = SerializedEmbeddings(embedder)
    baseline = MemMachineW1Memory(
        db_path, embedder=embeddings, config=CONFIG, complementary_tail=True,
        coverage_rescue=False, combined_tail=False, rule_quota=False,
        numeric_anchor_boost=False, date_annotations=False,
        session_rrf=False, temporal_soft_boost=False,
        complementary_coverage_weight=0.14, complementary_redundancy_weight=0.12)
    memory = ExactQuerySingleFlight(baseline, run / 'http-singleflight')
    app = create_app(db_path, service=memory, api_key=api_key, enforce_search_limit=True)

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            memory.close()
            embeddings.close()

    app.router.lifespan_context = lifespan
    source = Path(__file__).parent
    manifest = {
        'schema': 'textual-f04-only-version-v1', 'system': 'clmemory',
        'version': VERSION, 'track': 'textual', 'entry': 'agent_memory.f04_submission',
        'method': memory.identity, 'embedding': embeddings.specification,
        'f04': True, 'H': False, 'F': False, 'S14': False, 'KD': False, 'AgentDistill': False,
        'llm_calls_in_add': 0, 'llm_calls_in_search': 0, 'mini_calls': 0,
        'native_core_cap': 16, 'requested_top_k_min': 1, 'requested_top_k_max': 100,
        'static_score': {'coverage_weight': 0.14, 'redundancy_weight': 0.12,
                         'context': 'initial native R context; static scores, no recomputation'},
        'f04_scoring': 'static coverage and redundancy against initial R context',
        'f04_compatibility': 'frozen W1 implementation; no neighbor-ID fix or dynamic MMR',
        'f04_known_limit': 'configured neighbor8 may not fire because core Evidence.id differs from source_id',
        'source_delivery': 'original role and timestamp headers plus verbatim source content',
        'final_delivery': 'direct W1 evidence; native core order followed by complementary tail',
        'evidence_char_budget': None, 'embedding_concurrency': 1, 'embedding_max_attempts': 1,
        'http_admission_limit': 128, 'same_input_single_flight': True,
        'credentials_in_manifest': False,
        'source_files_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(source.glob('*.py'))},
    }
    app.state.memory_service = memory
    app.state.version_manifest = manifest
    (run / 'version-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', required=True)
    parser.add_argument('--api-key-file', required=True)
    parser.add_argument('--embedding-profile', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=18126)
    args = parser.parse_args()
    profile = load_profile(args.embedding_profile)
    if profile['dimensions'] != 1024 or profile.get('client_max_attempts') != 1:
        parser.error('Require Bailian text-embedding-v4 /1024 and client_max_attempts=1')
    app = build_app(db_path=args.db,
        api_key=Path(args.api_key_file).expanduser().read_text(encoding='utf-8').strip(),
        embedder=BailianEmbeddings(profile, user_confirmed=True, max_attempts=1), run_dir=args.run_dir)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, workers=1, limit_concurrency=128,
                timeout_keep_alive=5, log_level='info')


if __name__ == '__main__':
    main()
