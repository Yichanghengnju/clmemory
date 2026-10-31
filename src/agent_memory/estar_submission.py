"""Textual f04@100 retrieval + native mini harness + source fusion + frozen S14."""
import argparse
from contextlib import asynccontextmanager
from pathlib import Path
import hashlib
import json

from .api import create_app
from .bailian_embeddings import BailianEmbeddings, load_profile
from .deepseek_memory import EnsembleMemory
from .memmachine_submission import SerializedEmbeddings
from .method_memmachine import _index_lock
from .method_memmachine_w1 import MemMachineW1Memory
from .native_chat import NativeChatProvider
from .persistent_native_harness import NativeHarnessPool, PersistentNativeHarness

VERSION = 'textual-f04-estar-s14-v1.0'
SKILL_SHA256 = 'cb745515471e0a5828e99c390ffb3c9fb5f22b66809a0f67044b1e45af764f10'
CONFIG = {'embedding_dimensions':1024, 'deriver':'whole_text', 'segmenter':'passthrough',
          'rrf_k':60, 'bm25_tokenizer':'simple', 'dedup_overfetch':4,
          'expand_context':0, 'max_results':16}
NATIVE_CALL_CEILING = 384  # Finite operational bound for this one Smoke, including preflight.


class ExactQuerySingleFlight:
    """Wait for an identical request instead of colliding with its live task owner."""
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
        with _index_lock(self.directory/(key+'.lock')):
            return self.memory.search(request)
    def close(self):
        self.memory.close()


def build_app(*, db_path, api_key, embedder, run_dir, runtime_dir=None, mini_key_file=None,
              harness=None, native_call_ceiling=NATIVE_CALL_CEILING):
    if not api_key:
        raise ValueError('A nonempty private service token is required')
    if type(native_call_ceiling) is not int or native_call_ceiling < 1:
        raise ValueError('A finite positive native call ceiling is required')
    run = Path(run_dir)
    run.mkdir(parents=True, exist_ok=True)
    skill_file = Path(__file__).resolve().parents[2] / 'skills/S14.txt'
    skill = skill_file.read_text(encoding='utf-8')
    if hashlib.sha256(skill.encode()).hexdigest() != SKILL_SHA256:
        raise ValueError('Frozen S14 text changed')
    embeddings = SerializedEmbeddings(embedder)
    provider = pool = None
    if harness is None:
        provider = NativeChatProvider(run/'native-calls.sqlite3', key_file=mini_key_file,
            max_calls=native_call_ceiling, max_output_tokens=native_call_ceiling*1536,
            max_input_estimate=native_call_ceiling*180000, allow_dated_snapshot=True)
        pool = NativeHarnessPool(max_workers=6, max_sessions=16)
        harness = PersistentNativeHarness(provider=provider, pool=pool, runtime_dir=runtime_dir,
            artifact_dir=run/'native', skill=skill, max_steps=6, timeout_s=150,
            max_tokens=1536, enable_evidence_chain=False, model='gpt-4o-mini')
    if harness.identity['skill_sha256'] != SKILL_SHA256 or harness.identity['model'] != 'gpt-4o-mini':
        raise ValueError('Harness identity differs from frozen E*')
    baseline = MemMachineW1Memory(
        db_path, embedder=embeddings, config=CONFIG, complementary_tail=True,
        coverage_rescue=False, combined_tail=False, rule_quota=False,
        numeric_anchor_boost=False, date_annotations=False,
        session_rrf=False, temporal_soft_boost=False)
    memory = ExactQuerySingleFlight(EnsembleMemory(baseline, harness, ledger_path=run/'search.sqlite3'),
                                   run/'http-singleflight')
    app = create_app(db_path, service=memory, api_key=api_key, enforce_search_limit=True)
    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            if pool:
                pool.close()
            memory.close()
            if provider:
                provider.close()
            embeddings.close()
    app.router.lifespan_context = lifespan
    source = Path(__file__).parent
    manifest = {'system':'clmemory', 'version':VERSION, 'track':'textual', 'method':memory.identity,
        'embedding':embeddings.specification, 'S14_sha256':SKILL_SHA256,
        'H':True, 'F':True, 'S14':True, 'KD':False, 'AgentDistill':False, 'deep_tail':False,
        'native_transport':'official-sdk-persistent-fresh-session-v1', 'max_steps':6,
        'max_tool_calls':12, 'timeout_s':150, 'output_tokens_per_request':1536,
        'new_native_call_ceiling':native_call_ceiling,
        'f04':True, 'native_core_cap':16, 'requested_top_k_max':100,
        'f04_scoring':'static coverage and redundancy against initial R context',
        'f04_compatibility':'tested W1 implementation; no neighbor-ID fix or dynamic MMR',
        'f04_known_limit':'configured neighbor8 may not fire because core Evidence.id differs from source_id',
        'f04_disabled_tuning':['rule_quota','numeric_anchor_boost','coverage_rescue','date_annotations'],
        'final_delivery':'unchanged source RRF60, visibility rehydration, 60000-char whole-source budget',
        'final_core_protection':False,
        'final_100_item_guarantee':False,
        'call_ceiling_authority':'assistant operational guard for one user-authorized Smoke, not a user quoted number',
        'llm_calls_in_add':0, 'llm_calls_in_search':'0..6 native model requests; compatible successful requests cached',
        'H_failure_behavior':'preserve original H failure and use original R/F fallback when available',
        'http_admission_limit':128, 'native_workers':6, 'embedding_concurrency':1,
        'same_input_single_flight':True,
        'extra_raw_search_audit':False, 'private_receipts':'evaluation-only; retain at most30 days',
        'credentials_in_manifest':False,
        'source_files_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.glob('*.py'))}}
    app.state.memory_service = memory
    app.state.version_manifest = manifest
    (run/'version-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    return app


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--db', required=True)
    p.add_argument('--api-key-file', required=True)
    p.add_argument('--embedding-profile', required=True)
    p.add_argument('--mini-key-file', required=True)
    p.add_argument('--runtime-dir', required=True)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=18125)
    p.add_argument('--native-call-ceiling', type=int, default=NATIVE_CALL_CEILING)
    a = p.parse_args()
    profile = load_profile(a.embedding_profile)
    if profile['model'] != 'text-embedding-v4' or profile['dimensions'] != 1024:
        p.error('Formal embedding must be text-embedding-v4 /1024')
    app = build_app(db_path=a.db, api_key=Path(a.api_key_file).read_text().strip(),
        embedder=BailianEmbeddings(profile,user_confirmed=True,max_attempts=1),run_dir=a.run_dir,
        runtime_dir=a.runtime_dir,mini_key_file=a.mini_key_file,
        native_call_ceiling=a.native_call_ceiling)
    import uvicorn
    uvicorn.run(app,host=a.host,port=a.port,workers=1,limit_concurrency=128,
                timeout_keep_alive=5,log_level='info')


if __name__ == '__main__':
    main()
