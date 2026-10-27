"""Independent InvMem baseline: original top8 recipe, explicit text-embedding-v4, no LLM."""
import argparse
from contextlib import asynccontextmanager
import hashlib
import json
import logging
from pathlib import Path
import sqlite3
from threading import RLock
import time

from .api import create_app
from .bailian_embeddings import BailianEmbeddings, load_profile
from .method_invmem import InvMemMemory

VERSION='invmem-v4-baseline8-v1.0'
CONFIG={'chunk_size':320,'chunk_overlap':40,'rrf_k':60,'lexical_weight':0.5,
        'window':1,'seed_count':20,'max_results':8,'options_weight':0.0,'reranker':False}


class SerializedEmbeddings:
    """Bound provider concurrency and wait for identical in-flight cache inputs."""
    def __init__(self, embedder):
        self.embedder=embedder
        self.lock=RLock()
    @property
    def identity(self):return self.embedder.identity
    @property
    def dimensions(self):return self.embedder.dimensions
    @property
    def specification(self):return getattr(self.embedder,'specification',self.identity)
    def embed_documents(self,texts):
        with self.lock:return self.embedder.embed_documents(texts)
    def embed_query(self,text):
        with self.lock:return self.embedder.embed_query(text)
    def close(self):
        if hasattr(self.embedder,'store'):self.embedder.store.close()


class SearchAudit:
    """Private observation only. The callback never changes retrieval decisions."""
    def __init__(self,path):
        self.lock=RLock();self.failures=0
        self.db=sqlite3.connect(path,check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS search_events (id INTEGER PRIMARY KEY,at REAL,payload TEXT NOT NULL)')
        self.db.commit()
    def __call__(self,payload):
        try:
            text=json.dumps(payload,ensure_ascii=False,sort_keys=True,allow_nan=False)
            with self.lock:
                self.db.execute('INSERT INTO search_events(at,payload) VALUES (?,?)',(time.time(),text))
                self.db.commit()
        except Exception as error:
            self.failures+=1
            logging.getLogger('clmemory.service').error(json.dumps({
                'event':'invmem_trace_write_failed','error_type':type(error).__name__}))
    def close(self):self.db.close()


def build_app(*,db_path,api_key,embedder,manifest_dir):
    if not api_key:raise ValueError('A nonempty private service token is required')
    root=Path(manifest_dir);root.mkdir(parents=True,exist_ok=True)
    audit=SearchAudit(root/'search-audit.sqlite3')
    embeddings=SerializedEmbeddings(embedder)
    memory=InvMemMemory(db_path,embedder=embeddings,config=CONFIG,trace_hook=audit)
    app=create_app(db_path,service=memory,api_key=api_key)
    @asynccontextmanager
    async def lifespan(app):
        try:yield
        finally:
            memory.close();embeddings.close();audit.close()
    app.router.lifespan_context=lifespan
    source=Path(__file__).parent
    manifest={'system':'clmemory','version':VERSION,'track':'textual','method':memory.identity,
        'embedding':embeddings.specification,'llm_calls_in_add':0,'llm_calls_in_search':0,
        'final_filter_enabled':False,'embedding_concurrency':1,'http_admission_limit':128,
        'source_files_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.glob('*.py'))},
        'trace':{'enabled':True,'kind':'observed candidates/ranks/final IDs','private_inputs_recorded':True},
        'credentials_in_manifest':False}
    (root/'version-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    app.state.memory_service=memory;app.state.version_manifest=manifest
    return app


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--db',required=True);p.add_argument('--api-key-file',required=True)
    p.add_argument('--embedding-profile',required=True);p.add_argument('--manifest-dir',required=True)
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=18106)
    args=p.parse_args();profile=load_profile(args.embedding_profile)
    if profile['dimensions']!=1024:p.error('This baseline requires text-embedding-v4 / 1024 dimensions')
    app=build_app(db_path=args.db,api_key=Path(args.api_key_file).read_text().strip(),
        embedder=BailianEmbeddings(profile,user_confirmed=True),manifest_dir=args.manifest_dir)
    import uvicorn
    uvicorn.run(app,host=args.host,port=args.port,workers=1,limit_concurrency=128,log_level='info')


if __name__=='__main__':main()
