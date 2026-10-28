"""Attributed native MemMachine EventMemory baseline, not the unknown AML scored deployment."""
import argparse
from contextlib import asynccontextmanager
import hashlib
import json
from pathlib import Path
from threading import RLock

from .api import create_app
from .bailian_embeddings import BailianEmbeddings, load_profile
from .method_memmachine import MemMachineMemory

VERSION='memmachine-event-v4-top100-v1.0'
CONFIG={'embedding_dimensions':1024,'deriver':'whole_text','segmenter':'passthrough',
        'rrf_k':60,'bm25_tokenizer':'simple','dedup_overfetch':4,'expand_context':0,'max_results':100}


class SerializedEmbeddings:
    """Share identical in-flight cache entries without concurrent provider duplication."""
    def __init__(self,embedder):self.embedder=embedder;self.lock=RLock()
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


def build_app(*,db_path,api_key,embedder,manifest_dir):
    if not api_key:raise ValueError('A nonempty private service token is required')
    embeddings=SerializedEmbeddings(embedder)
    memory=MemMachineMemory(db_path,embedder=embeddings,config=CONFIG)
    app=create_app(db_path,service=memory,api_key=api_key)
    @asynccontextmanager
    async def lifespan(app):
        try:yield
        finally:memory.close();embeddings.close()
    app.router.lifespan_context=lifespan
    source=Path(__file__).parent
    manifest={'system':'clmemory','version':VERSION,'track':'textual','method':memory.identity,
        'embedding':embeddings.specification,'llm_calls_in_add':0,'llm_calls_in_search':0,
        'final_filter_enabled':False,'clmemory_neighbors':False,'clmemory_anchors':False,
        'extra_raw_search_audit':False,'embedding_concurrency':1,'http_admission_limit':128,
        'source_files_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.glob('*.py'))},
        'credentials_in_manifest':False}
    root=Path(manifest_dir);root.mkdir(parents=True,exist_ok=True)
    (root/'version-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    app.state.memory_service=memory;app.state.version_manifest=manifest
    return app


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--db',required=True);p.add_argument('--api-key-file',required=True)
    p.add_argument('--embedding-profile',required=True);p.add_argument('--manifest-dir',required=True)
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=18108)
    a=p.parse_args();profile=load_profile(a.embedding_profile)
    if profile['dimensions']!=1024:p.error('This frozen baseline requires text-embedding-v4 / 1024 dimensions')
    app=build_app(db_path=a.db,api_key=Path(a.api_key_file).read_text().strip(),
        embedder=BailianEmbeddings(profile,user_confirmed=True),manifest_dir=a.manifest_dir)
    import uvicorn
    uvicorn.run(app,host=a.host,port=a.port,workers=1,limit_concurrency=128,timeout_keep_alive=5,log_level='info')


if __name__=='__main__':main()
