"""Explicit v1.0.5 ablation: original delivery, no final relevance/safety filter."""
import argparse
import json
from pathlib import Path

from .bailian_embeddings import BailianEmbeddings, load_profile
from .submission import build_app as build_v105_app

VERSION='v1.0.5-nofilter-textual'


def build_app(*,db_path,api_key,embedder,manifest_dir):
    app=build_v105_app(db_path=db_path,api_key=api_key,embedder=embedder,
        track='textual',final_filter=None)
    manifest={**app.state.version_manifest,'version':VERSION,'base_recipe':'clmemory v1.0.5',
        'ablation':'entire final filter removed','final_filter_enabled':False,
        'credential_filter_stage_enabled':False,'llm_calls_in_add':0,'llm_calls_in_search':0,
        'http_admission_limit':128,'extra_raw_search_audit':False}
    root=Path(manifest_dir);root.mkdir(parents=True,exist_ok=True)
    (root/'version-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    app.state.version_manifest=manifest
    return app


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--db',required=True);p.add_argument('--api-key-file',required=True)
    p.add_argument('--embedding-profile',required=True);p.add_argument('--manifest-dir',required=True)
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=18107)
    args=p.parse_args();profile=load_profile(args.embedding_profile)
    if profile['dimensions']!=1024:p.error('This v1.0.5 ablation requires text-embedding-v4 / 1024 dimensions')
    app=build_app(db_path=args.db,api_key=Path(args.api_key_file).read_text().strip(),
        embedder=BailianEmbeddings(profile,user_confirmed=True),manifest_dir=args.manifest_dir)
    import uvicorn
    uvicorn.run(app,host=args.host,port=args.port,workers=1,limit_concurrency=128,timeout_keep_alive=5,log_level='info')


if __name__=='__main__':main()
