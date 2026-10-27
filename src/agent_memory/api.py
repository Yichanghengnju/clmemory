"""Official Textual HTTP adapter for the shared memory service."""

from contextlib import asynccontextmanager
from pathlib import Path
from secrets import compare_digest
import hashlib
import json
import logging
import traceback
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import Field
from .evidence_tracing import SearchTraceMiddleware, current_trace, finish_http_error

from .domain import (
    AddReceipt,
    AddRequest,
    ContractModel,
    Memory,
    RequestConflict,
    SearchRequest,
    Text,
)

logger=logging.getLogger('clmemory.service')


async def record_failure(request, exc, status):
    """Log safe error provenance without headers, source text or exception arguments."""
    record={'event':'memory_request_failed','http_status':status,'path':request.url.path,
        'error_type':type(exc).__name__, 'frames':[
            {'file':Path(f.filename).name,'function':f.name,'line':f.lineno}
            for f in traceback.extract_tb(exc.__traceback__)[-6:]]}
    diagnostic=getattr(exc,'diagnostic',None)
    if isinstance(diagnostic,dict):
        record['diagnostic']={k:v for k,v in diagnostic.items()
            if k in {'category','payload_bytes','input_tokens','candidates','limit'}
            and (v is None or type(v) in {str,int})}
    try:
        payload=await request.json()
        for key in ['user_id','query','request_id']:
            value=payload.get(key)
            if isinstance(value,str):record[key+'_sha256']=hashlib.sha256(value.encode()).hexdigest()
    except Exception:
        pass
    logger.error(json.dumps(record,sort_keys=True))


class _SearchHit(ContractModel):
    id: Text
    content: Text
    score: Annotated[float, Field(allow_inf_nan=False)]


class _SearchResponse(ContractModel):
    data: list[_SearchHit]


def create_app(
    db_path: str | Path, *, service: Memory | None = None, api_key: str | None = None,
    enforce_search_limit: bool = True,
    trace_recorder=None,
) -> FastAPI:
    """Build one app; injected services remain owned by their caller.

    Reject oversized Search results by default. Disabling the count check is
    for replaying the legacy adapter; it does not make excess results valid.
    """
    if api_key == "":
        raise ValueError("api_key must be nonempty when configured")
    if type(enforce_search_limit) is not bool:
        raise TypeError("enforce_search_limit must be a bool")
    if trace_recorder is not None and trace_recorder.enabled and service is None:
        raise ValueError("HTTP evidence tracing requires an injected AMI service")

    owned_service = service is None
    if service is None:
        from .memory import MemoryService

        service = MemoryService(db_path)
    memory = service
    if trace_recorder is not None and trace_recorder.enabled:
        from .method_active_memoryindex import ActiveMemoryIndexMemory
        if not isinstance(memory, ActiveMemoryIndexMemory):
            raise ValueError("Evidence tracing pilot supports AMI only")
        if trace_recorder.writer is None:
            raise ValueError("HTTP tracing requires an explicit existing-result writer")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            yield
        finally:
            if owned_service:
                memory.close()

    app = FastAPI(title="Agent Memory Textual API", lifespan=lifespan)
    if trace_recorder is not None and trace_recorder.enabled:
        app.add_middleware(SearchTraceMiddleware, recorder=trace_recorder)
        app.state.evidence_trace_recorder = trace_recorder
    bearer = HTTPBearer(auto_error=False)

    def authorize(
        request: Request,
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    ) -> None:
        if api_key is None:
            return
        presented = None
        if credentials is not None:
            presented = credentials.credentials
        else:
            auth_header = request.headers.get("authorization", "")
            for scheme in ("Token", "token"):
                if auth_header.startswith(scheme + " "):
                    presented = auth_header[len(scheme) + 1:]
                    break
            if presented is None:
                presented = request.headers.get("x-api-key")
        if presented is None or not compare_digest(
            presented.encode("utf-8"), api_key.encode("utf-8")
        ):
            from fastapi import HTTPException

            raise HTTPException(
                status_code=401,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @app.exception_handler(RequestConflict)
    async def conflict_handler(request: Request, exc: RequestConflict) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": "Request conflict"})

    @app.exception_handler(TimeoutError)
    async def timeout_handler(request: Request, exc: TimeoutError) -> JSONResponse:
        await record_failure(request,exc,504)
        return JSONResponse(status_code=504, content={"detail": "Memory service timed out"})

    @app.exception_handler(ConnectionError)
    async def unavailable_handler(request: Request, exc: ConnectionError) -> JSONResponse:
        await record_failure(request,exc,503)
        return JSONResponse(status_code=503, content={"detail": "Memory service unavailable"})

    @app.exception_handler(Exception)
    async def error_handler(request: Request, exc: Exception) -> JSONResponse:
        await record_failure(request,exc,500)
        response = JSONResponse(status_code=500, content={"detail": "Memory service failed"})
        if trace_recorder is not None and trace_recorder.enabled:
            finish_http_error(request, 500)
        return response

    @app.post("/add", response_model=AddReceipt, dependencies=[Depends(authorize)])
    def add(request: AddRequest) -> AddReceipt:
        return memory.add(request)

    @app.post("/search", response_model=_SearchResponse, dependencies=[Depends(authorize)])
    def search(request: SearchRequest) -> _SearchResponse:
        # Preserve the caller's bound even if an injected method mutates its
        # request. Invalid output is a service failure, never an empty result.
        top_k = request.top_k
        hits = memory.search(request)
        trace = current_trace()
        if trace is not None:
            trace.returned(hits, stage="http_method_result")
        if enforce_search_limit and len(hits) > top_k:
            if trace is not None:
                trace.event("protocol_rejected", category="result_count_exceeds_top_k",
                            requested_top_k=top_k, returned_count=len(hits))
            raise ValueError("Search result count exceeds top_k")
        return _SearchResponse(
            data=[
                _SearchHit(id=hit.id, content=hit.content, score=hit.score)
                for hit in hits
            ]
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app
