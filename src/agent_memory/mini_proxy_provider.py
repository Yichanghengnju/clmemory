"""Isolated, single-send GPT-4o-mini transport with a durable TaskStore ledger.

No retry or recovery is automatic. A dedicated ledger bounds the whole run,
including failed/unknown sends; successful exact requests are reused. Input
budget is a conservative UTF-8-byte estimate, not a tokenizer measurement.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import http.client
import json
import math
from pathlib import Path
import re
import sqlite3
import time
import threading

from .domain import Completion, ModelConfig
from .providers import ProviderError, _redact, _redact_json
from .tasks import TaskStore, coordinator_lock

BASE_URL = "https://api.openai-proxy.org/v1"
STAGE = "mini_proxy_request_v1"


class MiniProxyError(ProviderError):
    pass


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _send(payload: bytes, key: str, timeout: float):
    # HTTPSConnection never follows redirects and does not consult proxy env.
    connection = http.client.HTTPSConnection("api.openai-proxy.org", timeout=timeout)
    try:
        connection.request("POST", "/v1/chat/completions", body=payload,
                           headers={"Authorization": "Bearer " + key,
                                    "Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


class MiniProxyProvider:
    def __init__(self, ledger_path: str | Path, *, key_file: str | Path,
                 max_calls: int, max_output_tokens: int,
                 max_input_estimate: int | None = None,
                 allow_dated_snapshot: bool = False, transport=None):
        for name, value in (("max_calls", max_calls), ("max_output_tokens", max_output_tokens)):
            if type(value) is not int or value < 0:
                raise ValueError(name + " must be a nonnegative integer")
        if max_input_estimate is not None and (type(max_input_estimate) is not int or max_input_estimate < 0):
            raise ValueError("max_input_estimate must be a nonnegative integer")
        if str(ledger_path) == ":memory:":
            raise ValueError("a persistent dedicated ledger is required")
        self.path = Path(ledger_path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.store = TaskStore(self.path)
        self.key_file = Path(key_file).expanduser().resolve()
        repo = Path(__file__).resolve().parents[2]
        if self.key_file.is_relative_to(repo):
            raise ValueError("key_file must be outside the repository")
        self.max_calls = max_calls
        self.max_output_tokens = max_output_tokens
        self.max_input_estimate = max_input_estimate
        self.allow_dated_snapshot = allow_dated_snapshot
        self.transport = transport or _send
        self._admission_lock = threading.Lock()

    def complete(self, *, config: ModelConfig, prompt: str, system: str = "") -> Completion:
        if config.model != "gpt-4o-mini" or config.effort is not None:
            raise MiniProxyError("only gpt-4o-mini without effort is supported", reason="configuration")
        if type(config.retries) is not int or config.retries != 0:
            raise MiniProxyError("retries must be explicitly zero for single-send requests", reason="configuration")
        if type(config.max_tokens) is not int or config.max_tokens < 1:
            raise MiniProxyError("max_tokens must be positive", reason="configuration")
        if not math.isfinite(config.timeout_s) or config.timeout_s <= 0:
            raise MiniProxyError("timeout must be finite and positive", reason="configuration")
        if not isinstance(prompt, str) or not prompt.strip() or not isinstance(system, str):
            raise MiniProxyError("prompt/system must be text", reason="configuration")
        key = self.key_file.read_text(encoding="utf-8").strip()
        if not key or any(c.isspace() for c in key):
            raise MiniProxyError("invalid credential file", reason="configuration")
        messages = ([{"role": "system", "content": system}] if system else [])
        messages.append({"role": "user", "content": prompt})
        payload = {"model": config.model, "messages": messages, "max_tokens": config.max_tokens}
        if config.temperature is not None:
            if not math.isfinite(config.temperature) or not 0 <= config.temperature <= 2:
                raise MiniProxyError("invalid temperature", reason="configuration")
            payload["temperature"] = config.temperature
        encoded = _json(payload).encode("utf-8")
        identity = {"endpoint": BASE_URL + "/chat/completions", "request": payload,
                    "allow_dated_snapshot": self.allow_dated_snapshot}
        digest = hashlib.sha256(_json(identity).encode("utf-8")).hexdigest()
        task_key = STAGE + ":" + digest
        safe = json.loads(_redact_json(_json(identity), key))
        safe["input_estimate"] = len(encoded)
        safe["output_reservation"] = config.max_tokens
        # Only the local reservation is locked; network never holds a DB write.
        with self._admission_lock, coordinator_lock(str(self.path) + ".admission.lock"):
            cached = self.store.get_result(task_key)
            if cached is not None:
                return Completion(**cached["result"])
            existing = [t for t in self.store.list_tasks(stage=STAGE) if t["key"] == task_key]
            if existing and self.store.list_attempts(existing[0]["id"]):
                raise MiniProxyError("request is held; inspect its existing attempt, do not resend", reason="held")
            reserved = [t for t in self.store.list_tasks(stage=STAGE)
                        if self.store.list_attempts(t["id"])]
            if (len(reserved) >= self.max_calls or
                    sum(t["payload"]["output_reservation"] for t in reserved) + config.max_tokens > self.max_output_tokens or
                    (self.max_input_estimate is not None and
                     sum(t["payload"]["input_estimate"] for t in reserved) + len(encoded) > self.max_input_estimate)):
                raise MiniProxyError("run request/token reservation budget exhausted", reason="budget")
            task = self.store.enqueue(task_key, STAGE, digest, safe)
            claimed = self.store.claim(task_id=task["id"])
            if claimed is None:
                raise MiniProxyError("request is held", reason="held")
        owner, attempt_id = claimed["owner"], claimed["attempt_id"]
        started = time.monotonic()
        try:
            status, raw = self.transport(encoded, key, config.timeout_s)
        except Exception as exc:
            response = {"transport_error": type(exc).__name__, "outcome": "unknown",
                        "latency_s": time.monotonic() - started}
            self.store.record_attempt(task["id"], owner, response=response,
                                      validation={"valid": False, "outcome": "unknown"})
            self.store.fail(task["id"], owner, {"kind": "unknown", "held": True})
            # TaskStore has no public single-attempt uncertain setter; keep its
            # existing uncertainty column consistent without recovering tasks.
            with self.store._transaction(write=True) as db:
                db.execute("UPDATE attempts SET uncertain=1 WHERE id=?", (attempt_id,))
            raise MiniProxyError("transport outcome unknown; request held", attempt_id=attempt_id, reason="unknown") from None
        raw_text = _redact_json(raw.decode("utf-8", errors="replace"), key)
        response = {"http_status": status, "raw_response": raw_text,
                    "latency_s": time.monotonic() - started}
        self.store.record_attempt(task["id"], owner, response=response)
        reason = "invalid_response"
        try:
            if status != 200:
                reason = "http_status"
                raise ValueError()
            body = json.loads(raw_text, parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
            model = body.get("model", "")
            if not (model == config.model or (self.allow_dated_snapshot and
                    isinstance(model, str) and re.fullmatch(r"gpt-4o-mini-\d{4}-\d{2}-\d{2}", model))):
                reason = "model_identity"
                raise ValueError()
            choices = body["choices"]
            if not isinstance(choices, list) or len(choices) != 1:
                raise ValueError()
            choice = choices[0]
            message = choice["message"]
            if choice.get("finish_reason") != "stop" or message.get("refusal") or message.get("tool_calls"):
                reason = "incomplete_or_refusal"
                raise ValueError()
            text = message.get("content")
            if not isinstance(text, str) or not text.strip():
                reason = "empty_text"
                raise ValueError()
            usage = body.get("usage")
            if not isinstance(usage, dict):
                reason = "missing_usage"
                raise ValueError()
            if any(type(usage.get(field)) is not int or usage[field] < 0
                   for field in ("prompt_tokens", "completion_tokens", "total_tokens")):
                reason = "invalid_usage"
                raise ValueError()
            # Repository Provider convention differs from OpenAI finish_reason;
            # the original 'stop' remains in the immutable raw response.
            completion = Completion(text=_redact(text, key), stop_reason="end_turn", usage=usage,
                                    model=model, attempt_id=attempt_id)
        except (ValueError, TypeError, KeyError, AttributeError):
            self.store.record_attempt(task["id"], owner, response=response,
                                      validation={"valid": False, "reason": reason})
            self.store.fail(task["id"], owner, {"kind": reason, "held": True})
            raise MiniProxyError("response rejected; request held: " + reason,
                                 attempt_id=attempt_id, reason=reason) from None
        self.store.record_attempt(task["id"], owner, response=response, usage=usage,
                                  validation={"valid": True, "response_model": model})
        self.store.succeed(task["id"], owner, asdict(completion))
        return completion

    def close(self):
        self.store.close()


class PriorMiniProvider:
    """Explicitly frozen, read-only reuse of successful raw mini completions.

    Parser success is deliberately not required: a new parser may reinterpret
    the preserved completion without paying for the identical request again.
    Any prior attempted non-success is held, including interrupted requests.
    """
    def __init__(self, identity, fallback, *, on_reuse=None):
        self.identity=identity;self.fallback=fallback;self.on_reuse=on_reuse
        for suffix,field in [('', 'sha256'),('-wal','wal_sha256')]:
            p=Path(str(identity['database'])+suffix)
            if field in identity and hashlib.sha256(p.read_bytes()).hexdigest()!=identity[field]:
                raise MiniProxyError('prior mini ledger identity changed',reason='held')

    def complete(self, *, config, prompt, system=''):
        if config.model!='gpt-4o-mini' or config.effort is not None or type(config.retries) is not int or config.retries!=0:
            raise MiniProxyError('unsupported prior mini model/config',reason='configuration')
        if type(config.max_tokens) is not int or config.max_tokens<1 or not isinstance(prompt,str) or not prompt.strip() or not isinstance(system,str):
            raise MiniProxyError('invalid prior mini input',reason='configuration')
        if not math.isfinite(config.timeout_s) or config.timeout_s<=0:
            raise MiniProxyError('invalid timeout',reason='configuration')
        payload={'model':config.model,'messages':([{'role':'system','content':system}] if system else [])+
                 [{'role':'user','content':prompt}],'max_tokens':config.max_tokens}
        if config.temperature is not None:
            if not math.isfinite(config.temperature) or not 0<=config.temperature<=2:
                raise MiniProxyError('invalid temperature',reason='configuration')
            payload['temperature']=config.temperature
        identity={'endpoint':BASE_URL+'/chat/completions','request':payload,
                  'allow_dated_snapshot':self.identity['allow_dated_snapshot']}
        digest=hashlib.sha256(_json(identity).encode('utf-8')).hexdigest()
        expected={**identity,'input_estimate':len(_json(payload).encode('utf-8')),'output_reservation':config.max_tokens}
        path=Path(self.identity['database'])
        with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as db:
            db.row_factory=sqlite3.Row;db.execute('pragma query_only=ON')
            task=db.execute('select * from tasks where key=?',(STAGE+':'+digest,)).fetchone()
            if task is not None:
                if task['stage']!=STAGE or task['input_hash']!=digest or json.loads(task['payload'])!=expected:
                    raise MiniProxyError('prior mini payload mismatch',reason='held')
                result=db.execute('select * from results where task_id=?',(task['id'],)).fetchone()
                attempted=db.execute('select 1 from attempts where task_id=?',(task['id'],)).fetchone()
                if task['status']!='succeeded' or result is None:
                    if attempted or task['status']!='pending':raise MiniProxyError('exact prior mini outcome held; no resend',reason='held')
                else:
                    attempt=db.execute('select * from attempts where id=?',(result['attempt_id'],)).fetchone()
                    try:
                        c=Completion(**json.loads(result['result']))
                        valid_model=c.model==config.model or (identity['allow_dated_snapshot'] and isinstance(c.model,str) and re.fullmatch(r'gpt-4o-mini-\d{4}-\d{2}-\d{2}',c.model))
                        if not attempt or attempt['status']!='succeeded' or attempt['task_id']!=task['id'] or c.attempt_id!=attempt['id']:
                            raise ValueError()
                        raw=json.loads(json.loads(attempt['response'])['raw_response'])
                        if (c.stop_reason!='end_turn' or not isinstance(c.text,str) or not c.text.strip() or not valid_model
                                or raw.get('model')!=c.model or raw['choices'][0].get('finish_reason')!='stop'
                                or raw['choices'][0]['message'].get('content')!=c.text
                                or not isinstance(c.usage,dict) or any(type(c.usage.get(k)) is not int or c.usage[k]<0 for k in ['prompt_tokens','completion_tokens','total_tokens'])):
                            raise ValueError()
                    except (ValueError,TypeError,KeyError,IndexError,AttributeError):
                        raise MiniProxyError('invalid prior mini success; no resend',reason='held') from None
                    ref={'task_id':task['id'],'result_id':result['id'],'attempt_id':c.attempt_id,'reused':True,
                         'database':str(path),'database_sha256':self.identity['sha256'],
                         'database_wal_sha256':self.identity.get('wal_sha256'),'input_sha256':digest,
                         'reuse_scope':'prior_provider_success_revalidated_by_current_parser'}
                    if self.on_reuse:self.on_reuse(ref)
                    return c
        return self.fallback.complete(config=config,prompt=prompt,system=system)
