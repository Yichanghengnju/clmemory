"""Single-send, source-scoped OpenAI tool protocol on the existing TaskStore.

This is a transport, not an agent loop. DeepSeek owns all model/tool turns.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from contextlib import contextmanager
from contextvars import ContextVar

from .mini_proxy_provider import MiniProxyProvider, MiniProxyError, BASE_URL, _json
from .providers import _redact_json
from .tasks import coordinator_lock

STAGE = "deepseek_native_chat_v1"
_request_window = ContextVar('native_request_window', default=None)


@contextmanager
def native_request_window(deadline, timing):
    """Out-of-band per-query deadline; never changes request/cache identity.

    Context stays on the gateway handler thread through control/reuse wrappers.
    It cannot leak across concurrent queries or be included in a model prompt.
    """
    started = time.monotonic()
    token = _request_window.set((deadline, timing))
    try:
        yield
    finally:
        timing['gateway_elapsed_s'] = time.monotonic() - started
        _request_window.reset(token)


class NativeChatProvider(MiniProxyProvider):
    def admission_totals(self):
        """Attempted task reservations, including failures/unknown, in one read.

        Preserve the cumulative budget semantics without loading every prompt
        or repeatedly scanning the full attempt history during each admission.
        The caller retains the existing process and cross-process admission locks.
        """
        with self.store._transaction() as db:
            row = db.execute('''SELECT COUNT(*),
                COALESCE(SUM(json_extract(t.payload,'$.output_reservation')),0),
                COALESCE(SUM(json_extract(t.payload,'$.input_estimate')),0)
                FROM tasks t WHERE t.stage=?
                AND EXISTS(SELECT 1 FROM attempts a WHERE a.task_id=t.id)''', (STAGE,)).fetchone()
        return {'calls': row[0], 'output_reservation': row[1], 'input_estimate': row[2]}

    def _finished_empty(self, body, finalized):
        try:
            choice = body["choices"][0]
            return (finalized and len(body["choices"]) == 1 and choice["finish_reason"] == "stop"
                and choice["message"].get("content") in (None, "")
                and not choice["message"].get("refusal") and not choice["message"].get("tool_calls")
                and (body["model"] == "gpt-4o-mini" or self.allow_dated_snapshot and
                     re.fullmatch(r"gpt-4o-mini-\d{4}-\d{2}-\d{2}", body["model"]))
                and all(type(body["usage"].get(k)) is int and body["usage"][k] >= 0
                        for k in ("prompt_tokens", "completion_tokens", "total_tokens")))
        except (KeyError, IndexError, TypeError):
            return False

    def chat(self, payload, *, scope, finalized=False):
        window = _request_window.get()
        deadline, timing = window if window is not None else (None, {})
        def check_deadline():
            if deadline is not None and time.monotonic() >= deadline:
                timing['physical_send'] = False
                timing['outcome'] = 'deadline_before_send'
                raise MiniProxyError('native deadline expired before send', reason='deadline_before_send')
        check_deadline()
        if any((self.path.parent / name).exists() for name in ("STOP", "PAUSE")):
            raise MiniProxyError("native execution control is set", reason="paused")
        if payload.get("model") != "gpt-4o-mini":
            raise MiniProxyError("native retrieval model must be gpt-4o-mini", reason="configuration")
        payload = dict(payload)
        payload.pop("stream", None)
        payload.pop("stream_options", None)
        payload.pop("store", None)
        if "max_completion_tokens" in payload:
            payload["max_tokens"] = payload.pop("max_completion_tokens")
        cap = payload.get("max_tokens")
        if type(cap) is not int or cap < 1 or cap > 4096:
            raise MiniProxyError("invalid native output budget", reason="configuration")
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise MiniProxyError("missing native messages", reason="configuration")
        encoded = _json(payload).encode()
        identity = {"endpoint": BASE_URL + "/chat/completions", "request": payload,
                    "scope": scope, "allow_dated_snapshot": self.allow_dated_snapshot}
        digest = hashlib.sha256(_json(identity).encode()).hexdigest()
        task_key = STAGE + ":" + digest
        key = self.key_file.read_text(encoding="utf-8").strip()
        if not key or any(c.isspace() for c in key):
            raise MiniProxyError("invalid credential", reason="configuration")
        safe = json.loads(_redact_json(_json(identity), key))
        safe.update(input_estimate=len(encoded), output_reservation=cap)
        waiting = time.monotonic()
        with self._admission_lock, coordinator_lock(str(self.path) + ".admission.lock"):
            admitted = time.monotonic()
            timing['admission_wait_s'] = admitted - waiting
            check_deadline()
            cached = self.store.get_result(task_key)
            if cached is not None:
                return cached["result"]
            with self.store._transaction() as db:
                previous = [dict(row) for row in db.execute('SELECT id FROM tasks WHERE key=?', (task_key,))]
            if previous and self.store.list_attempts(previous[0]["id"]):
                # A completed finish tool legitimately needs no assistant prose.
                # Reinterpret its saved empty stop without altering the old
                # failure/attempt or sending again. All other failures stay held.
                attempts = self.store.list_attempts(previous[0]["id"])
                old = attempts[-1]
                response = old.get("response") or {}
                if response.get("http_status") == 200 and old.get("validation", {}).get("reason") == "invalid_response":
                    body = json.loads(response["raw_response"])
                    if self._finished_empty(body, finalized):
                        derived_key = "native_finished_stop_revalidation_v2:" + digest
                        if self.store.get_result(derived_key) is None:
                            derived = self.store.enqueue(derived_key, "native_raw_revalidation_v2", digest,
                                {"original_task": previous[0]["id"], "original_attempt": old["id"],
                                 "reason": "empty stop after server-validated finish_evidence", "new_sends": 0})
                            owner = self.store.claim(task_id=derived["id"])
                            self.store.succeed(derived["id"], owner["owner"], {"body": body, "original_attempt": old["id"]})
                        return body
                raise MiniProxyError("native request held; no automatic resend", reason="held")
            reserved = self.admission_totals()
            if (self.max_calls is not None and reserved['calls'] >= self.max_calls
                or self.max_output_tokens is not None and reserved['output_reservation'] + cap > self.max_output_tokens
                or self.max_input_estimate is not None and reserved['input_estimate'] + len(encoded) > self.max_input_estimate):
                raise MiniProxyError("native run budget exhausted", reason="budget")
            task = self.store.enqueue(task_key, STAGE, digest, safe)
            check_deadline()
            claim = self.store.claim(task_id=task["id"])
            if claim is None:
                raise MiniProxyError("native request already owned", reason="held")
            timing['admission_work_s'] = time.monotonic() - admitted
        owner, attempt = claim["owner"], claim["attempt_id"]
        try:
            check_deadline()
        except MiniProxyError:
            # A logical claim exists, but no transport was entered. Preserve
            # its reservation and proof; do not mislabel it transport unknown.
            self.store.record_attempt(task['id'], owner,
                response={'outcome': 'deadline_before_send', 'physical_send': False, 'timing': dict(timing)},
                validation={'valid': False, 'reason': 'deadline_before_send', 'physical_send': False})
            self.store.fail(task['id'], owner, {'kind': 'deadline_before_send', 'held': True}, retry=False)
            raise
        started = time.monotonic()
        remaining = 90 if deadline is None else min(90, deadline - started)
        if remaining <= 0:
            # Reuse the durable zero-send path even if the deadline ticks over
            # between the last check and timeout calculation.
            timing.update(physical_send=False, outcome='deadline_before_send')
            self.store.record_attempt(task['id'], owner,
                response={'outcome': 'deadline_before_send', 'physical_send': False, 'timing': dict(timing)},
                validation={'valid': False, 'reason': 'deadline_before_send', 'physical_send': False})
            self.store.fail(task['id'], owner, {'kind': 'deadline_before_send', 'held': True}, retry=False)
            raise MiniProxyError('native deadline expired before send', reason='deadline_before_send')
        timing['physical_send'] = True
        try:
            status, raw = self.transport(encoded, key, remaining)
        except Exception as exc:
            timing.update(transport_s=time.monotonic() - started, outcome='transport_unknown')
            self.store.record_attempt(task["id"], owner, response={"outcome": "unknown", "error_type": type(exc).__name__, "timing": dict(timing)},
                                      validation={"valid": False, "outcome": "unknown"})
            self.store.fail(task["id"], owner, {"kind": "unknown", "held": True})
            with self.store._transaction(write=True) as db:
                db.execute("UPDATE attempts SET uncertain=1 WHERE id=?", (attempt,))
            raise MiniProxyError("native transport unknown; held", reason="unknown") from None
        raw_text = _redact_json(raw.decode("utf-8", errors="replace"), key)
        timing.update(transport_s=time.monotonic() - started, outcome='response_received')
        response = {"http_status": status, "raw_response": raw_text, "latency_s": timing['transport_s'], 'timing': dict(timing)}
        self.store.record_attempt(task["id"], owner, response=response)
        reason = "invalid_response"
        try:
            if status != 200:
                reason = "http_status_" + str(status)
                raise ValueError()
            body = json.loads(raw_text)
            model = body.get("model", "")
            if not (model == "gpt-4o-mini" or self.allow_dated_snapshot and re.fullmatch(r"gpt-4o-mini-\d{4}-\d{2}-\d{2}", model)):
                reason = "model_identity"
                raise ValueError()
            choices = body["choices"]
            if len(choices) != 1:
                raise ValueError()
            choice, message = choices[0], choices[0]["message"]
            if message.get("refusal") or choice["finish_reason"] not in {"stop", "tool_calls"}:
                reason = "incomplete_or_refusal"
                raise ValueError()
            calls = message.get("tool_calls", [])
            if choice["finish_reason"] == "tool_calls" and not calls:
                raise ValueError()
            if not calls and not message.get("content") and not self._finished_empty(body, finalized):
                raise ValueError()
            usage = body["usage"]
            if any(type(usage.get(k)) is not int or usage[k] < 0 for k in ("prompt_tokens", "completion_tokens", "total_tokens")):
                raise ValueError()
        except (ValueError, KeyError, TypeError, IndexError):
            self.store.record_attempt(task["id"], owner, response=response, validation={"valid": False, "reason": reason})
            self.store.fail(task["id"], owner, {"kind": reason, "held": True})
            raise MiniProxyError("native response held: " + reason, reason=reason) from None
        self.store.record_attempt(task["id"], owner, response=response, usage=usage,
                                  validation={"valid": True, "response_model": model})
        self.store.succeed(task["id"], owner, body)
        return body
