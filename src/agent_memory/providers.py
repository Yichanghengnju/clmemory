"""Synchronous Magpie Messages gateway with durable, per-request attempts."""

from __future__ import annotations

import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import ssl
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Callable, Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import ProxyHandler, Request, build_opener
from uuid import uuid4

from .domain import Completion, ModelConfig


class ProviderError(RuntimeError):
    """Safe diagnostic; attempt_id links to the persisted response."""

    def __init__(self, message: str, *, attempt_id: str = "", reason: str = ""):
        super().__init__(message)
        self.attempt_id = attempt_id
        self.reason = reason


class TransientError(ProviderError):
    pass


class PermanentError(ProviderError):
    pass


class IncompleteError(ProviderError):
    pass


# A transport receives the actual urllib Request and its timeout, and returns
# (HTTP status, complete response bytes). Injected transports must not log headers.
Transport = Callable[[Request, float], tuple[int, bytes]]
_TRANSIENT_STATUS = {408, 425, 429, 500, 502, 503, 504, 529}
_INIT_LOCKS_GUARD = threading.Lock()
_INIT_LOCKS: dict[str, threading.Lock] = {}


def _initialization_lock(path: str) -> threading.Lock:
    identity = os.path.normcase(os.path.realpath(path))
    with _INIT_LOCKS_GUARD:
        return _INIT_LOCKS.setdefault(identity, threading.Lock())


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _redact(text: str, key: str) -> str:
    # Include JSON and percent-encoded echoes; never retain the credential in
    # raw responses, prompts, or exception diagnostics.
    if key:
        for value in sorted({key, json.dumps(key)[1:-1], quote(key, safe="")}, key=len, reverse=True):
            text = text.replace(value, "[REDACTED]")
    text = re.sub(r"(?i)\bBearer\s+[^\s\"',;<>]+", "Bearer [REDACTED]", text)
    text = re.sub(
        r'''(?i)(["']?(?:authorization|proxy-authorization|x-api-key|api[_-]?key|access_token)["']?\s*[:=]\s*)(["'])(.*?)(\2)''',
        lambda m: m[1] + m[2] + "[REDACTED]" + m[4], text,
    )
    text = re.sub(
        r"(?i)(\b(?:x-api-key|api[_-]?key|access_token)\s*[:=]\s*)[^\s\"'&,;<>]+",
        r"\1[REDACTED]", text,
    )
    text = re.sub(
        r"(?im)(\b(?:authorization|proxy-authorization)\s*:\s*)[^\r\n]+",
        r"\1[REDACTED]", text,
    )
    return re.sub(r"(https?://)[^\s/@]+:[^\s/@]+@", r"\1[REDACTED]@", text)


def _redact_json(text: str, key: str) -> str:
    """Redact JSON logs without treating escaped newlines as header text.

    The plaintext/Completion sanitizer above deliberately stays unchanged.
    """
    try:
        body = json.loads(text)
    except json.JSONDecodeError:
        return _redact(text, key)
    names = r"authorization|proxy-authorization|x-api-key|api[_-]?key|access_token"
    secrets = sorted({key, json.dumps(key)[1:-1], quote(key, safe="")}, key=len, reverse=True) if key else []

    def plain(value: str) -> str:
        for secret in secrets:
            value = value.replace(secret, "[REDACTED]")
        value = re.sub(
            rf'''(?i)(["']?(?:{names})["']?[ \t]*[:=][ \t]*)(["'])(?:\\.|(?!\2)[\s\S])*\2''',
            lambda m: m[1] + m[2] + "[REDACTED]" + m[2], value,
        )
        value = re.sub(r"(?i)\bBearer\s+[^\s\"',;<>]+", "Bearer [REDACTED]", value)
        value = re.sub(
            r"(?i)(\b(?:x-api-key|api[_-]?key|access_token)[ \t]*[:=][ \t]*)[^\s\"'&,;<>]+",
            r"\1[REDACTED]", value,
        )
        # Header lines may continue on indented lines, but never consume the
        # next source/observation or match an ordinary prose heading inline.
        value = re.sub(
            r"(?im)(^[ \t]*(?:authorization|proxy-authorization)[ \t]*:[ \t]*)\r?\n[ \t]+[^\r\n]+(?:\r?\n[ \t]+[^\r\n]+)*",
            r"\1[REDACTED]", value,
        )
        value = re.sub(
            r"(?im)(^[ \t]*(?:authorization|proxy-authorization)[ \t]*:[ \t]*)[^\r\n]+",
            r"\1[REDACTED]", value,
        )
        value = re.sub(
            r"(?i)(\b(?:authorization|proxy-authorization)[ \t]*:[ \t]*(?:Basic|Digest)[ \t]+)[^\r\n]+",
            r"\1[REDACTED]", value,
        )
        return re.sub(r"(https?://)[^\s/@]+:[^\s/@]+@", r"\1[REDACTED]@", value)

    def walk(value):
        if isinstance(value, dict):
            return {plain(name): "[REDACTED]" if re.fullmatch(names, name, re.I) else walk(item)
                    for name, item in value.items()}
        if isinstance(value, list):
            return [walk(item) for item in value]
        if isinstance(value, str):
            # ReFind stores its conversation transcript as a JSON string
            # inside a message; preserve that structure at every log layer.
            try:
                nested = json.loads(value)
            except json.JSONDecodeError:
                return plain(value)
            if isinstance(nested, (dict, list)):
                redacted = walk(nested)
                return value if redacted == nested else _json(redacted)
            return plain(value)
        return "[REDACTED]" if any(secret in _json(value) for secret in secrets) else value

    redacted = walk(body)
    return text if redacted == body else _json(redacted)


def _transport_diagnostic(error: BaseException) -> dict:
    """Keep useful socket/protocol types without exception args or credentials."""
    reason = error.reason if isinstance(error, URLError) else error
    if isinstance(reason, ssl.SSLCertVerificationError):
        kind = "tls_certificate"
    elif isinstance(reason, http.client.RemoteDisconnected):
        kind = "peer_closed_without_response"
    elif isinstance(reason, http.client.IncompleteRead):
        kind = "truncated_response"
    elif isinstance(reason, TimeoutError):
        kind = "timeout"
    elif isinstance(reason, ConnectionRefusedError):
        kind = "connection_refused"
    elif isinstance(reason, ConnectionResetError):
        kind = "connection_reset"
    elif isinstance(reason, http.client.HTTPException):
        kind = "http_protocol"
    elif isinstance(reason, OSError):
        kind = "os_error"
    elif isinstance(error, URLError):
        kind = "url_error"
    else:
        kind = "unexpected_transport"
    return {"kind": kind, "exception_type": type(error).__name__, "reason_type": type(reason).__name__,
            **{name: value for name in ("errno", "winerror")
               if type(value := getattr(reason, name, None)) is int}}


class MagpieProvider:
    def __init__(
        self,
        attempt_db: str | Path,
        *,
        url: str = "http://127.0.0.1:3425/v1/messages",
        proxy: str | None = "http://180.209.2.29:7890",
        transport: Transport | None = None,
        retry_incomplete: bool = False,
    ):
        self._check_url(url, proxy=False)
        if proxy is not None:
            self._check_url(proxy, proxy=True)
        self.url = url
        self.proxy = proxy
        self.attempt_db = str(attempt_db)
        if self.attempt_db == ":memory:":
            raise PermanentError("attempt_db must be a persistent SQLite file", reason="configuration")
        Path(self.attempt_db).parent.mkdir(parents=True, exist_ok=True)
        self._transport = transport or self._send
        self.retry_incomplete = retry_incomplete
        # Serialize only initialization for this path, never requests or network
        # waits. Other processes still contend, so retry SQLITE_BUSY/LOCKED with
        # a fresh short-lived connection and a bounded backoff.
        with _initialization_lock(self.attempt_db):
            for retry in range(5):
                try:
                    self._initialize()
                    break
                except sqlite3.OperationalError as exc:
                    code = getattr(exc, "sqlite_errorcode", None)
                    if code is None or (code & 0xFF) not in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED) or retry == 4:
                        raise
                    time.sleep(0.05 * 2 ** retry)

    def _initialize(self) -> None:
        # Reading the mode avoids requesting another journal-mode transition on
        # every instance after WAL is already installed.
        with self._connection(timeout=0.25) as db:
            mode = db.execute("PRAGMA journal_mode").fetchone()[0]
            if mode.lower() != "wal":
                db.execute("PRAGMA journal_mode=WAL")
            db.execute("""
                CREATE TABLE IF NOT EXISTS attempts (
                    attempt_id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL,
                    request_hash TEXT NOT NULL,
                    attempt_number INTEGER NOT NULL,
                    model TEXT NOT NULL,
                    url TEXT NOT NULL,
                    proxy TEXT,
                    request_json TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    valid INTEGER NOT NULL DEFAULT 0,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    duration_s REAL,
                    http_status INTEGER,
                    raw_response TEXT,
                    text TEXT,
                    usage_json TEXT,
                    stop_reason TEXT,
                    response_model TEXT,
                    error_type TEXT,
                    error TEXT,
                    validation_reason TEXT,
                    transport_error_json TEXT
                )
            """)
            if "transport_error_json" not in {row[1] for row in db.execute("PRAGMA table_info(attempts)")}:
                try:
                    db.execute("ALTER TABLE attempts ADD COLUMN transport_error_json TEXT")
                except sqlite3.OperationalError:
                    # A different process may have completed the same migration.
                    if "transport_error_json" not in {row[1] for row in db.execute("PRAGMA table_info(attempts)")}:
                        raise

    @staticmethod
    def _check_url(value: str, *, proxy: bool) -> None:
        try:
            parsed = urlsplit(value)
            valid = (
                parsed.scheme in ("http", "https") and parsed.hostname
                and parsed.port != 0 and parsed.username is None
                and parsed.password is None and not parsed.fragment
            )
            if proxy:
                valid = valid and parsed.scheme == "http" and parsed.path in ("", "/") and not parsed.query
            if not valid:
                raise ValueError
        except (TypeError, ValueError):
            raise PermanentError("Invalid endpoint or HTTP forward proxy", reason="configuration") from None

    @contextmanager
    def _connection(self, *, timeout: float = 10) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.attempt_db, timeout=timeout)
        try:
            db.execute("PRAGMA foreign_keys=ON")
            db.execute(f"PRAGMA busy_timeout={int(timeout * 1000)}")
            with db:
                yield db
        finally:
            db.close()

    def _update(self, attempt_id: str, **fields: object) -> None:
        with self._connection() as db:
            db.execute(
                "UPDATE attempts SET " + ",".join(f"{name}=?" for name in fields) + " WHERE attempt_id=?",
                (*fields.values(), attempt_id),
            )

    @staticmethod
    def _send(request: Request, timeout: float) -> tuple[int, bytes]:
        # Environment proxies/NO_PROXY cannot bypass the explicit forward proxy,
        # including for loopback destinations. HTTPS uses urllib's CONNECT tunnel.
        opener = build_opener(ProxyHandler({}))
        with opener.open(request, timeout=timeout) as response:
            return response.status, response.read()

    @staticmethod
    def _payload(config: ModelConfig, prompt: str, system: str) -> dict:
        if (
            not isinstance(config.model, str) or not config.model.strip()
            or not isinstance(prompt, str) or not prompt.strip()
            or not isinstance(system, str)
            or type(config.max_tokens) is not int or config.max_tokens <= 0
            or type(config.retries) is not int or config.retries < 0
            or isinstance(config.timeout_s, bool) or not isinstance(config.timeout_s, (int, float))
            or not math.isfinite(config.timeout_s) or config.timeout_s <= 0
            or (config.effort is not None and (not isinstance(config.effort, str) or not config.effort.strip()))
        ):
            raise PermanentError("Invalid model request configuration", reason="configuration")
        payload = {
            "model": config.model,
            "max_tokens": config.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system
        if config.effort is not None:
            payload["thinking"] = {"type": "adaptive"}
            payload["output_config"] = {"effort": config.effort}
        if config.temperature is not None:
            if (
                isinstance(config.temperature, bool) or not isinstance(config.temperature, (int, float))
                or not math.isfinite(config.temperature) or not 0 <= config.temperature <= 1
            ):
                raise PermanentError("Invalid temperature", reason="configuration")
            payload["temperature"] = config.temperature
        return payload

    @staticmethod
    def _budget_exhausted(body: object) -> bool:
        if not isinstance(body, dict) or not isinstance(body.get("error"), dict):
            return False
        error = body["error"]
        details = " ".join(str(error.get(name, "")) for name in ("type", "code", "message")).lower()
        return any(marker in details for marker in (
            "insufficient_quota", "quota_exceeded", "quota exhausted", "quota exceeded",
            "budget_exhausted", "budget exhausted", "budget exceeded", "budget limit",
            "insufficient credit", "credit balance", "billing", "payment required",
        ))

    @staticmethod
    def _policy_refused(body: object) -> bool:
        # Gateways can wrap an explicit upstream refusal in HTTP 502. Inspect
        # only the error envelope, never normal generated text or prompt echoes.
        if not isinstance(body, dict):
            return False
        error = body.get("error")
        if isinstance(error, dict):
            # Explicit context errors take precedence over policy words in a message.
            if any(str(error.get(name, "")).strip().lower() in (
                "context_length_exceeded", "context_window_exceeded",
            ) for name in ("type", "code")):
                return False
            details = " ".join(str(error.get(name, "")) for name in ("type", "code", "message"))
        elif isinstance(error, str):
            details = error
        else:
            return False
        details = details.lower()
        return any(marker in details for marker in (
            "policy_refusal", "policy refusal", "content_policy_violation",
            "flagged as potentially violating our usage policy",
        ))

    def complete(self, *, config: ModelConfig, prompt: str, system: str = "") -> Completion:
        payload = self._payload(config, prompt, system)
        # This is the sole credential source. It is never stored in a header log.
        key = os.environ.get("MAGPIE_API_KEY") or ""
        payload_json = _json(payload)
        config_json = _json(asdict(config))
        request_hash = hashlib.sha256(_json({"url": self.url, "payload": payload}).encode()).hexdigest()
        request_id = uuid4().hex
        for attempt_number in range(1, config.retries + 2):
            attempt_id = uuid4().hex
            started = time.monotonic()
            with self._connection() as db:
                db.execute("""
                    INSERT INTO attempts (
                        attempt_id, request_id, request_hash, attempt_number, model,
                        url, proxy, request_json, config_json, status, started_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', ?)
                """, (
                    attempt_id, request_id, request_hash, attempt_number, _redact(config.model, key),
                    _redact(self.url, key), _redact(self.proxy, key) if self.proxy else None,
                    _redact_json(payload_json, key), _redact_json(config_json, key), _now(),
                ))
            headers = {
                "Content-Type": "application/json", "Accept": "application/json",
                "anthropic-version": "2023-06-01",
            }
            if key:
                headers["x-api-key"] = key
            request = Request(self.url, data=payload_json.encode("utf-8"), headers=headers, method="POST")
            if self.proxy:
                parsed_proxy = urlsplit(self.proxy)
                request.set_proxy(parsed_proxy.netloc, parsed_proxy.scheme)
            failure: ProviderError | None = None
            transport_detail = None
            status: int | None = None
            raw: bytes | None = None
            try:
                status, raw = self._transport(request, config.timeout_s)
            except HTTPError as exc:
                status = exc.code
                try:
                    raw = exc.read()
                except http.client.IncompleteRead as exc:
                    transport_detail = _transport_diagnostic(exc)
                    raw = exc.partial
                    failure = TransientError("HTTP response body was truncated", reason="transport")
                except (OSError, http.client.HTTPException) as exc:
                    transport_detail = _transport_diagnostic(exc)
                    failure = TransientError("HTTP response body could not be read", reason="transport")
                finally:
                    exc.close()
            except http.client.IncompleteRead as exc:
                transport_detail = _transport_diagnostic(exc)
                raw = exc.partial
                failure = TransientError("HTTP response body was truncated", reason="transport")
            except (URLError, OSError, http.client.HTTPException) as exc:
                transport_detail = _transport_diagnostic(exc)
                if isinstance(exc, ssl.SSLCertVerificationError) or (
                    isinstance(exc, URLError) and isinstance(exc.reason, ssl.SSLCertVerificationError)
                ):
                    failure = PermanentError("TLS certificate verification failed", reason="transport")
                else:
                    failure = TransientError("HTTP transport failed", reason="transport")
            except Exception as exc:
                transport_detail = _transport_diagnostic(exc)
                # Never propagate arbitrary transport exception strings or args.
                failure = PermanentError("Unexpected transport failure", reason="transport")
            if transport_detail is not None:
                self._update(attempt_id, transport_error_json=_redact_json(_json(transport_detail), key))
            if failure is not None and isinstance(raw, bytes):
                self._update(
                    attempt_id, raw_response=_redact_json(raw.decode("utf-8", errors="replace"), key),
                )
            if failure is None:
                if type(status) is not int or not isinstance(raw, bytes):
                    failure = PermanentError("Invalid transport response", reason="protocol")
                else:
                    # Commit the full response before JSON/schema/end-state validation.
                    raw_text = _redact_json(raw.decode("utf-8", errors="replace"), key)
                    self._update(attempt_id, status="received", http_status=status, raw_response=raw_text)
                    failure, completion = self._validate(attempt_id, status, raw, key, config.model)
                    if failure is None:
                        self._update(
                            attempt_id, status="successful", valid=1, finished_at=_now(),
                            duration_s=time.monotonic() - started,
                        )
                        return completion
            assert failure is not None
            failure.attempt_id = attempt_id
            self._update(
                attempt_id, status={TransientError: "transient_error", PermanentError: "permanent_error",
                                    IncompleteError: "incomplete"}[type(failure)],
                http_status=status if type(status) is int else None,
                finished_at=_now(), duration_s=time.monotonic() - started,
                error_type=type(failure).__name__, error=str(failure), validation_reason=failure.reason,
            )
            retryable = isinstance(failure, TransientError) or (
                self.retry_incomplete and (
                    isinstance(failure, IncompleteError)
                    and failure.reason in {"empty_text", "non_end_stop", "transport"}
                    or isinstance(failure, PermanentError) and failure.reason == "protocol"
                )
            )
            if not retryable or attempt_number > config.retries:
                # Raise outside the except blocks so secret-bearing exceptions are
                # not retained as the public exception's context.
                raise failure from None
            time.sleep(min(0.25 * 2 ** min(attempt_number - 1, 5), 2.0))
        raise AssertionError("unreachable")

    def _validate(
        self, attempt_id: str, status: int, raw: bytes, key: str, requested_model: str,
    ) -> tuple[ProviderError | None, Completion | None]:
        try:
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            body = None
        if isinstance(body, dict):
            content = body.get("content")
            text = "".join(
                block["text"] for block in content
                if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
            ) if isinstance(content, list) else ""
            usage = body.get("usage", {})
            stop = body.get("stop_reason")
            model = body.get("model", requested_model)
            self._update(
                attempt_id, text=_redact(text, key), usage_json=_redact_json(_json(usage), key),
                stop_reason=_redact(stop, key) if isinstance(stop, str) else None,
                response_model=_redact(model, key) if isinstance(model, str) else None,
            )
        else:
            content, text, usage, stop, model = None, "", {}, None, requested_model
        if self._policy_refused(body):
            return PermanentError("Provider policy refusal", reason="policy_refusal"), None
        if self._budget_exhausted(body):
            return PermanentError("Provider budget exhausted", reason="budget_exhausted"), None
        if not 200 <= status < 300:
            error_cls = TransientError if status in _TRANSIENT_STATUS else PermanentError
            reason = "rate_limit" if status == 429 else "http"
            return error_cls(f"Provider HTTP {status}", reason=reason), None
        if not isinstance(body, dict) or body.get("type") == "error":
            return PermanentError("Invalid Messages response", reason="protocol"), None
        if not isinstance(content, list) or any(
            not isinstance(block, dict) or (
                block.get("type") == "text" and not isinstance(block.get("text"), str)
            ) for block in content
        ):
            return PermanentError("Invalid content blocks", reason="protocol"), None
        if not isinstance(usage, dict) or any(
            type(value) is not int or value < 0
            for name, value in usage.items() if name.endswith("_tokens")
        ) or not isinstance(model, str) or not model.strip():
            return PermanentError("Invalid usage or model metadata", reason="protocol"), None
        if stop != "end_turn":
            reason = "budget_exhausted" if stop == "max_tokens" else "non_end_stop"
            return IncompleteError("Model response did not end normally", reason=reason), None
        if not text.strip():
            return IncompleteError("Model response has no text", reason="empty_text"), None
        # Preserve reported token counts, including cache reads/creation. Nested
        # provider metadata remains intact in usage_json and raw_response.
        token_usage = {name: value for name, value in usage.items() if name.endswith("_tokens")}
        return None, Completion(
            text=_redact(text, key), stop_reason=stop, usage=token_usage,
            model=_redact(model, key), attempt_id=attempt_id,
        )
