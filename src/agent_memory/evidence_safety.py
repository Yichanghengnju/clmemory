"""Source visibility for the new ensemble recipe; no answer or refusal generation.

Only explicit, standalone user instructions have delivery semantics. Corrections,
historical facts, assistant prose, quoted instructions, and sensitive vocabulary
alone do not revoke anything. Raw storage is immutable. This deliberately narrow
parser exposes unsupported language as unknown, not as proven unsupported facts.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re

from .domain import Evidence
from .anchor_channel import _render_source

SAFETY_VERSION = "textual-source-visibility-v1"
_DIRECTIVE = re.compile(
    r'^(?:please\s+)?(?:forget|never disclose|do not (?:disclose|reveal|share)|'
    r"don['’]t (?:disclose|reveal|share))\s+(.+?)[.!。]?$", re.I)
_CN_DIRECTIVE = re.compile(r'^(?:请)?(?:忘记|不要再(?:披露|透露|提及)|禁止(?:披露|透露))(.+?)[。！]?$')
_QUOTED = re.compile(r'^["“](.+)["”]$')
_FIELDS = {"password": r"(?:password|passcode)", "address": r"(?:home address|address)",
           "phone number": r"(?:phone number|telephone number)", "email": r"(?:email address|email)",
           "ssn": r"(?:social security number|ssn)"}


@dataclass(frozen=True)
class GateDecision:
    support: str  # supported | unknown | proven_no_deliverable_sources
    reasons: tuple[str, ...] = ()


class Visibility:
    """A server-bound complete user scope, preserving legal cross-session reads."""

    def __init__(self, user_id: str, sources: list[dict]):
        self.user_id = user_id
        if any(s.get("user_id") != user_id for s in sources):
            raise ValueError("source scope violation")
        self.sources = {s["id"]: dict(s) for s in sources}
        self.blocked: dict[str, str] = {}
        self.literals: list[str] = []
        prior = []
        for source in sources:
            text = source["content"].strip()
            match = (_DIRECTIVE.fullmatch(text) or _CN_DIRECTIVE.fullmatch(text)) if source["role"] == "user" else None
            if not match:
                prior.append(source)
                continue
            target = match.group(1).strip()
            quoted = _QUOTED.fullmatch(target)
            targets = []
            if target.lower() in {"my previous message", "the previous message", "上一条消息", "我上一条消息"}:
                targets = [s for s in prior if s["role"] == "user"][-1:]
            elif target.startswith("source:"):
                targets = [s for s in prior if s["id"] == target[7:].strip()]
            elif quoted:
                literal = quoted.group(1)
                if len(literal) >= 3:
                    self.literals.append(literal.casefold())
                    targets = [s for s in prior if literal.casefold() in s["content"].casefold()]
            elif target.lower().removeprefix("my ").removesuffix(" again") in _FIELDS:
                field = target.lower().removeprefix("my ").removesuffix(" again")
                pattern = re.compile(r"\b(?:my\s+)?" + _FIELDS[field] + r"\s*(?:is|was|:|=)\s*([^\n]+)", re.I)
                for s in prior:
                    found = pattern.search(s["content"])
                    if found:
                        targets.append(s)
                        literal = found.group(1).strip().rstrip(".!。")
                        if len(literal) >= 3:
                            self.literals.append(literal.casefold())
            # A recognized control is withheld itself (it can repeat the secret).
            # Unresolved referents remain unknown and never wipe the whole corpus.
            self.blocked[source["id"]] = "explicit_delivery_control"
            for previous in targets:
                self.blocked[previous["id"]] = "explicit_user_withdrawal:" + source["id"]
                self.literals.append(previous["content"].casefold())
            prior.append(source)
        # Also block later copies of an explicitly withdrawn literal. Corrections
        # use a different path: no inferred deletion or keyword-level broad gate.
        for sid, source in self.sources.items():
            if any(value in source["content"].casefold() for value in self.literals):
                self.blocked.setdefault(sid, "withdrawn_literal_copy")

    @property
    def snapshot(self):
        return hashlib.sha256(json.dumps(
            [self.user_id, list(self.sources.values()), self.blocked, SAFETY_VERSION],
            ensure_ascii=False, sort_keys=True).encode()).hexdigest()

    def visible(self, sid):
        return sid in self.sources and sid not in self.blocked

    def read(self, sid):
        if not self.visible(sid):
            raise ValueError("source not available in this scope")
        return self.sources[sid]

    def filter(self, hits: list[Evidence]):
        """Never trust a branch's content or source IDs; rehydrate exact sources.

        Reject the whole mixed/derived candidate if any dependency is hidden.
        This prevents a hidden fact from surviving in a derived text or cache.
        """
        result, reasons, seen = [], [], set()
        for hit in hits:
            refs = tuple(dict.fromkeys(hit.source_ids))
            reason = None
            if not refs or any(sid not in self.sources for sid in refs):
                reason = "missing_or_foreign_source"
            elif any(sid in self.blocked for sid in refs):
                reason = "withdrawn_dependency"
            elif any(literal in hit.content.casefold() for literal in self.literals):
                reason = "withdrawn_content_copy"
            if reason:
                reasons.append({"id": hit.id, "reason": reason})
                continue
            fresh = tuple(sid for sid in refs if sid not in seen)
            if not fresh:
                continue
            seen.update(fresh)
            result.append(Evidence(id=hit.id, source_ids=fresh, score=hit.score,
                                   content="\n\n".join(_render_source(self.read(s)) for s in fresh),
                                   kind=hit.kind))
        return result, reasons

    def gate(self, hits):
        # No query keyword classifier and no similarity/zero-hit rejection.
        if hits:
            return GateDecision("supported")
        if not any(self.visible(sid) for sid in self.sources):
            return GateDecision("proven_no_deliverable_sources", ("empty_visible_scope",))
        return GateDecision("unknown", ("retrieval_returned_no_evidence",))
