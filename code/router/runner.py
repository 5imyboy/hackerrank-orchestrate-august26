"""Async driver for the cascade.

Concurrency is a bounded semaphore over stateless per-message work, so the same
code path serves a live stream and a bulk backfill. Every model response is cached
on disk keyed by a hash of the exact request, which is what makes re-runs both
free and byte-for-byte reproducible -- the determinism the project contract asks
for, given that the current models reject `temperature` outright.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .classify import Classifier, Tier, build_user_content, cost_of, usage_of
from .context import ContextBuilder, MessageContext
from .loaders import Row
from .rules import as_routed, escalation_reason, pre_pass, validate
from .schema import Decision, RoutedMessage


def _fingerprint(system_prompt: str, ctx: MessageContext, tier: Tier) -> str:
    """Identify a request by everything that can change its answer."""
    hasher = hashlib.sha256()
    for part in (tier.model, str(tier.thinking), str(tier.effort), system_prompt):
        hasher.update(part.encode("utf-8"))
        hasher.update(b"\x00")
    for block in build_user_content(ctx):
        if block["type"] == "text":
            hasher.update(block["text"].encode("utf-8"))
        else:  # image -- hash the bytes, not the whole base64 payload
            hasher.update(block["source"]["data"].encode("ascii"))
        hasher.update(b"\x00")
    return hasher.hexdigest()


class DecisionCache:
    """Disk-backed cache of fingerprint -> raw decision payload."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.entries: dict[str, dict[str, Any]] = {}
        if path.exists():
            self.entries = json.loads(path.read_text(encoding="utf-8"))
        self._lock = asyncio.Lock()
        self._dirty = False

    def get(self, key: str) -> Decision | None:
        entry = self.entries.get(key)
        if entry is None:
            return None
        try:
            return Decision(**entry["decision"])
        except Exception:
            return None

    async def put(self, key: str, decision: Decision, meta: dict[str, Any]) -> None:
        async with self._lock:
            self.entries[key] = {"decision": decision.model_dump(), **meta}
            self._dirty = True

    def save(self) -> None:
        if not self._dirty and self.path.exists():
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.entries, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )


@dataclass
class RunStats:
    total: int = 0
    by_decider: Counter = field(default_factory=Counter)
    escalations: Counter = field(default_factory=Counter)
    repairs: Counter = field(default_factory=Counter)
    cache_hits: int = 0
    api_calls: int = 0
    cost_usd: float = 0.0
    cached_input_tokens: int = 0
    uncached_input_tokens: int = 0
    seconds: float = 0.0

    def report(self) -> str:
        escalated = sum(self.escalations.values())
        llm = self.total - self.by_decider.get("rule", 0)
        lines = [
            "",
            "=" * 62,
            f"routed {self.total} messages in {self.seconds:.1f}s",
            f"  decided by rule (no API call): {self.by_decider.get('rule', 0)}",
            f"  decided by tier 1 (haiku):     {self.by_decider.get('tier1', 0)}",
            f"  decided by tier 2 (opus):      {self.by_decider.get('tier2', 0)}",
            f"  escalation rate: {escalated}/{llm} "
            f"({(escalated / llm * 100) if llm else 0:.0f}% of model-routed messages)",
        ]
        if self.escalations:
            for reason, count in self.escalations.most_common():
                lines.append(f"    {reason}: {count}")
        if self.repairs:
            lines.append("  validator repairs:")
            for repair, count in self.repairs.most_common(8):
                lines.append(f"    {repair}: {count}")
        cached = self.cached_input_tokens + self.uncached_input_tokens
        hit_rate = (self.cached_input_tokens / cached * 100) if cached else 0.0
        lines += [
            f"  api calls: {self.api_calls}, cache hits (disk): {self.cache_hits}",
            f"  prompt-cache read: {self.cached_input_tokens} tokens ({hit_rate:.0f}% of input)",
            f"  cost this run: ${self.cost_usd:.4f}"
            + (
                f"  (${self.cost_usd / self.total:.5f}/message)"
                if self.total
                else ""
            ),
            "=" * 62,
        ]
        return "\n".join(lines)


class Runner:
    def __init__(
        self,
        builder: ContextBuilder,
        classifier: Classifier,
        system_prompt: str,
        cache: DecisionCache,
        *,
        tier1: Tier,
        tier2: Tier,
        cascade: bool = True,
        rules: bool = True,
        concurrency: int = 8,
    ) -> None:
        self.builder = builder
        self.classifier = classifier
        self.system_prompt = system_prompt
        self.cache = cache
        self.tier1 = tier1
        self.tier2 = tier2
        self.cascade = cascade
        self.rules = rules
        self.sem = asyncio.Semaphore(concurrency)
        self.stats = RunStats()

    async def _call(self, ctx: MessageContext, tier: Tier) -> Decision:
        key = _fingerprint(self.system_prompt, ctx, tier)
        cached = self.cache.get(key)
        if cached is not None:
            self.stats.cache_hits += 1
            return cached

        async with self.sem:
            decision, response = await self.classifier.classify(ctx, tier)

        usage = usage_of(response)
        self.stats.api_calls += 1
        self.stats.cost_usd += cost_of(tier.model, usage)
        self.stats.cached_input_tokens += usage["cache_read"]
        self.stats.uncached_input_tokens += usage["input"] + usage["cache_write"]
        await self.cache.put(
            key, decision, {"model": tier.model, "message_id": ctx.message_id}
        )
        return decision

    async def route_one(self, msg: Row) -> RoutedMessage:
        ctx = self.builder.build(msg)

        if self.rules:
            hit = pre_pass(ctx)
            if hit is not None:
                self.stats.by_decider["rule"] += 1
                return RoutedMessage(
                    message_id=ctx.message_id,
                    action=hit.action,  # type: ignore[arg-type]
                    message_type=hit.message_type,  # type: ignore[arg-type]
                    reason=hit.reason,
                    confidence=hit.confidence,
                    evidence_message_ids=hit.evidence_message_ids,
                    decided_by=f"rule:{hit.rule}",
                    model=None,
                )

        decision = await self._call(ctx, self.tier1)
        decision, repairs, validator_reason = validate(decision, ctx)
        for repair in repairs:
            self.stats.repairs[repair.split(":")[0]] += 1

        reason = escalation_reason(decision, ctx, validator_reason)
        if reason and self.cascade:
            self.stats.escalations[reason] += 1
            try:
                upgraded = await self._call(ctx, self.tier2)
                upgraded, up_repairs, _ = validate(upgraded, ctx)
                for repair in up_repairs:
                    self.stats.repairs[repair.split(":")[0]] += 1
                self.stats.by_decider["tier2"] += 1
                return as_routed(
                    ctx.message_id,
                    upgraded,
                    decided_by="tier2",
                    model=self.tier2.model,
                    escalated=True,
                    escalation_reason_=reason,
                    repairs=up_repairs,
                )
            except Exception as exc:  # noqa: BLE001 -- tier 1 answer is still valid
                print(
                    f"[warn] escalation failed for {ctx.message_id}: {exc!r}; "
                    "keeping the tier-1 decision",
                    file=sys.stderr,
                )
        elif reason:
            self.stats.escalations[f"{reason} (cascade off)"] += 1

        self.stats.by_decider["tier1"] += 1
        return as_routed(
            ctx.message_id,
            decision,
            decided_by="tier1",
            model=self.tier1.model,
            escalated=False,
            escalation_reason_=reason,
            repairs=repairs,
        )

    async def route_all(self, messages: list[Row]) -> list[RoutedMessage]:
        started = time.monotonic()
        self.stats.total = len(messages)

        # Group by user so each user's profile/history prefix is warm in the
        # prompt cache across their own messages. Harmless for a 110-row batch,
        # and it is the dominant saving at production volume.
        ordered = sorted(messages, key=lambda m: (m["user_id"], m["message_id"]))

        results = await asyncio.gather(
            *(self.route_one(msg) for msg in ordered), return_exceptions=True
        )

        routed: list[RoutedMessage] = []
        failures = 0
        for msg, result in zip(ordered, results):
            if isinstance(result, BaseException):
                failures += 1
                print(
                    f"[error] {msg['message_id']} failed: {result!r}", file=sys.stderr
                )
                # Never drop a row: the contract requires one prediction per input.
                routed.append(
                    RoutedMessage(
                        message_id=msg["message_id"],
                        action="digest",
                        message_type="unknown",
                        reason="Routing failed for this message; held for later review.",
                        confidence=0.30,
                        evidence_message_ids=[],
                        decided_by="fallback",
                    )
                )
            else:
                routed.append(result)

        if failures:
            print(f"[error] {failures} messages fell back to digest/unknown", file=sys.stderr)

        self.cache.save()
        self.stats.seconds = time.monotonic() - started
        return routed
