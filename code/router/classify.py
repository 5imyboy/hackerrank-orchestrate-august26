"""The single Claude call that routes one message.

One call per message: the routing decision depends on sender trust, content,
timing and behavioural history simultaneously, so splitting it across calls would
throw away exactly the cross-signal reasoning that makes it personalised.

The cascade varies only the model. Same system prompt, same JSON schema, same
context block on both tiers -- escalation is a config decision, not a second
prompt to maintain.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any

from anthropic import AsyncAnthropic

from .context import MessageContext
from .schema import ACTIONS, MESSAGE_TYPES, Decision

# Hand-written rather than derived from Pydantic: structured outputs require
# `additionalProperties: false` and every property listed in `required`, and
# reject numeric constraints such as minimum/maximum. Writing it out keeps the
# wire schema honest instead of hoping a generator emits a supported subset.
DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "analysis": {
            "type": "string",
            "description": (
                "Two or three sentences weighing the decisive signals before you "
                "choose a label: who this sender is to this user, what the content "
                "is, what this user did with similar past messages, and any risk "
                "indicators."
            ),
        },
        "message_type": {
            "type": "string",
            "enum": list(MESSAGE_TYPES),
            "description": "The single best-fit category for this message.",
        },
        "action": {
            "type": "string",
            "enum": list(ACTIONS),
            "description": "notify to interrupt now, digest to show later, mute to suppress.",
        },
        "evidence_message_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Historical message IDs taken ONLY from the supplied candidate list, "
                "most relevant first. Usually exactly one. Empty if none is relevant."
            ),
        },
        "confidence": {
            "type": "number",
            "description": "Calibrated confidence between 0 and 1. See the rubric.",
        },
        "reason": {
            "type": "string",
            "description": (
                "One short third-person sentence under 25 words naming the decisive "
                "signal. No message IDs, no first person."
            ),
        },
    },
    "required": [
        "analysis",
        "message_type",
        "action",
        "evidence_message_ids",
        "confidence",
        "reason",
    ],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Tier:
    """One rung of the cascade."""

    name: str
    model: str
    max_tokens: int
    # Claude Haiku 4.5 rejects `effort` and uses the older budget_tokens style of
    # thinking, so tier 1 runs with neither. Claude Opus 5 gets adaptive thinking.
    thinking: bool = False
    effort: str | None = None


TIER1 = Tier(name="tier1", model="claude-haiku-4-5", max_tokens=2048)
TIER2 = Tier(
    name="tier2",
    model="claude-opus-5",
    max_tokens=8000,
    thinking=True,
    effort="high",
)


def sniff_media_type(data: bytes) -> str:
    """Detect the image format from its magic bytes.

    Not from the file extension: several files in this dataset are PNGs named
    `.jpg`, and the API rejects the request outright when the declared media type
    disagrees with the actual bytes.
    """
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def _image_block(ctx: MessageContext) -> dict[str, Any] | None:
    path = ctx.image_path
    if path is None or not path.exists():
        return None
    data = path.read_bytes()
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": sniff_media_type(data),
            "data": base64.standard_b64encode(data).decode("ascii"),
        },
    }


def build_user_content(ctx: MessageContext) -> list[dict[str, Any]]:
    """Image first, then the context block: the model should have looked at the
    poster before it reads the instruction to judge it."""
    content: list[dict[str, Any]] = []
    image = _image_block(ctx)
    if image is not None:
        content.append(image)
    content.append(
        {
            "type": "text",
            "text": (
                f"{ctx.render()}\n"
                "Route this message for this specific user. Cite evidence only from "
                "the candidate history above."
            ),
        }
    )
    return content


def _extract_text(response: Any) -> str:
    for block in response.content:
        if block.type == "text":
            return block.text
    raise ValueError("model returned no text block")


class Classifier:
    def __init__(self, client: AsyncAnthropic, system_prompt: str) -> None:
        self.client = client
        # A single cached system block. cache_control goes on the last block of the
        # stable prefix; everything volatile lives in the user turn after it.
        self.system = [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            }
        ]

    async def classify(self, ctx: MessageContext, tier: Tier) -> tuple[Decision, Any]:
        kwargs: dict[str, Any] = {
            "model": tier.model,
            "max_tokens": tier.max_tokens,
            "system": self.system,
            "messages": [{"role": "user", "content": build_user_content(ctx)}],
            "output_config": {
                "format": {"type": "json_schema", "schema": DECISION_SCHEMA}
            },
        }
        if tier.thinking:
            kwargs["thinking"] = {"type": "adaptive"}
        if tier.effort:
            kwargs["output_config"]["effort"] = tier.effort

        response = await self.client.messages.create(**kwargs)

        if response.stop_reason == "refusal":
            raise RuntimeError(
                f"{tier.model} declined to route {ctx.message_id} "
                f"({getattr(response, 'stop_details', None)})"
            )

        payload = json.loads(_extract_text(response))
        return Decision(**payload), response


def usage_of(response: Any) -> dict[str, int]:
    u = response.usage
    return {
        "input": getattr(u, "input_tokens", 0) or 0,
        "output": getattr(u, "output_tokens", 0) or 0,
        "cache_write": getattr(u, "cache_creation_input_tokens", 0) or 0,
        "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0,
    }


# Published per-million-token prices, used only for the cost report.
PRICES: dict[str, dict[str, float]] = {
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-opus-5": {"input": 5.00, "output": 25.00},
    "claude-sonnet-5": {"input": 3.00, "output": 15.00},
}


def cost_of(model: str, usage: dict[str, int]) -> float:
    price = PRICES.get(model)
    if not price:
        return 0.0
    return (
        usage["input"] * price["input"]
        + usage["cache_write"] * price["input"] * 1.25
        + usage["cache_read"] * price["input"] * 0.10
        + usage["output"] * price["output"]
    ) / 1_000_000
