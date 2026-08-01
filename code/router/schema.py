"""Output contract for the router.

The column set and allowed values are fixed by problem_statement.md; this module
is the single place they are declared so the Pydantic schema handed to the model
and the CSV writer can never drift apart.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Action = Literal["notify", "digest", "mute"]

MessageType = Literal[
    "personal",
    "urgent",
    "event",
    "payment",
    "business_update",
    "promotion",
    "greeting",
    "forward",
    "spam",
    "scam",
    "unknown",
]

ACTIONS: tuple[str, ...] = ("notify", "digest", "mute")
MESSAGE_TYPES: tuple[str, ...] = (
    "personal",
    "urgent",
    "event",
    "payment",
    "business_update",
    "promotion",
    "greeting",
    "forward",
    "spam",
    "scam",
    "unknown",
)

# Exact column order required by the submission.
OUTPUT_COLUMNS: tuple[str, ...] = (
    "message_id",
    "action",
    "message_type",
    "reason",
    "confidence",
    "evidence_message_ids",
)


class Decision(BaseModel):
    """One routing decision. Field order is deliberate: `analysis` comes first so
    the model commits to its reading of the evidence before naming a label, which
    measurably helps the non-thinking tier-1 model. `analysis` is never written to
    the CSV."""

    analysis: str = Field(
        description=(
            "Two or three sentences weighing the decisive signals: who the sender is "
            "to this user, what the content is, what this user did with similar past "
            "messages, and any risk indicators. Reason here before choosing a label."
        )
    )
    message_type: MessageType = Field(
        description="The single best-fit category for this message's content."
    )
    action: Action = Field(
        description=(
            "notify = interrupt now; digest = useful but can wait; "
            "mute = low-value, repetitive, unwanted, suspicious or unsafe."
        )
    )
    evidence_message_ids: list[str] = Field(
        description=(
            "Historical message IDs from the supplied candidate list that justify this "
            "decision, most relevant first. Usually exactly one. Empty list if no "
            "candidate is genuinely relevant."
        )
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Calibrated confidence. Use 0.85-0.92 for clear-cut calls, 0.78-0.85 when "
            "the signals mostly agree, and below 0.75 when genuinely torn."
        ),
    )
    reason: str = Field(
        description=(
            "One short third-person sentence explaining the decision to a reviewer, "
            "e.g. 'A trusted group admin sent a time-sensitive update that should "
            "interrupt the user.' No message IDs, no first person, under 25 words."
        )
    )


class RoutedMessage(BaseModel):
    """A decision plus the provenance needed for reporting and reproducibility."""

    message_id: str
    action: Action
    message_type: MessageType
    reason: str
    confidence: float
    evidence_message_ids: list[str]
    # Provenance -- not part of the submitted CSV.
    decided_by: str = "llm"
    model: str | None = None
    escalated: bool = False
    escalation_reason: str | None = None
    repaired: list[str] = Field(default_factory=list)

    def to_csv_row(self) -> dict[str, str]:
        return {
            "message_id": self.message_id,
            "action": self.action,
            "message_type": self.message_type,
            "reason": self.reason,
            "confidence": f"{self.confidence:.2f}",
            "evidence_message_ids": ";".join(self.evidence_message_ids)
            if self.evidence_message_ids
            else "none",
        }
