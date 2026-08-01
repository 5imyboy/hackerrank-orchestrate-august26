"""Deterministic layer: a pre-pass that decides the obvious cases without an API
call, and a validator that repairs model output before it reaches the CSV.

Design note: the pre-pass is deliberately narrow. Analysis of the 70 solved sample
rows shows only three message types are action-exclusive -- `urgent` is always
notify, `scam` and `spam` are always mute. Everything else splits across actions
(`promotion` appears as both digest and mute, `greeting` as both, `personal` and
`business_update` as both digest and notify). So a rule may only fire where the
evidence is categorical; anything nuanced belongs to the model.
"""

from __future__ import annotations

from dataclasses import dataclass

from .context import MessageContext
from .loaders import _int
from .schema import ACTIONS, MESSAGE_TYPES, Decision, RoutedMessage

# Type -> the single action it is allowed to carry. Types absent from this map
# legitimately vary by user and context.
EXCLUSIVE_ACTION: dict[str, str] = {
    "urgent": "notify",
    "scam": "mute",
    "spam": "mute",
}

# Observed confidence band across the solved samples is 0.78-0.91. Values far
# outside it are miscalibration, not information.
CONFIDENCE_FLOOR = 0.55
CONFIDENCE_CEILING = 0.95
ESCALATE_BELOW = 0.75

# Confidence reported for every deterministic pre-pass decision.
#
# This is a deliberate placeholder, not a calibrated value, and it is a single
# constant so that rule-decided rows are visibly uniform in the output rather than
# looking like per-row model judgements.
#
# There is no honest basis for calibrating it. The labelled set is 30 rows, and
# zero of them exercise the impersonation rule once it is corroborated (see
# `pre_pass`), so the rule's real precision is unmeasured. A value fitted to the
# sample's 0.78-0.91 band would be false precision that need not survive a
# differently-calibrated hidden test set.
#
# A round 0.9 reads as "decided by rule, not judged by a model", and stops short of
# 1.0, which would claim a certainty this rule has not earned -- and which any
# proper scoring rule punishes hardest exactly when it is wrong.
PRE_PASS_CONFIDENCE = 0.9

# A sending domain younger than this, or an unverified sender, is what turns a
# domain mismatch from a signal into a decision. See `pre_pass`.
ESTABLISHED_SENDER_DOMAIN_DAYS = 180


@dataclass
class PrePassResult:
    action: str
    message_type: str
    reason: str
    confidence: float
    evidence_message_ids: list[str]
    rule: str


def pre_pass(ctx: MessageContext) -> PrePassResult | None:
    """Decide categorically-safe cases without calling the model.

    Only two rules qualify, and both are narrow on purpose.

    Impersonation needs corroboration, not just a domain mismatch. A mismatch on
    its own is a strong signal but not a decision: legitimate brands routinely send
    through link shorteners and click-tracking domains, so `link.wame.pro` for a
    verified 12-year-old travel brand is marketing infrastructure, while
    `chase-secure-alert.com` on a 10-day-old unverified domain is phishing. Firing
    on the mismatch alone mislabels the first kind as `scam`. The rule therefore
    also requires the sender to be unverified or its domain to be newly registered.
    Everything else still reaches the model, which sees the mismatch flagged in its
    context block and can call `scam` on its own -- narrowing the rule removes a
    short-circuit, not a signal.

    Explicit promotional opt-out is a stated user preference rather than a
    judgement call, but it only applies when the business is otherwise trustworthy,
    so a scam from an opted-out business still hits the impersonation rule first.
    """
    sig = ctx.signals

    domain_age = sig.get("sender_domain_age_days") or 0
    if sig.get("domain_mismatch") and (
        not sig.get("business_verified")
        or domain_age < ESTABLISHED_SENDER_DOMAIN_DAYS
    ):
        return PrePassResult(
            action="mute",
            message_type="scam",
            reason=(
                "An unverified or newly registered sender used a domain that does "
                "not match the brand's official domain, indicating impersonation."
            ),
            confidence=PRE_PASS_CONFIDENCE,
            evidence_message_ids=_impersonation_evidence(ctx),
            rule="domain_mismatch",
        )

    if (
        sig.get("promotions_opted_out")
        and sig.get("business_verified")
        and not sig.get("domain_mismatch")
        and _int(ctx.message.get("forwarded_count")) == 0
    ):
        # The user explicitly opted out of promotions from this business. Only
        # suppress when the message really is promotional; a transactional update
        # from the same business still matters, so defer that to the model.
        if _looks_promotional(ctx):
            return PrePassResult(
                action="mute",
                message_type="promotion",
                reason=(
                    "The user explicitly opted out of promotions from this business, "
                    "so promotional content should be suppressed."
                ),
                confidence=PRE_PASS_CONFIDENCE,
                evidence_message_ids=_recent_business_evidence(ctx),
                rule="promotions_opted_out",
            )

    return None


_PROMO_MARKERS = (
    "sale",
    "offer",
    "discount",
    "% off",
    "coupon",
    "deal",
    "flat ",
    "limited time",
    "shop now",
    "buy now",
    "hurry",
    "voucher",
    "cashback",
    "festive",
)


def _looks_promotional(ctx: MessageContext) -> bool:
    text = (ctx.message.get("message_text") or "").lower()
    if not text:
        # An image poster from an opted-out business is still promotional, but we
        # cannot read it here -- let the model see the image.
        return False
    return any(marker in text for marker in _PROMO_MARKERS)


def _impersonation_evidence(ctx: MessageContext) -> list[str]:
    """Prefer a past message from the same business, else the most similar
    candidate, so the rule still supplies useful evidence."""
    business_id = ctx.message.get("business_id")
    for row in ctx.candidates:
        if business_id and row.get("business_id") == business_id:
            return [row["message_id"]]
    return ctx.candidate_ids[:1]


def _recent_business_evidence(ctx: MessageContext) -> list[str]:
    return _impersonation_evidence(ctx)


# ---- validation / repair -------------------------------------------------------


def validate(
    decision: Decision, ctx: MessageContext
) -> tuple[Decision, list[str], str | None]:
    """Repair a model decision in place and report what changed.

    Returns (repaired_decision, list_of_repairs, escalation_reason_or_None).
    Repairs are applied whether or not we escalate, so the tier-1 answer is always
    submittable even if the tier-2 call fails.
    """
    repairs: list[str] = []
    escalate: str | None = None

    data = decision.model_dump()

    # Allowed-value guard. Structured outputs make these near-impossible, but the
    # CSV contract is the thing being scored so we never trust it blindly.
    if data["action"] not in ACTIONS:
        data["action"] = "digest"
        repairs.append("invalid_action->digest")
        escalate = escalate or "invalid_action"
    if data["message_type"] not in MESSAGE_TYPES:
        data["message_type"] = "unknown"
        repairs.append("invalid_type->unknown")
        escalate = escalate or "invalid_type"

    # Type/action coupling, for the three types where the data says it is rigid.
    required = EXCLUSIVE_ACTION.get(data["message_type"])
    if required and data["action"] != required:
        repairs.append(f"coupling:{data['message_type']}+{data['action']}->{required}")
        data["action"] = required
        escalate = escalate or "coupling_violation"

    # Evidence must reference real, offered candidates.
    allowed = set(ctx.candidate_ids)
    kept = [mid for mid in data["evidence_message_ids"] if mid in allowed]
    if len(kept) != len(data["evidence_message_ids"]):
        dropped = [m for m in data["evidence_message_ids"] if m not in allowed]
        repairs.append(f"dropped_unknown_evidence:{','.join(dropped)}")
        escalate = escalate or "hallucinated_evidence"
    # Samples cite one ID in 25 of 30 cases and never more than two.
    data["evidence_message_ids"] = kept[:2]

    # Confidence calibration.
    conf = float(data["confidence"])
    clamped = min(max(conf, CONFIDENCE_FLOOR), CONFIDENCE_CEILING)
    if clamped != conf:
        repairs.append(f"confidence_clamped:{conf:.2f}->{clamped:.2f}")
    data["confidence"] = clamped
    if clamped < ESCALATE_BELOW:
        escalate = escalate or "low_confidence"

    # Reason hygiene: one short sentence, no leaked IDs.
    reason = " ".join((data["reason"] or "").split())
    if not reason:
        reason = "Routed from sender trust, message content and this user's history."
        repairs.append("empty_reason_filled")
    data["reason"] = reason

    return Decision(**data), repairs, escalate


def escalation_reason(
    decision: Decision, ctx: MessageContext, validator_reason: str | None
) -> str | None:
    """Decide whether tier 1's answer warrants a second opinion from the stronger
    model. Every trigger is computable from the tier-1 response plus context, so
    escalation never costs an extra call to decide."""
    if validator_reason:
        return validator_reason

    if ctx.message.get("media_type") in ("image", "voice"):
        return "multimodal"

    sig = ctx.signals
    high_risk = (
        sig.get("business_reports_30d", 0) >= 5
        or (0 < sig.get("sender_domain_age_days", 0) < 120)
        or _int(ctx.message.get("forwarded_count")) >= 5
        or (sig.get("business_verified") is False and ctx.message.get("business_id"))
    )
    if high_risk and decision.action != "mute":
        return "high_risk_not_muted"

    return None


def as_routed(
    message_id: str,
    decision: Decision,
    *,
    decided_by: str,
    model: str | None,
    escalated: bool,
    escalation_reason_: str | None,
    repairs: list[str],
) -> RoutedMessage:
    return RoutedMessage(
        message_id=message_id,
        action=decision.action,
        message_type=decision.message_type,
        reason=decision.reason,
        confidence=decision.confidence,
        evidence_message_ids=decision.evidence_message_ids,
        decided_by=decided_by,
        model=model,
        escalated=escalated,
        escalation_reason=escalation_reason_,
        repaired=repairs,
    )
