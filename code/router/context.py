"""Build the per-message context block handed to the model.

This is where most of the real work happens. Signals that are cheap and reliable
to compute in Python (quiet-hours overlap, sender-domain mismatch, opt-out state,
group mute state, historical engagement) are computed here and stated plainly,
rather than left for the model to infer from raw table dumps. The model is then
asked to weigh them, which is the part it is actually good at.

Message text is never truncated: the longest message in the dataset is 345
characters, so the whole corpus fits comfortably in context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .loaders import Dataset, Row, _bool, _int

# How many historical messages to offer as evidence candidates. Generous on
# purpose: recall costs a few hundred cached tokens, whereas a gold-relevant
# message that is never offered cannot be cited at all.
MAX_CANDIDATES = 16

_DND_RE = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\s*$")
_TS_FORMATS = ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S")


def parse_timestamp(value: str) -> datetime | None:
    for fmt in _TS_FORMATS:
        try:
            return datetime.strptime(value.strip(), fmt)
        except (ValueError, AttributeError):
            continue
    return None


def in_quiet_hours(window: str, when: datetime | None) -> bool | None:
    """True if `when` falls inside a HH:MM-HH:MM window, handling midnight wrap.

    Returns None when the window or timestamp cannot be parsed, so callers can
    distinguish "outside quiet hours" from "unknown".
    """
    if when is None:
        return None
    match = _DND_RE.match(window or "")
    if not match:
        return None
    sh, sm, eh, em = (int(g) for g in match.groups())
    start, end = sh * 60 + sm, eh * 60 + em
    now = when.hour * 60 + when.minute
    if start <= end:
        return start <= now < end
    return now >= start or now < end  # window wraps past midnight


def minutes_to_quiet_hours(window: str, when: datetime | None) -> int | None:
    """Minutes until quiet hours begin, or None. Lets the model see near-misses
    (a 22:19 message against a 23:00 window) rather than a bare boolean."""
    match = _DND_RE.match(window or "")
    if not match or when is None:
        return None
    sh, sm, _, _ = (int(g) for g in match.groups())
    start = sh * 60 + sm
    now = when.hour * 60 + when.minute
    delta = start - now
    return delta if delta >= 0 else delta + 24 * 60


@dataclass
class MessageContext:
    message: Row
    lines: list[str] = field(default_factory=list)
    candidates: list[Row] = field(default_factory=list)
    signals: dict[str, Any] = field(default_factory=dict)
    image_path: Path | None = None
    voice_path: Path | None = None
    transcript: str | None = None

    @property
    def message_id(self) -> str:
        return self.message["message_id"]

    @property
    def candidate_ids(self) -> list[str]:
        return [c["message_id"] for c in self.candidates]

    def render(self) -> str:
        return "\n".join(self.lines)


class ContextBuilder:
    def __init__(self, ds: Dataset, transcripts: dict[str, str] | None = None) -> None:
        self.ds = ds
        self.transcripts = transcripts or {}

    # ---- history retrieval -----------------------------------------------------

    def _rank_candidates(self, msg: Row) -> list[Row]:
        """Rank the receiving user's own history by how directly it bears on this
        message. Evidence in the solved samples is always drawn from the receiving
        user's history, so the candidate pool never crosses users."""
        user_id = msg["user_id"]
        pool = self.ds.history_by_user.get(user_id, [])
        sender, group, business = (
            msg["sender_user_id"],
            msg["group_id"],
            msg["business_id"],
        )

        scored: list[tuple[int, str, Row]] = []
        for row in pool:
            score = 0
            if sender and row["sender_user_id"] == sender:
                score += 4
            if group and row["group_id"] == group:
                score += 3
            if business and row["business_id"] == business:
                score += 3
            if row["conversation_type"] == msg["conversation_type"]:
                score += 1
            if row["media_type"] and row["media_type"] == msg["media_type"]:
                score += 1
            if score:
                # created_at is ISO-like, so lexical sort is chronological.
                scored.append((score, row["created_at"], row))

        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        ranked = [row for _, _, row in scored[:MAX_CANDIDATES]]

        # Top up with the user's most recent history regardless of score. Related
        # messages are ranked first, but a strict score>0 filter silently starves
        # the model: some messages share no sender, group, business or conversation
        # type with anything in the user's history, and were offered as few as four
        # candidates. Padding costs a few cached tokens and cannot hurt precision --
        # the model is still free to return no evidence at all.
        if len(ranked) < MAX_CANDIDATES:
            seen = {row["message_id"] for row in ranked}
            for row in pool:
                if len(ranked) >= MAX_CANDIDATES:
                    break
                if row["message_id"] not in seen:
                    ranked.append(row)
                    seen.add(row["message_id"])
        return ranked

    def _render_candidate(self, row: Row, user_id: str) -> str:
        origin = (
            f"from user {row['sender_user_id']}"
            if row["sender_user_id"]
            else f"from business {row['business_id']}"
            if row["business_id"]
            else "unknown sender"
        )
        if row["group_id"]:
            origin += f" in group {row['group_id']}"

        parts = [
            f"  - {row['message_id']} [{row['created_at']}] {row['conversation_type']}, {origin}"
        ]
        if row["media_type"]:
            parts.append(f"    media: {row['media_type']}")
        if row["message_text"]:
            parts.append(f"    text: {row['message_text']}")
        if _int(row["forwarded_count"]) > 0:
            parts.append(f"    forwarded {row['forwarded_count']} times")

        ev = self.ds.event(user_id, row["message_id"])
        if ev:
            reactions = []
            reactions.append("opened" if _bool(ev["message_opened"]) else "not opened")
            if _bool(ev["message_replied"]):
                reactions.append("replied")
            if _bool(ev["notification_dismissed"]):
                reactions.append("dismissed the notification")
            if _bool(ev["muted_after_message"]):
                reactions.append("MUTED the conversation afterwards")
            if _bool(ev["message_reported"]):
                reactions.append("REPORTED it")
            rt = ev.get("reaction_time_minutes", "")
            if rt not in ("", None) and _bool(ev["message_opened"]):
                reactions.append(f"reacted after {rt} min")
            parts.append(f"    this user: {', '.join(reactions)}")
        else:
            parts.append("    this user: no recorded reaction")
        return "\n".join(parts)

    # ---- signal extraction -----------------------------------------------------

    def _user_block(self, msg: Row, ctx: MessageContext) -> list[str]:
        user = self.ds.users.get(msg["user_id"], {})
        sent_at = parse_timestamp(msg["created_at"])
        window = user.get("do_not_disturb_window", "")
        quiet = in_quiet_hours(window, sent_at)
        until = minutes_to_quiet_hours(window, sent_at)

        ctx.signals["in_quiet_hours"] = quiet
        ctx.signals["minutes_to_quiet_hours"] = until

        opened = _int(user.get("messages_opened_30d"))
        dismissed = _int(user.get("notifications_dismissed_30d"))
        reported = _int(user.get("messages_reported_30d"))
        ctx.signals["user_reported_30d"] = reported

        load = self.ds.notification_load(msg["user_id"])
        lines = [
            "RECEIVING USER",
            f"  user_id: {msg['user_id']}",
            f"  quiet hours: {window or 'not set'}"
            + (
                f" -- this message arrives DURING quiet hours"
                if quiet
                else f" -- arrives {until} min before quiet hours start"
                if until is not None and until <= 90
                else " -- arrives outside quiet hours"
                if quiet is False
                else ""
            ),
            f"  last 30d: opened {opened}, replied {_int(user.get('messages_replied_30d'))}, "
            f"dismissed {dismissed}, reported {reported}",
            f"  notification load: {load['sent_per_day']}/day over {load['days']}d, "
            f"dismissal rate {load['dismissal_rate']}",
        ]
        return lines

    def _group_block(self, msg: Row, ctx: MessageContext) -> list[str]:
        group = self.ds.groups.get(msg["group_id"])
        if not group:
            return []
        mine = self.ds.membership(msg["group_id"], msg["user_id"])
        sender = (
            self.ds.membership(msg["group_id"], msg["sender_user_id"])
            if msg["sender_user_id"]
            else None
        )

        muted = _bool(mine.get("group_muted_by_user")) if mine else False
        sender_role = sender.get("role") if sender else "unknown"
        ctx.signals["group_muted_by_user"] = muted
        ctx.signals["sender_is_admin"] = sender_role == "admin"

        lines = [
            "GROUP",
            f"  {group['group_name']} ({group['group_type']}), {group['member_count']} members, "
            f"{group['messages_30d']} messages in 30d",
            f"  sender {msg['sender_user_id']} role in this group: {sender_role}",
        ]
        if mine:
            lines.append(
                f"  this user in this group: role {mine['role']}, "
                f"read {mine['messages_read_30d']}, sent {mine['messages_sent_30d']}, "
                f"replied {mine['replies_sent_30d']}, "
                f"dismissed {mine['notifications_dismissed_30d']} notifications in 30d"
            )
            lines.append(
                "  this user has MUTED this group"
                if muted
                else "  this group is not muted by this user"
            )
        else:
            lines.append("  this user has no membership record for this group")
        return lines

    def _business_block(self, msg: Row, ctx: MessageContext) -> list[str]:
        biz = self.ds.businesses.get(msg["business_id"])
        if not biz:
            return []
        verified = _bool(biz.get("verified"))
        official = (biz.get("official_domain") or "").strip().lower()
        used = (biz.get("domain_used_by_sender") or "").strip().lower()
        mismatch = bool(official and used and official != used)
        reports = _int(biz.get("user_reports_30d"))
        domain_age = _int(biz.get("domain_used_by_sender_age_days"))

        ctx.signals["domain_mismatch"] = mismatch
        ctx.signals["business_verified"] = verified
        ctx.signals["business_reports_30d"] = reports
        ctx.signals["sender_domain_age_days"] = domain_age

        lines = [
            "BUSINESS SENDER",
            f"  {biz['display_name']} (brand: {biz['brand_name']}, category: {biz['category']})",
            f"  verified: {'yes' if verified else 'NO'}",
            f"  official domain: {official or 'unknown'}",
            f"  domain actually used by this sender: {used or 'unknown'}"
            + ("   <-- MISMATCH, impersonation risk" if mismatch else ""),
            f"  sender domain age: {domain_age} days"
            + ("   <-- very new domain" if 0 < domain_age < 120 else ""),
            f"  account age: {biz['account_age_days']} days, "
            f"{biz['messages_sent_30d']} messages sent in 30d, {reports} user reports in 30d",
        ]

        rel = self.ds.relationship(msg["user_id"], msg["business_id"])
        if rel:
            allows = _bool(rel.get("allows_promotions"))
            opted_out = (rel.get("promotions_opted_out_at") or "").strip()
            ctx.signals["allows_promotions"] = allows
            ctx.signals["promotions_opted_out"] = bool(opted_out)
            lines += [
                "  RELATIONSHIP WITH THIS USER",
                f"    why the user knows this account: {rel['why_user_knows_account']}",
                f"    last activity: {rel['last_activity_at'] or 'none'}, "
                f"{rel['activity_count_180d']} activities in 180d",
                f"    promotions allowed: {'yes' if allows else 'NO'}"
                + (f" (opted out at {opted_out})" if opted_out else ""),
                f"    last 30d with this business: opened {rel['messages_opened_30d']}, "
                f"dismissed {rel['messages_dismissed_30d']}, replied {rel['messages_replied_30d']}",
            ]
        else:
            ctx.signals["allows_promotions"] = None
            ctx.signals["promotions_opted_out"] = False
            lines.append("  this user has NO prior relationship with this business")
        return lines

    def _message_block(self, msg: Row, ctx: MessageContext) -> list[str]:
        forwarded = _int(msg["forwarded_count"])
        ctx.signals["forwarded_count"] = forwarded

        lines = [
            "INCOMING MESSAGE",
            f"  message_id: {msg['message_id']}",
            f"  received: {msg['created_at']}",
            f"  conversation type: {msg['conversation_type']}",
        ]
        if msg["sender_user_id"]:
            lines.append(f"  sender: user {msg['sender_user_id']}")
        if msg["business_id"]:
            lines.append(f"  sender: business {msg['business_id']}")
        if forwarded > 0:
            lines.append(
                f"  forwarded {forwarded} times"
                + ("   <-- heavily forwarded chain content" if forwarded >= 5 else "")
            )

        if msg["message_text"]:
            lines.append(f"  text: {msg['message_text']}")

        media_id = msg["media_id"]
        if msg["media_type"] == "image" and media_id:
            ctx.image_path = self.ds.media_path(media_id)
            lines.append(
                "  media: an image is attached to this prompt -- read any text in it "
                "(poster, screenshot, receipt) and judge the image content itself."
            )
        elif msg["media_type"] == "voice" and media_id:
            ctx.voice_path = self.ds.media_path(media_id)
            transcript = self.transcripts.get(media_id)
            ctx.transcript = transcript
            if transcript:
                lines.append(f"  media: voice note. Transcript: {transcript}")
            else:
                lines.append(
                    "  media: voice note, but no transcript is available. "
                    "Decide from sender, conversation and history alone, and lower confidence."
                )
        return lines

    # ---- entry point -----------------------------------------------------------

    def build(self, msg: Row) -> MessageContext:
        ctx = MessageContext(message=msg)
        blocks: list[list[str]] = [
            self._message_block(msg, ctx),
            self._user_block(msg, ctx),
            self._group_block(msg, ctx),
            self._business_block(msg, ctx),
        ]

        ctx.candidates = self._rank_candidates(msg)
        if ctx.candidates:
            cand_lines = [
                "CANDIDATE HISTORY (the only IDs you may cite as evidence)",
            ]
            for row in ctx.candidates:
                cand_lines.append(self._render_candidate(row, msg["user_id"]))
            blocks.append(cand_lines)
        else:
            blocks.append(
                ["CANDIDATE HISTORY", "  none available -- return an empty evidence list"]
            )

        ctx.lines = []
        for block in blocks:
            if block:
                ctx.lines.extend(block)
                ctx.lines.append("")
        return ctx
