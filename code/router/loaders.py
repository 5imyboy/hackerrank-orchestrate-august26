"""Load every participant-facing CSV in `dataset/` into indexed lookups.

Everything here is pure I/O plus indexing: no derived judgement, no API calls.
Indexes are built once and shared across all messages, so the per-message context
build is O(1) lookups rather than repeated table scans.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

Row = dict[str, str]


def _read(path: Path) -> list[Row]:
    if not path.exists():
        raise FileNotFoundError(f"required dataset file missing: {path}")
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _int(value: str | None, default: int = 0) -> int:
    """CSV integers arrive as strings and may be blank."""
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def _bool(value: str | None) -> bool:
    return _int(value) == 1


@dataclass
class Dataset:
    """All participant-facing data, indexed for per-message lookup."""

    root: Path

    messages: list[Row] = field(default_factory=list)
    samples: list[Row] = field(default_factory=list)

    users: dict[str, Row] = field(default_factory=dict)
    groups: dict[str, Row] = field(default_factory=dict)
    businesses: dict[str, Row] = field(default_factory=dict)

    # (group_id, user_id) -> membership row
    memberships: dict[tuple[str, str], Row] = field(default_factory=dict)
    # (user_id, business_id) -> relationship row
    business_history: dict[tuple[str, str], Row] = field(default_factory=dict)

    history: dict[str, Row] = field(default_factory=dict)
    history_by_user: dict[str, list[Row]] = field(default_factory=dict)
    # (user_id, message_id) -> event row
    events: dict[tuple[str, str], Row] = field(default_factory=dict)

    media_paths: dict[str, Path] = field(default_factory=dict)
    # user_id -> list of daily notification rows, ascending by date
    daily_load: dict[str, list[Row]] = field(default_factory=dict)

    @classmethod
    def load(cls, root: str | Path) -> "Dataset":
        root = Path(root)
        ds = cls(root=root)

        ds.messages = _read(root / "messages.csv")
        ds.samples = _read(root / "sample_messages.csv")

        ds.users = {r["user_id"]: r for r in _read(root / "users.csv")}
        ds.groups = {r["group_id"]: r for r in _read(root / "groups.csv")}
        ds.businesses = {
            r["business_id"]: r for r in _read(root / "business_accounts.csv")
        }

        ds.memberships = {
            (r["group_id"], r["user_id"]): r for r in _read(root / "group_members.csv")
        }
        ds.business_history = {
            (r["user_id"], r["business_id"]): r
            for r in _read(root / "user_business_history.csv")
        }

        history = _read(root / "message_history.csv")
        ds.history = {r["message_id"]: r for r in history}
        by_user: dict[str, list[Row]] = defaultdict(list)
        for row in history:
            by_user[row["user_id"]].append(row)
        # Most recent first -- retrieval and prompt rendering both want recency order.
        for rows in by_user.values():
            rows.sort(key=lambda r: r["created_at"], reverse=True)
        ds.history_by_user = dict(by_user)

        ds.events = {
            (r["user_id"], r["message_id"]): r for r in _read(root / "message_events.csv")
        }

        for row in _read(root / "images.csv"):
            ds.media_paths[row["image_id"]] = root / row["file_path"]
        for row in _read(root / "voice_notes.csv"):
            ds.media_paths[row["voice_note_id"]] = root / row["file_path"]

        load: dict[str, list[Row]] = defaultdict(list)
        for row in _read(root / "daily_notification_summary.csv"):
            load[row["user_id"]].append(row)
        for rows in load.values():
            rows.sort(key=lambda r: r["date"])
        ds.daily_load = dict(load)

        return ds

    # ---- convenience accessors -------------------------------------------------

    def membership(self, group_id: str, user_id: str) -> Row | None:
        return self.memberships.get((group_id, user_id))

    def relationship(self, user_id: str, business_id: str) -> Row | None:
        return self.business_history.get((user_id, business_id))

    def event(self, user_id: str, message_id: str) -> Row | None:
        return self.events.get((user_id, message_id))

    def media_path(self, media_id: str) -> Path | None:
        return self.media_paths.get(media_id)

    def notification_load(self, user_id: str, days: int = 14) -> dict[str, Any]:
        """Recent notification volume and dismissal rate -- a fatigue signal."""
        rows = self.daily_load.get(user_id, [])[-days:]
        if not rows:
            return {"days": 0, "sent_per_day": 0.0, "dismissal_rate": 0.0}
        sent = sum(_int(r["notifications_sent"]) for r in rows)
        dismissed = sum(_int(r["notifications_dismissed"]) for r in rows)
        return {
            "days": len(rows),
            "sent_per_day": round(sent / len(rows), 2),
            "dismissal_rate": round(dismissed / sent, 2) if sent else 0.0,
        }
