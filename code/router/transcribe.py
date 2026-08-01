"""Voice-note transcription.

The Claude API accepts text, images and PDFs -- not audio -- so voice notes need
a separate speech-to-text step before they can be routed on content. We use
faster-whisper locally: no extra API key, no second vendor, and it ships its own
audio decoder so no system ffmpeg is required.

Transcripts are cached to disk and committed with the code. That makes the run
reproducible, keeps re-runs free, and means a reviewer never has to run ASR at
all -- the pipeline reads the cache and only transcribes what is missing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .loaders import Dataset

DEFAULT_MODEL = "small"


class TranscriptStore:
    """A JSON-backed cache of media_id -> transcript."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.entries: dict[str, dict[str, str]] = {}
        if path.exists():
            self.entries = json.loads(path.read_text(encoding="utf-8"))

    def text(self, media_id: str) -> str | None:
        entry = self.entries.get(media_id)
        if not entry:
            return None
        text = (entry.get("text") or "").strip()
        return text or None

    def as_mapping(self) -> dict[str, str]:
        """media_id -> transcript, annotated with the detected language when it is
        not English so the classifier knows it is reading a translation-free
        transcript of another language."""
        out: dict[str, str] = {}
        for media_id, entry in self.entries.items():
            text = (entry.get("text") or "").strip()
            if not text:
                continue
            lang = entry.get("language", "")
            if lang and lang != "en":
                text = f"[spoken language: {lang}] {text}"
            out[media_id] = text
        return out

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.entries, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )


def voice_media_ids(ds: Dataset) -> list[str]:
    """Every voice note referenced anywhere we might route or cite.

    Includes `sample_messages.csv`: those rows are routed by `--eval`, and leaving
    them out silently routes evaluation voice notes with no transcript, which
    reads as a model error rather than a missing input.
    """
    ids: list[str] = []
    for row in list(ds.messages) + list(ds.history.values()) + list(ds.samples):
        if row.get("media_type") == "voice" and row.get("media_id"):
            ids.append(row["media_id"])
    return sorted(set(ids))


def ensure_transcripts(
    ds: Dataset,
    cache_path: Path,
    model_size: str = DEFAULT_MODEL,
    force: bool = False,
) -> TranscriptStore:
    """Fill any missing transcripts, then return the store.

    If faster-whisper is unavailable, this degrades to whatever is already cached
    and the pipeline continues -- voice notes are then routed on metadata alone and
    the prompt says so explicitly.
    """
    store = TranscriptStore(cache_path)
    wanted = voice_media_ids(ds)
    missing = [m for m in wanted if force or m not in store.entries]

    if not missing:
        print(f"[asr] {len(wanted)} voice notes, all cached", file=sys.stderr)
        return store

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print(
            f"[asr] faster-whisper not installed; {len(missing)} voice notes will be "
            "routed without a transcript. Install it from requirements.txt to fix.",
            file=sys.stderr,
        )
        return store

    print(
        f"[asr] transcribing {len(missing)} voice notes with whisper '{model_size}' "
        "(one-off; results are cached)",
        file=sys.stderr,
    )
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    for media_id in missing:
        path = ds.media_path(media_id)
        if path is None or not path.exists():
            print(f"[asr]   {media_id}: file missing, skipped", file=sys.stderr)
            continue
        # Auto-detect the language rather than forcing English: this dataset is
        # plausibly code-switched, and Claude reads the source language fine.
        segments, info = model.transcribe(str(path), beam_size=5, vad_filter=True)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        store.entries[media_id] = {
            "text": text,
            "language": info.language,
            "duration": round(info.duration, 1),
        }
        print(
            f"[asr]   {media_id}: {info.language} {info.duration:.0f}s -> {len(text)} chars",
            file=sys.stderr,
        )

    store.save()
    return store
