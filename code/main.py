#!/usr/bin/env python3
"""Message Notification Router -- entry point.

    python code/main.py                 # route dataset/messages.csv -> output.csv
    python code/main.py --eval          # score against the solved sample rows
    python code/main.py --help          # all options

Reads ANTHROPIC_API_KEY from the environment (or code/.env). No secrets are read
from, or written to, any file in this repository.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import sys
from collections import Counter
from pathlib import Path

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))

from router.classify import TIER1, TIER2, Classifier, Tier  # noqa: E402
from router.context import ContextBuilder  # noqa: E402
from router.loaders import Dataset, Row  # noqa: E402
from router.prompt import build_examples, build_system_prompt  # noqa: E402
from router.runner import DecisionCache, Runner  # noqa: E402
from router.schema import OUTPUT_COLUMNS, RoutedMessage  # noqa: E402
from router.transcribe import ensure_transcripts  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
CODE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WhatsApp message notification router")
    p.add_argument("--dataset", default=str(REPO / "dataset"), help="dataset directory")
    p.add_argument("--out", default=str(REPO / "output.csv"), help="output CSV path")
    p.add_argument("--cache-dir", default=str(CODE / "cache"))
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="route only the first N messages")
    p.add_argument("--tier1-model", default=TIER1.model)
    p.add_argument("--tier2-model", default=TIER2.model)
    p.add_argument(
        "--no-cascade",
        action="store_true",
        help="never escalate to the tier-2 model (cheaper, lower accuracy)",
    )
    p.add_argument(
        "--no-rules",
        action="store_true",
        help="disable the deterministic pre-pass and route everything through the model",
    )
    p.add_argument("--asr-model", default="small", help="faster-whisper model size")
    p.add_argument("--force-asr", action="store_true", help="re-transcribe all voice notes")
    p.add_argument(
        "--eval",
        action="store_true",
        help="route the solved sample rows and score against their labels",
    )
    return p.parse_args()


def count_system_tokens(system_prompt: str, model: str) -> int | None:
    """Exact token count for the cached prefix, via the token-counting endpoint."""
    try:
        from anthropic import Anthropic

        result = Anthropic().messages.count_tokens(
            model=model,
            system=[{"type": "text", "text": system_prompt}],
            messages=[{"role": "user", "content": "x"}],
        )
        return result.input_tokens
    except Exception as exc:  # noqa: BLE001 -- diagnostics only, never fatal
        print(f"[setup] token count unavailable: {exc!r}", file=sys.stderr)
        return None


def build_runner(args: argparse.Namespace, ds: Dataset) -> tuple[Runner, str]:
    load_dotenv(CODE / ".env")
    load_dotenv(REPO / ".env")
    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
        sys.exit(
            "ANTHROPIC_API_KEY is not set. Export it, or put it in code/.env "
            "(see code/.env.example)."
        )

    store = ensure_transcripts(
        ds,
        Path(args.cache_dir) / "transcripts.json",
        model_size=args.asr_model,
        force=args.force_asr,
    )
    builder = ContextBuilder(ds, transcripts=store.as_mapping())
    system_prompt = build_system_prompt(ds)

    # Measure the prompt against the tier-1 minimum cacheable prefix for real,
    # rather than estimating: 4096 tokens is a hard cliff on Haiku, and a prompt
    # that lands just under it caches nothing while reporting no error.
    tokens = count_system_tokens(system_prompt, args.tier1_model)
    if tokens is None:
        print("[setup] could not count system prompt tokens", file=sys.stderr)
    else:
        print(
            f"[setup] system prompt {tokens} tokens -- "
            f"{'above' if tokens >= 4096 else 'BELOW'} the 4096-token minimum "
            f"cacheable prefix for {args.tier1_model}",
            file=sys.stderr,
        )

    client = AsyncAnthropic(max_retries=5)
    classifier = Classifier(client, system_prompt)

    tier1 = Tier(**{**TIER1.__dict__, "model": args.tier1_model})
    tier2 = Tier(**{**TIER2.__dict__, "model": args.tier2_model})

    runner = Runner(
        builder,
        classifier,
        system_prompt,
        DecisionCache(Path(args.cache_dir) / "decisions.json"),
        tier1=tier1,
        tier2=tier2,
        cascade=not args.no_cascade,
        rules=not args.no_rules,
        concurrency=args.concurrency,
    )
    return runner, system_prompt


def write_output(path: Path, order: list[str], routed: list[RoutedMessage]) -> None:
    """Write predictions in the same row order as the template, one row per input."""
    by_id = {r.message_id: r for r in routed}
    missing = [mid for mid in order if mid not in by_id]
    if missing:
        raise SystemExit(f"internal error: no prediction for {len(missing)} messages")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(OUTPUT_COLUMNS))
        writer.writeheader()
        for mid in order:
            writer.writerow(by_id[mid].to_csv_row())
    print(f"[done] wrote {len(order)} predictions to {path}")


def summarise(routed: list[RoutedMessage]) -> None:
    actions = Counter(r.action for r in routed)
    types = Counter(r.message_type for r in routed)
    no_evidence = sum(1 for r in routed if not r.evidence_message_ids)
    print("\ndistribution")
    print("  actions:", dict(actions.most_common()))
    print("  types:  ", dict(types.most_common()))
    print(f"  rows with no evidence: {no_evidence}/{len(routed)}")


def score(ds: Dataset, routed: list[RoutedMessage]) -> None:
    """Score against the solved sample labels.

    The few-shot block is built from one example per (action, type) pair, so those
    rows are reported separately -- they are in the prompt and scoring them would
    flatter the result.
    """
    gold = {r["message_id"]: r for r in ds.samples}
    shown = set()
    by_pair: dict[tuple[str, str], str] = {}
    for row in sorted(ds.samples, key=lambda r: r["message_id"]):
        by_pair.setdefault((row["action"], row["message_type"]), row["message_id"])
    shown = set(by_pair.values())

    held = [r for r in routed if r.message_id not in shown]
    action_hits = sum(1 for r in held if r.action == gold[r.message_id]["action"])
    type_hits = sum(
        1 for r in held if r.message_type == gold[r.message_id]["message_type"]
    )
    both = sum(
        1
        for r in held
        if r.action == gold[r.message_id]["action"]
        and r.message_type == gold[r.message_id]["message_type"]
    )
    evidence_hits = 0
    evidence_total = 0
    for r in held:
        want = gold[r.message_id]["evidence_message_ids"]
        if want == "none":
            continue
        evidence_total += 1
        if set(want.split(";")) & set(r.evidence_message_ids):
            evidence_hits += 1

    n = len(held) or 1
    print("\n" + "=" * 62)
    print(f"EVAL on {len(held)} held-out sample rows ({len(shown)} used as few-shots)")
    print(f"  action accuracy:      {action_hits}/{len(held)} ({action_hits / n:.1%})")
    print(f"  message_type accuracy:{type_hits}/{len(held)} ({type_hits / n:.1%})")
    print(f"  both correct:         {both}/{len(held)} ({both / n:.1%})")
    if evidence_total:
        print(
            f"  evidence overlap:     {evidence_hits}/{evidence_total} "
            f"({evidence_hits / evidence_total:.1%})"
        )

    confusion = Counter(
        (gold[r.message_id]["action"], r.action)
        for r in held
        if r.action != gold[r.message_id]["action"]
    )
    if confusion:
        print("  action errors (gold -> predicted):")
        for (want, got), count in confusion.most_common():
            print(f"    {want} -> {got}: {count}")

    type_conf = Counter(
        (gold[r.message_id]["message_type"], r.message_type)
        for r in held
        if r.message_type != gold[r.message_id]["message_type"]
    )
    if type_conf:
        print("  type errors (gold -> predicted):")
        for (want, got), count in type_conf.most_common():
            print(f"    {want} -> {got}: {count}")
    print("=" * 62)


async def main() -> None:
    args = parse_args()
    ds = Dataset.load(args.dataset)
    runner, _ = build_runner(args, ds)

    if args.eval:
        rows: list[Row] = list(ds.samples)
    else:
        rows = list(ds.messages)
    if args.limit:
        rows = rows[: args.limit]

    routed = await runner.route_all(rows)
    print(runner.stats.report())
    summarise(routed)

    if args.eval:
        score(ds, routed)
    else:
        write_output(Path(args.out), [r["message_id"] for r in rows], routed)


if __name__ == "__main__":
    asyncio.run(main())
