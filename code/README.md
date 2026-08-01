# Message Notification Router

Routes every message in `dataset/messages.csv` to `notify`, `digest` or `mute`,
personalised to the receiving user, and writes `output.csv`.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r code/requirements.txt
cp code/.env.example code/.env      # then add your key
```

`ANTHROPIC_API_KEY` is read from the environment or `code/.env`. No secret is
read from or written to any other file.

## Run

```bash
.venv/bin/python code/main.py                 # route messages.csv -> output.csv
.venv/bin/python code/main.py --eval          # score against the solved samples
.venv/bin/python code/main.py --help          # all options
```

A full run is roughly 60 seconds and about $1. Re-runs are free and produce a
byte-identical `output.csv`, because every model response is cached on disk.

Useful flags: `--no-cascade` (tier 1 only, cheaper), `--no-rules` (route
everything through the model), `--tier1-model` / `--tier2-model`, `--concurrency`,
`--limit N`.

## How it works

```
CSVs ──┐
       ├─→ ContextBuilder ─→ deterministic pre-pass ─→ decided? emit
media ─┘         │                    │
   images: attached to the prompt      └─→ tier 1: claude-haiku-4-5
   audio:  Whisper transcript, cached          │
                                               ├─→ validator (repair + escalate?)
                                               │        │
                                               │        └─→ tier 2: claude-opus-5
                                               └────────────────→ output.csv
```

**Context building** (`router/context.py`) does most of the work. Signals that are
cheap and reliable in Python are computed and stated plainly rather than left for
the model to infer from raw tables: whether the message lands inside the user's
do-not-disturb window (and how near the boundary it is), whether the sender is a
group admin, whether the user has muted the group, whether the sender's domain
matches the brand's official domain, whether the user opted out of promotions,
and the user's notification-fatigue rate.

**Evidence retrieval** ranks the receiving user's own history by shared sender,
group, business and conversation type, then tops the list up to 16 with recent
history. The model may only cite IDs from that list, and the validator drops
anything outside it — so `evidence_message_ids` cannot be hallucinated.

**One API call per message.** The decision depends on sender trust, content,
timing and behavioural history at once; splitting it would discard exactly the
cross-signal reasoning that makes it personalised.

**The cascade varies only the model** — same system prompt, same JSON schema, same
context on both tiers. Escalation is config, not a second prompt to maintain.
Tier 1 is `claude-haiku-4-5`; escalation to `claude-opus-5` fires on low
confidence, a type/action coupling violation, evidence outside the candidate list,
any image or voice message, or an unmuted high-risk sender. Every trigger is
computable from the tier-1 response, so escalation never costs a call to decide.

**Structured outputs** constrain the response to a JSON schema, so allowed values
cannot drift from the submission contract.

## Design decisions worth knowing

**The Claude API does not accept audio.** Voice notes carry no text at all, so
they need speech-to-text before they can be routed on content. `faster-whisper`
runs locally: no second vendor, no second key, and it bundles its own decoder so
no system `ffmpeg` is needed. Transcripts are cached in `code/cache/` and shipped
with the code, so a reviewer never has to run ASR — the pipeline reads the cache
and only transcribes what is missing.

**A bigger system prompt is cheaper here, not more expensive.** Prompt caching has
a minimum cacheable prefix of 4096 tokens on Claude Haiku 4.5. A lean prompt would
sit under that line and cache nothing; the current 4,779-token rubric caches and
then bills at roughly a tenth of list price. Measured on the eval run, 78% of
input tokens are served from cache. `main.py` prints the measured token count
against that threshold on every run, because landing just under it fails silently.

**Determinism.** The current models reject `temperature` outright, so
reproducibility comes from caching instead: every response is keyed by a hash of
the exact request (model, thinking config, system prompt, rendered context, image
bytes). Re-running produces identical output without calling the API. Changing the
prompt or the context changes the hash and correctly forces a fresh run.

**Ordering by user.** Messages are processed grouped by `user_id` so each user's
profile and history prefix stays warm in the prompt cache across their own
messages. Marginal on 110 rows; it is the dominant saving at production volume.

**Images are PNG and WebP despite `.jpg` filenames.** 9 of 20 image files are
mislabelled, and the API rejects a declared media type that disagrees with the
bytes. The media type is sniffed from magic bytes.

## Results on the solved samples

`--eval` routes `sample_messages.csv` and scores it. The few-shot block uses one
example per observed (action, type) pair, so those 15 rows are excluded from
scoring and the result is measured on the 15 held out.

| metric | result |
|---|---|
| action accuracy | 14/15 |
| message_type accuracy | 14/15 |
| both correct | 14/15 |
| evidence overlap | 7/14 |

Two caveats, stated rather than buried. First, 15 rows is a small set and there is
run-to-run variance of roughly one row; treat these as indicative, not precise.

Second, **evidence overlap has a semantic ceiling well below 100%**, because the
gold evidence in `sample_messages.csv` is largely a generation artifact rather
than a semantic relationship: `sample_msg_001` pairs with `message_0001`,
`002` with `0002`, and so on, drifting to a constant `+5` offset. In one case
(`sample_msg_042`, a group voice note) the gold evidence is an unrelated business
health notice. That offset was deliberately **not** exploited — reverse-engineering
an ID pattern to inflate a score is the "hardcoded labels" the brief prohibits, and
it generalises to nothing. Where the model disagrees with gold it usually cites a
defensible alternative: for `sample_msg_005` it cited a *byte-identical* message
from the same business that gold did not pick.

## Scaling beyond the submission

The brief is 110 messages, but the architecture assumes thousands per user over
time. Cost per message is driven down by four levers, all live in the code:

1. **Prompt caching** — 78% of input tokens served from cache on the eval run.
2. **The deterministic pre-pass** — domain-mismatch impersonation and explicit
   promotional opt-outs are decided with no API call at all.
3. **The cascade** — the strong model is spent only on the ~1/3 of messages that
   need it, rather than on everything.
4. **Ordering by user** — keeps each user's context prefix warm.

The Batch API halves token cost again for backfills and is the natural next step
for non-latency-sensitive volume; the live concurrent path is kept as the default
because routing must not delay message delivery.

The measured escalation rate and blended per-message cost are printed at the end
of every run, so the cost claim is reported from the run rather than asserted.

## Layout

```
code/
  main.py              CLI: routing, evaluation, reporting
  requirements.txt
  .env.example
  router/
    loaders.py         CSV loading and indexing
    context.py         derived signals + evidence-candidate retrieval
    prompt.py          system prompt: rubric + worked examples
    classify.py        the single Claude call, schema, cascade tiers, costing
    rules.py           deterministic pre-pass + validator/repair
    runner.py          async driver, disk cache, escalation, run stats
    transcribe.py      voice-note ASR with a committed transcript cache
    schema.py          output contract
  cache/
    transcripts.json   committed: voice transcripts
    decisions.json     committed: model responses, for reproducible re-runs
```
