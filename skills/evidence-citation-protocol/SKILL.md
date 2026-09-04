---
name: evidence-citation-protocol
category: analysis
description: How to cite numbers so the citation gate accepts them — cell_id syntax, the V1-V4 checks, visible vs invisible evidence, NaN as evidence, and why approximate matching was removed.
---

# Evidence & Citation Protocol (M-RS2)

## Operating core

**Use when** — before writing any number into an answer.
**Settles** — how to cite so the gate accepts your draft.

**Procedure**
1. Put every number inside `[[e:run:call:jsonpath]]`. Bare numbers are rejected except years and values flagged `~approx`.
2. Cite only cells you actually saw (`visible=True`).
3. Name the metric exactly as `cell.metric` — the gate compares your wording against it.
4. If a digest says `truncated: true`, either read the artifact or say your conclusion covers only the visible portion.
5. Report NaN as NaN with its cell. It is evidence of absence, not absence of evidence.

**Never**
- Recall a number from model memory, or round one to "close enough" — there is no tolerance matching.
- Invent a `cell_id`; the id space contains the provider's `call_id` and will not resolve.
- Emit a number from a fallback path and label it verified.


**You do not write numbers. You write citations.** The renderer substitutes the value together with the
metric name. This is HARD RULE #2 of the system prompt, and it is enforced by a deterministic gate — not
by good intentions.

## 1. The citation form

```text
[[e:{run_id}:{call_id}:{jsonpath}]]        e.g.  [[e:run7:call3:data.ic]]
```

Write that token where the number belongs. The renderer expands it to the value plus the metric name.

The id space is **not guessable** — it contains the provider's `call_id`. An invented `cell_id` fails to
resolve and the answer is REJECTED. There is no partial credit for a plausible-looking id.

## 2. `EvidenceCell` — the record behind a citation

```python
class EvidenceCell(BaseModel):
    cell_id: str            # "e:{run_id}:{call_id}:{jsonpath}" — stable, citable
    call_id: str
    tool: str
    jsonpath: str           # "data.ic" — kept verbatim
    metric: str             # CANONICAL metric name from the feature registry, NOT the jsonpath
    value: float | None
    status: Literal["observed", "nan", "unavailable", "redacted"]
    unit: str | None
    universe_id: str        # stops a number from one universe being cited for another
    registry_version: str
    n_obs: int | None
    visible: bool
    source_paths: list[str]
    byte_sha256: str
    content_fingerprint: str
```

Four properties that carry weight:

- **`status="nan"` instead of dropping the row.** Non-finite values used to be filtered out, so NaN
  vanished from the ledger. A NaN metric is **evidence of absence**, not absence of evidence — report it, cite it.
- **`metric` is separate from `jsonpath`.** Sharing one field for both is how a cell gets described as
  the wrong quantity.
- **`universe_id` + `registry_version`** stamp every cell, replacing an identity state machine that could
  never work here.
- **`visible`** — see §4.

## 3. The gate: V1–V4, deterministic and language-independent

| Code | Check |
| :-- | :-- |
| **V1** | every digit in the output lies inside a substituted citation span |
| **V2** | every `cell_id` resolves **and** has `visible=True` |
| **V3** | the metric name used in the sentence matches `cell.metric` |
| **V4** | a bare number outside a span ⇒ **REJECT** (fail-closed), except for the declared whitelist |

**V4 whitelist, and nothing else:** years (`19xx` / `20xx`), iteration counts, and values explicitly
tagged `~approx`.

Everything that used to break this is gone at the root rather than patched: Vietnamese decimal commas,
`%` suffixes, `|` table characters, "based on" incantations, English-vs-Vietnamese vocabulary, segment
splitting that needed an ASCII period. None of those regexes exist any more.

Two consequences you should not fight:

- **A citation-only answer that says almost nothing is the CORRECT output when there is no evidence.**
  It is also observable: `citation_density = 0` with a pending `AnomalySignal` raises an operational
  alert. Silence is not how you fail here.
- Citing the right cell but describing it wrongly (`[[…zscore]]` labelled as volatility) is caught by V3, because
  the renderer injects the metric name and the gate compares it with your sentence.

## 4. Visible vs invisible evidence — do not mix the two

| | used for | rule |
| :-- | :-- | :-- |
| `visible=True` | `validate_final_answer` | you are accountable **only** for what you were shown |
| `visible=False` | audit + the `incomplete_analysis` gate | **never** used to catch fabricated numbers |

Mixing them both punishes truth and hides lies: you get flagged for a conflict with a value at byte 30,000
that you could not read, while a fabricated number gets a better chance of landing near one of 2,000
invisible values.

**`incomplete_analysis` is a separate error code**: a conclusion drawn while the unread region contains
values outside the reported range. That is a *process* error, not fabrication. They need different fixes,
so they have different codes.

## 5. Tool results are digested, never truncated

Raw tool output is redacted **once**, before any consumer, then digested:

```json
{
  "cells": [{"cell_id": "e:run7:call3:data.zscore", "metric": "zscore", "value": 3.1, "unit": "std"}],
  "omitted": 2847,
  "aggregates": {"zscore": {"min": -3.5, "max": 4.2, "count": 2848, "nan_count": 0}},
  "artifact_ref": "tool-results/9f2c….json",
  "truncated": true,
  "order_by": "abs_zscore_desc",
  "k": 50
}
```

- `truncated` and `omitted` are **always present**. Silent truncation is what makes a model invent the
  tail.
- `aggregates` covers **100%** of the omitted portion, so a claim outside the range is caught without
  reading everything.
- `order_by` and `k` come from **configuration, not from you**. If you could pick `k`, `k` would be an
  attack surface. Do not ask for a different `k` to make a result look better.

To read the omitted part, call `read_artifact(artifact_ref, jsonpath)`. The cell flips
`visible=False → True` and is journalled as a **new observation**.

## 6. No approximate matching

The old validator compared a claimed number against every number in the payload with a 0.5% tolerance.
With 2,000 scattered values the number line is densely covered, so *any* invented number is "close to"
some evidence. That tolerance is gone: a value is **substituted from its cell**, so it is either exactly
right or it is not a citation.

## 7. Anchoring: two hashes answer two different questions

| Hash | Question |
| :-- | :-- |
| `byte_sha256` | are these the same bytes on disk? |
| `content_fingerprint` = `sha256(canonical_json(values))` | are these the same values, regardless of codec or float sign? |

Every `AnomalySignal` and every evidence cell must carry `source_paths` + `content_fingerprint` +
`byte_sha256`. This is the **ONE SOURCE OF BYTES** invariant: a conclusion must be traceable back to a
specific parquet partition. A hypothesis resting on evidence that has since evaporated must be detectable.

**Known gap:** all seven M-RS1 scanners currently emit `source_paths: []`. Until that is fixed, an
`AnomalySignal` has no provenance anchor. When you report one, say the provenance anchor was empty rather
than implying it was verified.

## 8. Compaction must preserve citation anchors

When context is compacted, a cleared tool result becomes:

```json
{"cleared": true, "artifact_ref": "tool-results/…", "cell_ids": ["e:run7:call3:data.zscore", "…"]}
```

The values are gone; **the anchors remain** and are re-resolvable via `read_artifact`. Never accept a
`"[cleared]"` placeholder that drops the ids — that strands you in an unwinnable correction loop.

An LLM-written compaction summary is **not evidence**. It is post-processed by V1–V4 like anything else,
and every bare number in it is replaced by `[[value elided — see cell <id>]]`.

And the last line of defence: **`safe_fallback` must never emit a number.** A system built to stop
fabricated numbers must not end by printing one under the word "verified". It may state cell ids and
status only.

## 9. Identity of symbols

Symbol membership is checked against `AssetRegistry`, not guessed by a regex over your prompt.

- `symbol ∉ registry` ⇒ hard error naming the symbol. Never a silent pass.
- `registry.resolve()` returning > 1 ⇒ `ambiguous` ⇒ blocked.
- `asset_registry.db` currently holds **6 active PERP symbols**. That small whitelist is the truth of the
  data.
- **`instrument_type` is `PERP`, not `PERPETUAL`.** `all_active()` matches exactly, so one wrong word
  empties the whitelist and silently falls back to the old regex. That failure has no symptom — check the
  spelling.
- The measured difference: for *"Garman-Klass volatility rose, BTCUSDT zscore is 3.1, EMA20 flat"* the old regex
  returned `['VOLATILITY','ZSCORE','BTCUSDT','EMA20','ETHUSDT']`; the whitelist returns `['BTCUSDT']`.

## 10. Working checklist

1. Every number in your final answer sits inside `[[…]]`. No exceptions except the V4 whitelist.
2. Every cited cell is `visible=True` — you actually saw it.
3. The metric name in your sentence matches `cell.metric`.
4. NaN is reported as NaN with its cell, never omitted and never rounded to zero.
5. If the digest says `truncated: true`, either call `read_artifact` or state that your conclusion covers
   only the visible portion.
6. If you have no evidence, say so. Do not fill the gap from memory.
