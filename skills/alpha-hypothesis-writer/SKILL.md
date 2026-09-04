---
name: alpha-hypothesis-writer
category: strategy
description: How to build a HypothesisSpec that survives the mechanism-anchor gate — pre-registered commitments, the 8 fingerprint fields, horizon clamping, event clustering, and the 14 gaming paths that are already closed.
---

# Alpha Hypothesis Writer (M-RS5)

> **Phạm vi hoạt động (Data & Hypothesis Only):**
> Nhiệm vụ duy nhất của quy trình là **nghiên cứu dữ liệu, phát hiện điểm bất thường (anomalies) và nêu ra giả thuyết kinh tế/vi mô thị trường**.
> Không tạo hay xử lý các công thức toán học, biểu thức mã hay mở rộng toán tử.

## Operating core

**Use when** — before emitting a `HypothesisSpec`.
**Settles** — what makes a spec dispatchable rather than silently misattributed.

**Procedure**
1. Start from `mechanism_columns` — real panel columns. **Never use prose as the resolution key.**
2. Cluster the signals first (S9): one event cluster ⇒ one spec, `tests_independent` = **SUM**, not max.
3. Commit the eight fingerprint fields (§2.1) before anything runs.
4. Check anchor coverage: `1.0` ANCHORED → dispatch; `0 < c < 1` PARTIAL → dispatch flagged; `0.0` UNANCHORED → **do not dispatch**.
5. Clamp `predicted_horizon_band` against holdout. Empty feasible set ⇒ data queue, never the cemetery.

**Never**
- Generate mathematical formulas or code expressions.
- Choose `predicted_sign` after seeing a result.
- Emit `family`, `trigger_keywords`, or `economic_rationale` — all deleted.
- Let `tests_independent` or `gate_token` leave empty; either one breaks a cross-layer invariant.


Your output is **not prose**. Your output is a `HypothesisSpec` — a structured pre-registration whose
decision fields are hashed *before* any backtest runs. Prose is documentation attached to it and has
**no gate power**.

## 0. The single most important correction

An earlier version of this skill taught a sentence template and told you to feed it to
`resolve_templates()`. That gate is **dead code and actively harmful**:

- `resolve_templates` is fail-open by design. It sorts all templates and returns the top *n* with **no
  score floor** — the sorted list always has ≥ n elements. Zero overlap emits one `logger.warning` and
  continues.
- Measured today: `resolve_templates("wubba lubba dub dub zzzqqq xyzzy plugh")` returns **47 templates**
  and a real dealer-gamma hypothesis returns **39**. Both non-empty. The gate never rejects.
- So an out-of-vocabulary hypothesis is **not dropped — it is silently replaced** by a default grid
  (`close / high / low / volume / garman_klass_volatility`), backtested, and the result labelled
  with *your* hypothesis. Both outcomes are fatal:
  - grid **passes** → you deploy an alpha carrying a fictional economic story;
  - grid **fails** → the cemetery buries *your hypothesis text* and bans it permanently, though the real
    idea was never tested for one millisecond.

This is a **false-attribution** defect, not a coverage defect. Finite vocabulary is an acceptable
trade-off; manufacturing fake evidence is not.

**Never use prose as the resolution key. Use `mechanism_columns`.**

## 1. Four research principles (still binding, but only #1 is enforceable by text)

1. **Economic causality first** — a hypothesis starts from a concrete economic narrative, never from
   randomly combined operators.
2. **Signal parsimony** — prefer 2–3 variable interactions; avoid 5–6 variable formulas.
3. **Regime sensitivity** — state the target regime explicitly.
4. **Anti-pattern avoidance** — do not chase momentum at trend exhaustion; do not build on illiquid
   assets.

Be honest about principle 1: **it cannot be enforced by reading text.** A language model can write a
plausible economic story for any random formula, including auto-generated composites like
`min_coupling_return_skew_x_volume_ratio`. The old `economic_rationale` field appeared **0 times** in the
entire Python codebase — nobody read it, nobody checked it. It has been deleted. What replaces it is §3:
commitments that out-of-sample data adjudicates.

## 2. The payload

```json
{
  "hypothesis_id": "hyp_8f9a2b1c",
  "title": "Garman-Klass Volatility Jump Mean Reversion",

  "mechanism_columns": ["close", "high", "low", "volume"],
  "predicted_sign": -1,
  "predicted_horizon_band": ["1h", "4h"],
  "conditional_prediction": {"stronger_in": "high_vol", "weaker_in": "thin_depth"},
  "target_regime": "HIGH_VOLATILITY",
  "target_universes": ["top10_liquid"],
  "expansion_budget": 96,
  "vocab_version": "vocab_2026w31_a41c",

  "resolution_mode": "ANCHORED",
  "anchor_coverage": 1.0,
  "resolved_template_ids": ["volatility_garman_klass_1h", "volume_surge"],
  "resolved_families": ["V", "VOL"],

  "spec_fingerprint": "sha256:3c1f...",
  "evidence_signal_ids": ["sig_a1b2", "sig_c3d4"],
  "evidence_hash": "sha256:7f83b1...",
  "cemetery_verdict": { "...": "CemeteryVerdict object, NOT a float" },

  "tests_independent": 328,
  "gate_token": "hmac:9c4e...",

  "horizon_clamped": false,
  "budget_truncated": false,

  "narrative": "Extreme Garman-Klass volatility jump accompanied by abnormal volume surge reflects temporary liquidity exhaustion; momentum overextension triggers short-horizon price reversion.",
  "hypothesis_text": "Garman-Klass volatility jump normalized by ATR for short horizon price mean reversion",

  "generated_by": "ai_autonomous_layer2.6",
  "created_at": "2026-08-04T10:35:00Z"
}
```

### 2.1 The eight fingerprint fields — exactly eight, no more, no fewer

```text
spec_fingerprint = sha256(
    sorted(mechanism_columns) || predicted_sign || predicted_horizon_band ||
    canonical(conditional_prediction) || target_regime || target_universes ||
    vocab_version || expansion_budget
)
```

| Field | Meaning | Adjudicated by |
| :-- | :-- | :-- |
| `mechanism_columns` | the **real panel columns** the economic mechanism claims. This is the resolution key. | panel schema + anchor coverage (§4) |
| `predicted_sign` | ±1, committed before dispatch | sign of measured anomaly |
| `predicted_horizon_band` | a **band**, `len <= 4` | holdout arithmetic (§5) |
| `conditional_prediction` | ordering of effect size between two regimes, committed up front | effect comparison across regimes |
| `target_regime` | committed target regime | measured regime |
| `target_universes` | committed universe | checked at dispatch |
| `expansion_budget` | ceiling on distinct `ast_logic_hash` | enforced before generation |
| `vocab_version` | content hash of `vocab_templates.yaml` | recomputed at startup |

A commitment declared as "pre-registered" but **left out of the fingerprint** can be edited after seeing
results with nobody noticing. That is the most dangerous silent failure in this module. If you add a
field, add it in both places or not at all.

**`hypothesis_text` and `narrative` are NOT in the fingerprint.** Two different wordings of the same
commitment must give the same fingerprint, hence the same pool — structurally, not by luck. Measured on
the old prose-keyed path, four rewordings of one idea shared as little as **1%** of tested formulas
(AST Jaccard 0.010 / 0.117 / 0.011). That is not a research chain; that is four different studies wearing
one name.

### 2.2 Deleted fields — do not emit them

| Field | Why gone |
| :-- | :-- |
| `family` | dead parameter, see `alpha-family-taxonomy` §4. Use derived `resolved_families`. |
| `trigger_keywords` | appears 0 times in code. `resolve_templates` extracts its own keywords. |
| `economic_rationale` | appears 0 times in code. Replaced by §3 + non-gating `narrative`. |
| `cemetery_overlap: float` | a scalar cannot distinguish tier T-A1 (always 1.0) from T-C (text overlap) from T-B (Hamming distance — a different scale entirely). Pass the whole `CemeteryVerdict`. |

### 2.3 Two cross-layer fields you must never drop

| Field | Where it goes | What breaks if it is 0/empty |
| :-- | :-- | :-- |
| `tests_independent` | carried in `HypothesisSpec` | multiple testing adjustment loses track of test multiplicity. |
| `gate_token` | stage-1, spec-scoped: `HMAC(gate_key, spec_fingerprint ‖ plan_id ‖ gate_version)`; `execute()` refuses any job without it | the nightly path builds `VariantJob`s by hand from a queue file and enters the funnel having passed the cemetery gate **zero times**. |

Copy `tests_independent` from the contributing `AnomalySignal`s by **summation, never max** (§6).

## 3. Verifiable commitments replace the rationale field

| Field | How it is checked | What it actually stops |
| :-- | :-- | :-- |
| `mechanism_columns` | schema + `anchor_coverage` | the volatility story / volume formula mismatch. The mechanism must be **the same thing** that gets backtested. |
| `predicted_sign` | sign of effect | retrospective storytelling. A story that explains both signs is worthless; a pre-registered sign is not. |
| `predicted_horizon_band` | anomaly peak must fall inside the band | "right for the wrong reason" — alpha working at 24h while the mechanism claimed 15 minutes. |
| `conditional_prediction` | effect size ordering across two regimes | **the strongest gate.** A random formula has no reason to satisfy a pre-committed conditional ordering. |

`predicted_sign` combined with `conditional_prediction` drops the random pass probability of the pair to
**25%**. The cumulative `sign_hit_rate` per generator is persisted: a generator writing retrospective
stories converges to 0.5; one with a real economic premise does not. **This is the only metric in the
system that measures Layer 2.6 itself rather than the alphas it produces.**

## 4. The mechanism-anchor gate `phan_giai_co_neo()`

The gate does not ask *"are there any templates?"* (there always are). It asks *"can these templates
express the mechanism this hypothesis claims?"*

```text
mechanism_columns  (real panel columns, NOT prose)
     |
     v
phan_giai_co_neo()
     |
     +-- anchor_coverage == 1.0  --> ANCHORED    -> dispatch
     +-- 0 < coverage < 1.0      --> PARTIAL     -> dispatch WITH FLAG, not eligible for burial
     +-- anchor_coverage == 0.0  --> UNANCHORED  -> NO dispatch, NO burial
                                                  -> vocabulary-gap queue (Type A or B)
```

`anchor_coverage` = fraction of `mechanism_columns` that actually appear in the union of the resolved
templates' feature lists.

**Anti column-stuffing:** you could declare the easiest-to-match columns instead of the ones you mean —
keyword stuffing reborn at the column level. Two barriers: (a) the column must exist in the real panel
schema; (b) the column must be consistent with `predicted_sign` measured — a wrong column does not
systematically produce the right predicted direction. The cost of cheating moves from **free** (text
matching) to **has to be right out of sample**.

## 5. Horizon vs holdout — clamp the band, do not kill the idea

```text
band     = predicted_horizon_band
feasible = { h in band : test_end + h <= holdout_start }        # invariant I5

feasible == band        -> run the whole band
empty < feasible < band -> run feasible, set horizon_clamped = true, record the original band
|feasible| < 2          -> treat as empty (a single horizon cannot separate signal from coincidence)
feasible == empty       -> do NOT run, do NOT bury -> insufficient-data queue (needs more history)
```

This is decided by M-RS5 **deterministically, with no LLM involved**. `len(band) <= 4` is a hard cap;
without it you would declare an absurd band (`1m`–`30d`) so something is always feasible.

Cutting *inside* the band still tests the claim. Losing the *whole* band means the claim is not testable
with the data on hand — and **not testable is not the same as false**, so the empty branch goes to the
data queue, never the cemetery.

## 6. Event clustering (S9) — one event, one spec

```text
Group AnomalySignals into CLUSTERS before building any spec:
  same event_cluster_id                            => one cluster
  a.signal_id in b.co_fires_with (or vice versa)   => merge the two clusters

Each cluster => EXACTLY ONE HypothesisSpec:
  mechanism_columns   = union of raw_columns / feature_names over the cluster (deduplicated)
  predicted_sign      = sign of the member with the largest |effect_size|
  evidence_signal_ids = every signal_id in the cluster
  tests_independent   = SUM over the cluster — never max
```

**Why sum, not max:** the cluster was found by several independent tests running in parallel.
Multiple-testing multiplicity adds across trials. Taking the max is shrinking your own denominator.

## 7. Gaming paths that are already closed — do not attempt them

| # | Path | Why it fails |
| :-- | :-- | :-- |
| 1 | stuff dictionary keywords into `hypothesis_text` | resolution left prose; text is not in the fingerprint |
| 2 | declare easy-to-anchor columns | schema check + sign check out of sample |
| 3 | absurdly wide horizon band | `len(band) <= 4`; a wide band weakens the claim |
| 4 | pick the sign after peeking at correlation | sign hashed before dispatch; `sign_hit_rate` tracked per generator |
| 5 | throwaway `conditional_prediction` | hashed and checked; paired with sign it is 25% random |
| 6 | over-declare `expansion_budget` and truncate later | budget limit enforced before AST generation |
| 7 | split one event cluster into 5 specs | S9 forces clustering before spec construction |
| 8 | ignore the negative controls | control matrix (§8 of `alpha-diagnostic-loop`) distinguishes `MAPPING_DEGENERATE` |
| 9 | hand-built `plan_id` on the nightly path | `execute()` verifies the HMAC keyed on `spec_fingerprint` |
| 10 | stale `vocab_version` string | recomputed as a content hash at startup |
| 11 | collapse the cemetery verdict to a float | the full `CemeteryVerdict` object is passed through |
| 12 | let `tests_independent` fall to 0 or 1 | S8 requires `>= 1` and the value is traced end to end |
| 13 | launder a hybrid alpha into one family | budget charged to **all** `resolved_families` |
| 14 | assume SOFT_BLOCK is a hidden pass | by design it dispatches, and it is counted in `n_soft_block` |

## 8. Exit conditions — a spec may not leave the module unless all hold

| # | Condition |
| :-- | :-- |
| N1 | `mechanism_columns` non-empty and every element is in the panel schema |
| N2 | `ANCHORED` → dispatch; `PARTIAL` → dispatch flagged; `UNANCHORED` → **no dispatch** |
| N3 | `predicted_sign` in {+1, −1} and in the fingerprint before dispatch |
| N4 | `conditional_prediction` committed up front |
| N5 | `predicted_horizon_band` is a band, `len <= 4`, `feasible` non-empty |
| N6 | `expansion_budget` declared up front; enforced before generation |
| N7 | `evidence_hash` anchors to real parquet partitions |
| N8 | `vocab_version` and `spec_fingerprint` are recorded |
| N9 | cemetery eligibility requires `ANCHORED` AND not `budget_truncated` AND not `horizon_clamped` AND the negative control failed |
| N10 | `tests_independent >= 1` and `gate_token` non-empty |

Blocking rule at the entry gate: read `cemetery_verdict.outcome` and **only `HARD_BLOCK` blocks**.
`SOFT_BLOCK` (text tier) and `WARNING` still dispatch — see `cemetery-containment-guard`.

## 9. Worked example — a good spec and a bad one

**Good.** Anomaly A2' fires: Volatility regime & Garman-Klass jump on 1h bars, effect size −2.9,
`tests_independent = 31`.

```text
mechanism_columns       ["open", "high", "low", "close"]   <- real columns, ground truth §2
predicted_sign          -1        <- extreme volatility jump precedes negative short-term return
predicted_horizon_band  ["1h", "4h"]
conditional_prediction  {"stronger_in": "high_vol", "weaker_in": "low_vol"}
target_regime           "HIGH_VOLATILITY"
tests_independent       31        <- copied from the signal, summed if clustered
=> resolved_families    {"V"}  (derived)     resolution_mode ANCHORED, anchor_coverage 1.0
```

**Bad — and exactly how it fails.** "When dealer gamma exposure flips negative, spot variance
acceleration predicts the delta-hedge feedback loop."

```text
mechanism_columns  ["gex", "iv_skew"]  -> absent from the panel  -> N1 violated
                                       -> anchor_coverage = 0.0 -> UNANCHORED
                                       -> Type-B gap (missing DATA) -> data ticket
                                       -> NOT dispatched, NOT buried, NOT counted as a KILL
```

Report the second case as a data gap with the exact missing columns named. Do **not** rewrite it into a
price volatility hypothesis to make it dispatchable — that is the false attribution this whole module exists to
prevent.
