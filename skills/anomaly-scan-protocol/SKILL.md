---
name: anomaly-scan-protocol
category: analysis
description: How the 7 Phase-A scanners filter (BY-FDR + adaptive tau, no bootstrap p-values), what tests_independent means downstream, and how to read INSUFFICIENT_DATA correctly.
---

# Anomaly Scan Protocol (M-RS1)

## Operating core

**Use when** — before calling `scan_lakehouse_anomalies`, and before interpreting its result.
**Settles** — what a scan outcome means, and what `tests_independent` is for.

**Procedure**
1. Read `outcome` first, not `signal_count`.
2. `INSUFFICIENT_DATA` / `SKIPPED` ⇒ report it as a coverage fact **with the reason**, and stop. That is the correct answer today (§7).
3. If signals exist, read `tests_independent` and carry it downstream.
4. Group by `event_cluster_id` / `co_fires_with` **before** counting. One cluster is one hypothesis, not five.
5. State the `source_paths` status honestly — all seven scanners currently emit an empty list.

**Never**
- Lower a threshold, shorten a baseline, or set `pit_override` to make something fire.
- Reuse `n_eff_symbol = 1.30` for a statistic other than returns.
- Invent an L2 definition to fill the Phase-B gap.


The scanner reads `ho_du_lieu/` with Polars and emits `AnomalySignal`s. It is **read-only** and never
overwrites source data.

Read `lakehouse-ground-truth` first. Everything here assumes you know the panel is 1 venue, 5 symbols,
30 columns, and 17 discrete days.

## 1. Why p < 0.01 was abandoned

Naive multiple testing at full coverage:

```text
7 scanners x 5 symbols x 3 windows x 3 venues x 8760 bars = 2,759,400 tests per pass
E[false positives] = 0.01 x 2,759,400 = 27,594 "anomalies" per random pass
```

Even at the realistic count (7 scanners × 5 symbols × windows ≈ **420,185 tests/year**):

| Method | Cut-off | Expected false anomalies/yr | Fatal flaw |
| :-- | --: | --: | :-- |
| uncorrected `p < 0.01` | 0.01 | **4,201.8** | p-hacking storm |
| bootstrap `N=10,000` | `p_min = 1e-4` | **42.0** | cannot resolve below `1e-4` — the resolution wall |
| Bonferroni `alpha=0.01` | `2.38e-8` | 0.01 | crushes the power of real signals |
| **Benjamini–Yekutieli FDR, q=0.10** | `p <= p_BY` | **<= 10% of emitted signals** | — robust under strong cross-correlation |

The decisive point: the correct Bonferroni threshold is **smaller than the smallest p-value a
10,000-draw bootstrap can represent**. So the bootstrap tail p-value is not conservative or liberal — it
is *unrepresentable*. `AnomalySignal.p_value` is therefore **nullable**, and ranking is by
`effect_size`, not by p.

## 2. The five-step filter

```text
DISCOVERY = partitions <= discovery_end (default: first 11)
CONFIRM   = partitions >  discovery_end (last 6; must contain >= 30 continuous days)

1 SCREEN        Z = (stat - median(baseline)) / (1.4826 * MAD(baseline))
                n_eff < 30 : p = 2 * Student_t_SF(|Z|, df = n_eff - 1)
                n_eff >= 30: p = 2 * Normal_SF(|Z|)

2 ADJUST        collect all M p-values for the pass; sort p(1) <= ... <= p(M)
                largest k with  p(k) <= (k / (M * C_M)) * q,  q = by_fdr_q (0.10)
                C_M = sum(1/i for i in 1..M)          <- Benjamini-Yekutieli, dependent case
                p_BY_threshold = p(k); none qualifies -> 0.0

3 ADAPTIVE TAU  tau_effective = Normal_PPF(1 - p_BY_threshold / 2)
                keep only tests with |Z_discovery| >= tau_effective

4 CONFIRM       emit a signal if and ONLY if all four hold:
                (a) |Z_discovery| >= tau_effective on DISCOVERY
                (b) |Z_confirm|   >= tau_effective * 0.5, same sign or regime-symmetric, on CONFIRM
                (c) n_effective   >= min_effective_obs
                (d) CONFIRM has >= 30 continuous days, else SKIPPED("confirm_data_insufficient")

5 DEGRADE       a scanner lacking its baseline is disabled INDIVIDUALLY and returns
                SKIPPED("insufficient_baseline") — it must not fail the whole pass.
                When the lakehouse grows, it flips back to ACTIVE with no code change.
```

## 3. `effective_n` is per scanner — never reuse 1.30

```python
# 1. temporal autocorrelation, Newey-West with Andrews bandwidth
q_bandwidth = int(4.0 * (n_rows / 100.0) ** (2.0 / 9.0))
rho_lags    = [autocorr(stat_k, lag=j) for j in range(1, q_bandwidth + 1)]
n_eff_time  = n_rows / (1.0 + 2.0 * sum((1.0 - j / (q_bandwidth + 1.0)) * rho_lags[j - 1]
                                        for j in range(1, q_bandwidth + 1)))

# 2. spatial correlation OF THIS STATISTIC across the 5 symbols (not of returns)
R_k          = correlation_matrix(stat_k_across_symbols)      # 5x5 on DISCOVERY
eig          = eigen_values(R_k)
n_eff_sym_k  = (sum(eig) ** 2) / sum(e ** 2 for e in eig)

# 3. combine
n_effective_k = n_eff_time * (n_eff_sym_k / S)
```

`n_eff_symbol = 1.30` from `lakehouse-ground-truth` §4 is the **returns** figure. Each statistic
(e.g. volatility, volume ratio) has its own correlation structure — do not reuse cross-statistic `n_eff` blindly.

## 4. Active OHLC & Volume Scanners

| # | Scanner | `raw_columns` | `feature_names` | Window / baseline | Non-obvious requirement |
| :-- | :-- | :-- | :-- | :-- | :-- |
| **A1'** | Moment regime shift | `close` | `return_skewness_24h`, `return_kurtosis_24h` | 24×1h / 30d | Sample kurtosis is heavy-tailed: apply a **Box-Cox** transform over the baseline before median/MAD. If λ does not converge ⇒ `INSUFFICIENT_DATA` for that cell. |
| **A2'** | Volatility regime & jump | `open,high,low,close` | `volatility_garman_klass_1h`, `volatility_parkinson_1h`, `gk_parkinson_ratio` | 24h / 30d | Emit only if `gk_parkinson_ratio > 2.0` (jump) **or** GK-z and Parkinson-z both exceed τ. `sub_type = "jump_event"` when the ratio > 2.0. |
| **A3'** | Volume–price decoupling | `close,volume` | `return_1h`, `volume_ratio_1h`, `return_volume_correlation_24h` | 24h / 30d | Rolling-window ρ has autocorrelation ≈ **0.96**. Baseline must use **non-overlapping 24h blocks** plus Newey-West standard errors. |

### Out-of-Scope (Non-OHLC) Scanners
*(Excluded for pure OHLC research: A4' Funding extreme, A5' OI–price lead-lag, A6' Liquidation cluster, A7' Sentiment crowding, B1-B3 Order book scanners)*

## 5. Loader-enforced rules — do not reimplement them in a scanner

| # | Rule |
| :-- | :-- |
| 1 | **Gap segmentation.** `bar_gap_ns = event_time_ns.diff()`; `is_break = bar_gap_ns > expected_bar_ns`; `segment_id = is_break.cum_sum()`. Every `rolling_*` / `shift` applies `.over(["symbol", "segment_id"])`. |
| 2 | **Gap manifest cross-check.** Read `ho_du_lieu/metadata/gap_manifest/{VENUE}/{YYYY-MM}.jsonl`. A manifest gap with no break in the frame ⇒ `ERROR`. |
| 3 | **Drop bad bars.** `is_gap_filled == True` or `dq_flags != 0`. Do not test `data_quality > 1` — its max is 0, so that branch is dead. |
| 4 | **Causal baseline + embargo.** Baseline over `[t − W − L, t − L)`, embargo `L` equal to the statistic's window length. |
| 5 | **Coverage + null run.** A cell is usable when non-null ≥ `max(min_effective_obs, 0.80 × L)` **and** the longest contiguous null run ≤ `0.10 × L` (A7: ≤ 2 days). Otherwise `INSUFFICIENT_DATA` for that cell only. |
| 6 | **At most 2 collects.** Pass 1 computes Z and p; pass 2 applies the BY-FDR filter. Peak RAM < 500 MB. |
| 7 | **PiT fail-closed.** Empty `PiTUniverseManager.get_universe(as_of)` ⇒ `INSUFFICIENT_DATA` unless `pit_override=True` is explicit. |
| 8 | **Two name spaces.** `raw_columns` from the parquet schema; `feature_names` from `FEATURE_REGISTRY`. Never mix them in one list. |

## 6. `AnomalySignal` — the fields that matter downstream

```text
signal_id  symbol  tier  raw_columns  feature_names  stat  value
effect_size  effect_confirm  n_effective  p_value(nullable)
tests_performed  tests_independent
segment_id  event_cluster_id  co_fires_with  window  observed_at
source_paths  content_fingerprint  byte_sha256
by_fdr_threshold  tau_effective  config_git_sha  pit_source  data_provenance
```

Three of these you must never drop or paraphrase:

- **`tests_independent`** — the statistical multiplicity for hypothesis testing. Read it off the tool
  result and **report it**. It is summed across an event cluster to accurately measure effective independent tests.
- **`event_cluster_id` / `co_fires_with`** — M-RS5 must produce **one** `HypothesisSpec` per cluster, not
  one per signal. Report the clustering, not the raw count.
- **`source_paths` + `content_fingerprint` + `byte_sha256`** — the provenance anchor. **Known defect: all
  seven scanners currently emit `source_paths: []`.** Say so when reporting; do not present an unanchored
  signal as traceable.

## 7. Reading the result correctly

Against today's snapshot **all seven scanners return `INSUFFICIENT_DATA` / `SKIPPED`**. Baselines of
7d/30d are not computable across 17 discrete days with 16 gaps of ~30 days.

**This is the correct output.** Report it as a data-coverage fact:

- ✅ "A1'–A7' returned `INSUFFICIENT_DATA(insufficient_baseline)`: the 30-day rolling baseline is not
  computable over 17 discrete days. This is a coverage limit, not a verdict on any hypothesis."
- ❌ "No anomalies found, so there is no alpha in this data."
- ❌ Loosening a threshold, shortening a baseline, or overriding PiT to make something fire.

`scan_lakehouse_anomalies` defaults to `lookback_days=365`, which exceeds the data by a wide margin. Do
not read the default as a claim about coverage.

## 8. Invariants

| # | Invariant |
| :-- | :-- |
| M1 | Read-only. Never overwrite source data. |
| M2 | Lazy, at most 2 `.collect(engine="streaming")` calls, peak RAM < 500 MB. |
| M3 | Filter by BY-FDR threshold; rank by `effect_size`; `p_value` is nullable. |
| M4 | Every signal is anchored: `source_paths` + `content_fingerprint` + `byte_sha256`. |
