# PA1 — Baseline ("any random day")

Spec: `mpt/research/defense_entry_score.md` §B6.1 (plain idea: §L11). Today's plan: `mpt/research/target_11_oct.md`.

## 1. What it is

PA1 answers: **"If I ignore everything about today and buy this ticker on any day, how fast does money grow per day under the monitor's sell rules?"**

It takes every pretend trade in the ticker's last three years (the entry table: buy at each day's close, walk forward minute by minute through the GTT 10% and daily SL-M 5% nets until sold) and averages them with Option 2:

```text
pa1_r_tw = Σ e_i · G_i  /  Σ e_i · τ_i
```

- `G_i = ln(sell / buy)` — log growth of pretend trade `i`.
- `τ_i` — trading days held (fractional; 375 minutes = 1 day).
- `e_i ∈ {0,1}` — eligibility. 0 only for recent unfinished trades with `τ < 5` days (near-copies of today, almost no information). Everything else counts.

Unit: log return per trading day. `0.0005` = 0.05%/day. Display derived: `exp(250·r) − 1` as "% per year if repeated".

Unfinished trades (no net touched yet, `τ ≥ 5`) are included, valued at the latest close. This biases PA1 slightly (they look like short trades); PB1 later fixes that properly.

## 2. Why it is a number worth having

Two reasons, one practical and one mathematical.

**Practical.** It is the ticker's own track record under this exact exit machine. If HAL's baseline is +0.03%/day, a random entry earned that. Any claim that "today is a good day to enter" has to beat it.

**Mathematical (Wald, §L11).** For a random walk with drift μ, a stop-based exit does not change the per-day time-weighted return: it equals μ with or without nets. So for a stock with no timing signal, `pa1_r_tw ≈ μ`, the ticker's raw drift. Nets change the *shape* (many small losses, a few large wins), not the average per day. Consequences:
- PA1 is mostly "the ticker's drift over the last three years" seen through the monitor.
- Any predictor whose forecast differs from PA1 is claiming that **state at entry matters**. That difference is the only possible source of edge.
- The Wald property is also the engine's correctness test: simulate minute GBM with a known drift, PA1 must recover it. If it doesn't, the simulator or the stats are wrong.

## 3. How it is used as the baseline

Everything is compared against PA1; nothing is compared against zero.

| Use | Definition |
|---|---|
| **Edge of a predictor** | `edge_m = m_r_tw − pa1_r_tw` — what the predictor claims beyond "any day". |
| **Combined edge** | `cmb_edge = cmb_r_tw − pa1_r_tw` (§B7). The single "is today special for this ticker?" number. |
| **Shrinkage anchor** | The combiner shrinks every other predictor toward PA1: `(N0·pa1 + neff_m·m) / (N0 + neff_m)`, `N0 = 20`. Thin evidence → the answer collapses to the baseline. |
| **Validation** | "Edge vs PA1" metric (§B12): realised `ΣG/Στ` of entries where `p_m > p_pa1` minus where `p_m ≤ p_pa1`. Must be > 0 to pass. |
| **Limiting case** | PA3 → PA1 as bandwidth `h → ∞`; PB1 → PA3 → PA1 when nothing is censored and nothing is state-conditioned. Used as unit tests. |
| **neff for simulators** | PC2 and PD1 report PA1's `neff` (simulating more paths does not create more history). |

How to read a result:
- `pa3_r_tw` (or any predictor) ≈ `pa1_r_tw` → today looks ordinary, no timing claim.
- Predictor ≫ PA1 and its `[lo, hi]` excludes PA1 → state-conditioned evidence that today is better than random.
- PA1's own `[lo, hi]` includes 0 → the ticker's baseline isn't distinguishable from "nothing", whatever the predictors say.

## 4. Inputs

Entry table of one ticker as of session `T` (§B3.3): `d0, P0, G, τ, r = G/τ, censored, by, state(d0)`, plus `e_i`. No state features are used by PA1 itself. Needs ≥ `min_history_days` (250) valid sessions.

## 5. Computation

```text
w_i = e_i
stats.compute(G, tau, w, d)
```

Outputs (the standard block, prefix `pa1_`; §B4):

| Field | Formula | Meaning |
|---|---|---|
| `r_tw` | `Σ w G / Σ w τ` | the baseline per-day return (main) |
| `r_mean` | `Σ w r / Σ w` | simple average of r, shown beside only |
| `sd` | `sqrt(Σ v (r − r_tw)² / Σ v)`, `v = w τ` | spread of per-trade r |
| `score` | `r_tw / sd` | risk-adjusted; null if `sd = 0` |
| `p_pos` | `Σ w 1[G>0] / Σ w` | share of trades that made money |
| `days` | `Σ w τ / Σ w` | average holding time |
| `p_1d` | `Σ w 1[τ≤1, not censored] / Σ w` | chance of a sale within the first day |
| `n` | count of `w > 0` | rows used |
| `neff` | `kish / over`, `kish = (Σw)²/Σw²`, `over = max(1, Σwτ/Σw)` | independent-trade evidence |
| `lo`, `hi` | 5th/95th pct of block-bootstrap `r_tw` | 90% range |
| `status` | `ok / thin / empty / failed / insufficient_history` | `thin` if `neff < 10` |

PA1 extras: `pa1_share_gtt` (share of finished exits via GTT; the rest via SL-M), `pa1_censored_share`, `pa1_n_days` (distinct entry days used).

Block bootstrap (§B4.2): rows sorted by `d0`, block length `L = ceil(over)` consecutive entry days, `B = 500` resamples, seed = hash(ticker, T) so the same inputs give the same range.

## 6. Worked example (§L8e)

Three back-to-back pretend trades:

| Trade | G | τ (days) | r = G/τ |
|---|---|---|---|
| A | −0.0513 | 0.3 | −17.1% |
| B | +0.1823 | 60 | +0.30% |
| C | +0.0953 | 30 | +0.32% |

- Simple mean of r: −5.49%/day (misleading, A dominates).
- `r_tw = (−0.0513 + 0.1823 + 0.0953) / (0.3 + 60 + 30) = 0.2263 / 90.3 = +0.25%/day` (matches the 25.4% total gain over 90.3 days).

This exact case must be a unit test of `stats.compute` with `w = 1`.

## 7. Assumptions and known biases

- The ticker is stationary enough over three years. A one-way trend (long bull run) goes straight into PA1 (Wald: nets don't remove drift). A baseline of +0.15%/day in a bull market is the *market's* gift, not an entry skill.
- Neighbouring entries share futures; handled via `neff` and block bootstrap, not by dropping rows.
- Slippage (0.1%) is an assumption, not measured from prod fills.
- Unfinished trades with `τ ≥ 5` are valued at the last close (slightly optimistic or pessimistic depending on the recent move); PB1 corrects this.

## 8. Look-ahead guards

Entry table ends at session `T − 1` (entries up to and including yesterday's close); unfinished rows are valued at `close(T)`, never later; today's row is not in the table.

## 9. Tests

1. **Wald check:** minute GBM with drift μ and no state dependence → `pa1_r_tw ≈ μ`, with μ inside `[lo, hi]` for ≥ 90% of 100 seeds. Validates simulator + stats + bootstrap together.
2. **Three-trade example** (§6 above) → 0.25%/day, simple mean −5.49%.
3. **Constant-r table** → `r_tw = r`, `sd = 0`, `score` null.
4. **Recent-censored rule:** rows with `censored` and `τ < 5` get `w = 0` and do not move the result.
5. **Determinism:** same inputs → identical output including `lo, hi`.
6. **Status:** `neff < 10` → `thin`; no eligible rows → `empty`; < 250 valid days → `insufficient_history`.

## 10. Failure modes

| Condition | Status |
|---|---|
| ticker history < 250 valid days | `insufficient_history` (no numbers) |
| `neff < 10` | `thin` |
| no eligible rows | `empty` |
| exception | `failed` (runner isolates; others continue) |

## 11. Relationships

- **PA3** reduces to PA1 as `h → ∞` (unit test). PA3 is "PA1 restricted to look-alike days".
- **PA2** is PA1 restricted to U-signal days.
- **PB1** equals PA3 (and PA1 in the limit) when nothing is censored.
- **CMB** shrinks toward PA1 and reports `cmb_edge` against it.
