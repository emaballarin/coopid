# NOTES.md — coopid

Running log of decisions that are not recoverable from the code or from `git log`.

## 09/09/2026 — GECO's EMA does not damp a pure-integral dual

Found while writing `tests/test_filters.py`, and it changed what the tests assert.

The intuitive claim — "smoothing the violation makes the multiplier quieter" — is **false** for a
dual driven by plain projected ascent. Measured on `min 0.5x² − 9x` s.t. `x ≤ 1` with N(0, 0.5)
noise on the violation, tail standard deviation of the multiplier over the last 1000 of 12000
steps:

| `decay` | tail sd of `mu` |
| ------- | --------------- |
| 0.0 (off) | 0.104 |
| 0.9     | 0.144 |
| 0.95    | 0.135 |
| 0.99    | 0.087 |

No useful reduction, and at shorter budgets heavy smoothing is clearly *worse* because the lag
has not washed out yet (decay 0.99 at 4000 steps: sd 0.525 against 0.122). The reason is
structural: `mu_{t+1} = mu_t + lr · violation` is **already** a low-pass filter, so pre-filtering
its input barely moves its output variance while definitely adding phase lag.

**Where the averaging does earn its keep**, and what the tests now assert instead:

- The **measurement** the dual consumes is genuinely quieter — variance ratio below 0.05 at
  `decay = 0.99`. That is the direct, mechanistic claim.
- A **proportional** path passes raw noise straight through, so it is the P term, not the I term,
  that needs a clean signal. This is why `coopid`'s filter and Cooper's `nuPI` compose rather
  than duplicate: `nuPI` smooths only its proportional path and integrates the raw error.
- **Gating** decisions (`GatedLevel`) read a level off a single number, and tightening on one
  lucky batch is unrecoverable — the level never loosens again.

Consequence for the docs: do not sell the filter as variance reduction for the multiplier. It is
a change of *what signal the dual integrates*, which matters most when something downstream is
not itself an integrator.
