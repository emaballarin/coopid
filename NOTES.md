# NOTES.md — coopid

Running log of decisions that are not recoverable from the code or from `git log`.

## 09/09/2026 — GECO's EMA does not damp a pure-integral dual

Found while writing `tests/test_filters.py`, and it changed what the tests assert.

The intuitive claim — "smoothing the violation makes the multiplier quieter" — is **false** for a
dual driven by plain projected ascent. Measured on `min 0.5x² − 9x` s.t. `x ≤ 1` with N(0, 0.5)
noise on the violation, tail standard deviation of the multiplier over the last 1000 of 12000
steps:

| `decay`   | tail sd of `mu` |
| --------- | --------------- |
| 0.0 (off) | 0.104           |
| 0.9       | 0.144           |
| 0.95      | 0.135           |
| 0.99      | 0.087           |

No useful reduction, and at shorter budgets heavy smoothing is clearly _worse_ because the lag
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
a change of _what signal the dual integrates_, which matters most when something downstream is
not itself an integrator.

## 09/09/2026 — the D term, and why it is a second difference

`nuPI` is a PI controller in **velocity form**: it emits `Delta mu_t`, not `mu_t`. Read that way
its update is the textbook incremental PI,

    Delta mu_t = lr * ( Ki * e_t + Kp * (xi_t - xi_{t-1}) )

with `xi` an EMA of the error, so the proportional path sees a _filtered_ first difference. The
incremental form of PID adds a **second** difference, which is what `nuPID` contributes:

    Delta mu_t += lr * Kd * (xi_t - 2 xi_{t-1} + xi_{t-2})

Defined on `xi`, never on the raw error. A first difference amplifies white noise by sqrt(2) and
a second by 2, so a raw D term on a per-batch constraint violation is a noise generator. Hence
the warning when `Kd != 0` with `ema_nu == 0`.

**Measured effect**, on `min 0.5x^2 - 9x` s.t. `x <= 1` (closed form: `x* = 1`, `mu* = 8`), with
`Kp = 0`, `Ki = 1`, `nu = 0.9`, both learning rates 0.3:

| `Kd` | multiplier overshoot | `x` final | `mu` final |
| ---- | -------------------- | --------- | ---------- |
| 0    | 2.282                | 1.0000    | 8.0000     |
| 2    | 1.982                | 1.0000    | 8.0000     |
| 5    | 1.561                | 1.0000    | 8.0000     |
| 10   | 0.934                | 1.0000    | 8.0000     |
| 20   | **0.263**            | 1.0000    | 8.0000     |

Monotone, 8.8x at `Kd = 20`, equilibrium untouched. Unlike the EMA result recorded above, this
one matched the intuition -- because overshoot is a _transient_ property and damping is exactly
what a derivative term buys, whereas the earlier claim was about steady-state variance through
an integrator.

**Design note.** The PI part is delegated to `nuPI.step()` verbatim and the derivative applied
afterwards from separate state (`d_xi`, `d_diff`). That makes `Kd = 0` bit-identical to stock
`nuPI` _by construction_ rather than by numerical luck, keeps Cooper's two init schemes and both
sparse paths working untouched, and means a Cooper upgrade cannot silently change the PI
behaviour here. The cost is that the D path is dense-only; sparse gradients raise while `Kd != 0`.

## 09/09/2026 — an unreachable constraint does not merely fail to bind; it wrecks the primal

Found while building the calibration end-to-end test. The setup: `min 0.5t² − 3t` subject to
`mean((t + eps)²) ≤ level`, where the statistic is a mean of squares and therefore has a **floor
of 1**. A hardcoded `level = 0` is unreachable by construction.

The expected symptom is the multiplier pinning at its ceiling, and that happens. The _unexpected_
one is that the run diverged: `theta` reached 1.8e12. The cause is conditioning, not the
constraint. The primal objective is `0.5t² − 3t + mu(t² + 1)`, whose curvature is `1 + 2mu`. With
`mu` pinned at 100 that is **201**, so gradient descent is stable only for `lr < 2/201 ≈ 0.00995`
— and the primal was running at `1e-2`, a hair over the line.

So a saturated multiplier is not a contained failure. It silently multiplies the primal's
effective curvature, and any step size chosen against the unconstrained problem can cross its
stability boundary without anything in the constraint machinery reporting a problem. Two
practical consequences:

- `BoundedMultiplier.is_saturated()` should be logged every run, not inspected after a failure.
- A cap is not only a weighting decision (as its docstring says) but a **step-size** decision for
  the primal. Set `upper` with the primal's learning rate in mind, or the cap that was supposed
  to be a safety backstop becomes the thing that detonates.

The test uses `lr = 1e-3` so both arms are stable and the contrast is about reachability alone:
naive level 0 → multiplier pinned at the cap, constraint still violated by more than the entire
floor; calibrated level → multiplier at 0.69, well off the ceiling, constraint met.
