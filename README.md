# coopid

Controller-shaped Lagrange multipliers for [Cooper](https://github.com/cooper-org/cooper).

Cooper already provides the substrate for constrained optimisation in PyTorch: a
`ConstrainedMinimizationProblem`, multipliers as `nn.Module`s, Lagrangian / quadratic-penalty /
augmented-Lagrangian formulations, simultaneous, alternating and extragradient update schemes,
and `nuPI` — a proportional-integral controller for the dual variable.

`coopid` adds the few things it does not have, each of which is a documented failure mode rather
than a hypothetical:

| what                           | why                                                                                                                                                                                                                                                              |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`BoundedMultiplier`**        | Cooper projects inequality multipliers onto the non-negative orthant and nothing more. With no ceiling a multiplier can grow until its term dwarfs the rest of the loss, at which point the objective is ill-posed rather than merely badly conditioned.         |
| **`EMAViolation` (GECO)**      | `nuPI` smooths the error signal on its _proportional_ path only; its integral acts on the raw violation. GECO integrates the _smoothed_ constraint. These are different controllers, and Cooper implements one of them.                                          |
| **Constraint-level schedules** | Cooper's `penalty_coefficient_updaters` move the augmented-Lagrangian penalty `c`. Nothing moves the constraint _level_ itself, which is what a warm-up or a gated tightening needs.                                                                             |
| **`nuPID`**                    | `nuPI` is a proportional-integral controller. Adding `Kd` on the second difference of the filtered error completes the PID. Measured on a problem with a closed-form saddle: dual overshoot falls **8.8x** from `Kd=0` to `Kd=20`, with the equilibrium unmoved. |

| **`calibrate` / `Calibration`** | A constraint level is a number someone has to choose, and for a statistic estimated from a finite batch there is no defensible constant: it has a floor, the floor is not zero, and it moves with sample size, dimension and preprocessing. Measure it instead. |

### Calibration, concretely

```python
from coopid import calibrate

# Push draws from the NULL through the SAME pipeline the real statistic goes through.
cal = calibrate(lambda g: my_statistic(sample_from_null(g)), n_repeats=2000)

level = cal.level(n_sd=3.0)  # three per-batch spreads above the measured floor
dual_lr = cal.dual_lr(gain=0.1)  # dimensionless: `gain` of multiplier per spread of violation
margin = cal.margin_sd(observed)  # report in spreads, never in raw units
```

`cal.standard_error` is deliberately separate from `cal.per_batch_sd`; they differ by
`sqrt(n_repeats)` and confusing them fails in both directions at once — the level comes out too
tight to reach _and_ the ascent rate too slow to arrive.

## Install

```bash
pip install coopid
```

## Use

Everything here is additive. A `coopid` object is a Cooper object, and anything Cooper accepts
still works.

```python
import cooper
import torch
from coopid import BoundedMultiplier, EMAViolation

multiplier = BoundedMultiplier(num_constraints=1, upper=1e3)
constraint = cooper.Constraint(
    multiplier=multiplier,
    constraint_type=cooper.ConstraintType.INEQUALITY,
    formulation_type=cooper.formulations.Lagrangian,
)
smoother = EMAViolation(decay=0.99)

# inside compute_cmp_state, with `statistic` differentiable and `level` the target:
state = smoother.constraint_state(statistic - level)
```

`smoother.constraint_state` returns a `cooper.ConstraintState` whose `violation` is the raw
differentiable quantity (so the primal gradient is exact) and whose `strict_violation` is the
detached EMA (so the dual integrates the smoothed signal). That is GECO, expressed in Cooper's
own API.
