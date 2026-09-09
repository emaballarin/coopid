# PROJECT.md — coopid

## What this is

A thin extension layer over [Cooper](https://github.com/cooper-org/cooper), the
deep-learning-first constrained-optimisation library. `coopid` does **not** reimplement
constrained optimisation: Cooper owns the `ConstrainedMinimizationProblem` abstraction,
the multiplier classes, the formulations (`Lagrangian`, `QuadraticPenalty`,
`AugmentedLagrangian`), the update schemes (simultaneous, alternating both ways,
extragradient) and `nuPI`, a proportional-integral controller for the dual.

Everything here is additive and composes with stock Cooper.

## Scope, in build order

1. **`BoundedMultiplier`** — done. `ExplicitMultiplier.post_step_` is exactly
   `weight.data = relu(weight.data)`: non-negativity, no ceiling. An unbounded multiplier
   is a documented failure mode, not a hypothetical.
2. **`EMAViolation`** — done. GECO's smoothing, routed through Cooper's own
   `ConstraintState.strict_violation` seam, which exists precisely to let the dual see a
   different (possibly non-differentiable) measurement from the primal.
3. **Constraint-level schedules** — done. `OpenLoopLevel`, `GatedLevel`.
4. **`nuPID`** — done. `nuPI` is PI; `Kd` on the second difference of the filtered error
   completes it in velocity form. The PI part is delegated to `nuPI` verbatim and the
   derivative increment applied from independent state, so `Kd = 0` is bit-identical by
   construction — and checked over 48 gain/init/maximize combinations anyway.
5. **Threshold calibration** — done. `calibrate` / `Calibration`: a level derived from a
   measured null rather than guessed, margins reported in per-batch spreads rather than raw
   units, and a dual learning rate made dimensionless as `gain / per_batch_sd`. The caller
   supplies the null-sampling callable, because only the caller knows the pipeline and the
   floor is a property of the pipeline.

## Scope is complete

All five items are implemented and tested. Further work is deepening rather than filling in:
a robust (median/MAD) calibration variant for heavy-tailed statistics, multi-constraint
convenience wrappers, and eventually a JAX backend as a sibling rather than a fork.

## Conventions

Mirrors `mdthermo`: hatchling + hatch-vcs (`coopid/_version.py` is generated, never
committed), `ruff.toml` vendored at the root so CI and pre-commit share one config,
pytest with a `slow` marker, Python >= 3.14, MIT.

## Non-goals

- Reimplementing anything Cooper already does.
- A JAX backend, for now. If it comes, it comes as a sibling backend and not a fork.
- Being a general control library. The controllers here exist to move Lagrange
  multipliers.
