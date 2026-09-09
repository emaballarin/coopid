"""Schedules for the constraint LEVEL, which is not the same thing as a penalty coefficient."""

import math
from typing import Any

__all__ = ["GatedLevel", "OpenLoopLevel"]


class OpenLoopLevel:
    r"""Move a constraint level from ``start`` to ``end`` on a fixed clock, then hold.

    Cooper ships :py:mod:`~cooper.penalty_coefficients` updaters, but those move the augmented
    Lagrangian's penalty coefficient :math:`c` -- how hard a violation is punished. This moves the
    **level** the constraint is measured against: the :math:`\kappa` in :math:`c(x) \leq \kappa`.
    They are different knobs and tightening the wrong one is a silent mis-experiment.

    Open-loop means the level ignores what the run is doing. That is a feature for a warm-up --
    a level that starts loose and tightens on a known clock is reproducible and cheap to reason
    about -- and a bug for anything adaptive, for which :py:class:`GatedLevel` exists.

    Args:
        start: Level at step 0.
        end: Level held from ``steps`` onward.
        steps: Number of calls to :py:meth:`step` spent in transit. ``0`` jumps immediately.
        mode: ``"linear"`` interpolates in the value, ``"exponential"`` in its logarithm.
            Exponential requires ``start`` and ``end`` to be strictly positive and of the same
            sign, and is the right choice whenever the level spans orders of magnitude -- which
            a constraint on a variance, a divergence or a test statistic usually does.

    Raises:
        ValueError: If ``steps`` is negative, if ``mode`` is unknown, or if ``mode`` is
            ``"exponential"`` with a non-positive endpoint.
    """

    def __init__(self, start: float, end: float, steps: int, *, mode: str = "linear") -> None:
        if steps < 0:
            raise ValueError(f"steps must be >= 0, got {steps}")
        if mode not in ("linear", "exponential"):
            raise ValueError(f"mode must be 'linear' or 'exponential', got {mode!r}")
        if mode == "exponential" and (start <= 0.0 or end <= 0.0):
            raise ValueError(f"exponential interpolation needs positive endpoints, got {start} and {end}")
        self.start = float(start)
        self.end = float(end)
        self.steps = int(steps)
        self.mode = mode
        self._t = 0

    @property
    def level(self) -> float:
        """The level for the current step."""
        if self.steps == 0 or self._t >= self.steps:
            return self.end
        fraction = self._t / self.steps
        if self.mode == "linear":
            return self.start + fraction * (self.end - self.start)
        return math.exp(math.log(self.start) + fraction * (math.log(self.end) - math.log(self.start)))

    def step(self) -> float:
        """Advance the clock by one and return the NEW level."""
        self._t += 1
        return self.level

    def state_dict(self) -> dict[str, Any]:
        """Dynamic state."""
        return {"t": self._t}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore from :py:meth:`state_dict`."""
        self._t = int(state["t"])

    def __repr__(self) -> str:
        return f"{type(self).__name__}(start={self.start}, end={self.end}, steps={self.steps}, mode={self.mode!r})"


class GatedLevel:
    r"""Tighten a constraint level only once the current one is actually being met.

    A closed-loop counterpart to :py:class:`OpenLoopLevel`. The level moves a fixed fraction of
    the way toward ``end`` each time the constraint has been satisfied for ``patience``
    consecutive observations, and never moves back. Between tightenings it holds, so the primal
    is always given time to reach the level it is currently being held to.

    This is the schedule to reach for when the *achievable* level is unknown in advance, which is
    the usual case: an open-loop clock that tightens past what the primal can reach makes the
    constraint permanently infeasible, and an infeasible constraint does not announce itself --
    it just pins the multiplier and quietly converts the objective into a penalty.

    Args:
        start: Initial (loose) level.
        end: Tightest level ever returned; the schedule approaches it geometrically.
        factor: Fraction of the remaining distance closed at each tightening, in ``(0, 1]``.
            ``1.0`` jumps straight to ``end`` on the first success.
        patience: Consecutive satisfied observations required before tightening.
        margin: A violation counts as satisfied when ``violation <= -margin``, so a positive
            margin demands the constraint be met with room to spare before tightening. Guards
            against ratcheting on a statistic that is only marginally satisfied by noise.

    Raises:
        ValueError: If ``factor`` is outside ``(0, 1]`` or ``patience`` is not positive.
    """

    def __init__(
        self,
        start: float,
        end: float,
        *,
        factor: float = 0.5,
        patience: int = 100,
        margin: float = 0.0,
    ) -> None:
        if not 0.0 < factor <= 1.0:
            raise ValueError(f"factor must be in (0, 1], got {factor}")
        if patience <= 0:
            raise ValueError(f"patience must be positive, got {patience}")
        self.start = float(start)
        self.end = float(end)
        self.factor = float(factor)
        self.patience = int(patience)
        self.margin = float(margin)
        self._level = float(start)
        self._streak = 0
        self.n_tightenings = 0

    @property
    def level(self) -> float:
        """The level for the current step."""
        return self._level

    def step(self, violation: float) -> float:
        """Record one observation and return the (possibly tightened) level.

        Args:
            violation: The constraint violation under Cooper's sign convention, so ``<= 0``
                means satisfied. Pass the SMOOTHED violation if one is available -- gating on a
                single noisy batch tightens on luck.
        """
        if float(violation) <= -self.margin:
            self._streak += 1
        else:
            self._streak = 0
        if self._streak >= self.patience:
            self._level += self.factor * (self.end - self._level)
            self._streak = 0
            self.n_tightenings += 1
        return self._level

    def state_dict(self) -> dict[str, Any]:
        """Dynamic state."""
        return {"level": self._level, "streak": self._streak, "n_tightenings": self.n_tightenings}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore from :py:meth:`state_dict`."""
        self._level = float(state["level"])
        self._streak = int(state["streak"])
        self.n_tightenings = int(state["n_tightenings"])

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(start={self.start}, end={self.end}, factor={self.factor}, "
            f"patience={self.patience}, margin={self.margin})"
        )
