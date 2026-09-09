"""GECO's smoothing, routed through Cooper's own primal/dual measurement split."""

from typing import Any
from typing import Optional
from typing import Union

import torch
from cooper.constraints import ConstraintState

__all__ = ["EMAViolation"]


class EMAViolation:
    r"""An exponential moving average of the constraint violation, for the DUAL update only.

    This is the smoothing of GECO (Rezende and Viola, 2018, arXiv:1810.00597):

    .. math::
        \bar{c}_t = \gamma \, \bar{c}_{t-1} + (1 - \gamma) \, c_t

    with the multiplier integrating :math:`\bar{c}_t` rather than the raw per-batch
    :math:`c_t`.

    **This is not what** :py:class:`~cooper.optim.nuPI` **does, and the difference is the point.**
    ``nuPI`` also carries an EMA, but of the error signal on its *proportional* path::

        xi_t     = nu * xi_{t-1} + (1 - nu) * e_t
        theta_t1 = theta_t - lr * (Ki * e_t + Kp * (xi_t - xi_{t-1}))

    -- the **integral term acts on the raw** :math:`e_t`. GECO integrates the smoothed signal.
    The two are complementary: a `nuPI` dual optimiser fed a violation smoothed by this class
    gets both, which is a controller neither library offers on its own.

    The mechanism is Cooper's own :py:class:`~cooper.constraints.ConstraintState`, which already
    separates the measurement that drives the primal from the one that drives the dual. The raw,
    differentiable violation stays in ``violation``, so the primal gradient is exact and
    unsmoothed; the detached average goes in ``strict_violation``, which Cooper documents as
    "the measurement of the constraint violation used to update the dual variables". No
    subclassing and no fork.

    **What this does not buy.** Smoothing does *not* meaningfully quieten a multiplier driven by
    plain projected ascent: that update is itself a low-pass filter, so pre-filtering its input
    changes its output variance little while definitely adding lag. Measured numbers are in
    `NOTES.md`. The averaging earns its keep where something downstream is *not* an integrator --
    a proportional term, or a gating decision such as :py:class:`~coopid.schedules.GatedLevel`.

    Args:
        decay: :math:`\gamma`, in ``[0, 1)``. ``0`` disables smoothing exactly.
        init: Seed for :math:`\bar{c}_{-1}`. The default, ``None``, seeds at the **first observed
            violation**, so the first dual step is bit-identical to the unsmoothed one and
            enabling smoothing cannot change where a run starts. Pass a number to seed
            explicitly; seeding at ``0`` reintroduces the usual warm-up bias toward zero, which
            reads as "the constraint is satisfied" for the first few hundred steps.

    Raises:
        ValueError: If ``decay`` is outside ``[0, 1)``.
    """

    def __init__(self, decay: float = 0.99, *, init: float | None = None) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        self.decay = float(decay)
        self.init = init
        self._average: torch.Tensor | None = None

    @property
    def average(self) -> torch.Tensor | None:
        """The current smoothed violation, or ``None`` before the first update."""
        return self._average

    def reset(self) -> None:
        """Forget the running average; the next update re-seeds it."""
        self._average = None

    @torch.no_grad()
    def update(self, violation: torch.Tensor) -> torch.Tensor:
        """Fold one observation into the average and return it, detached.

        The stored state is always detached. Keeping it attached would retain the autograd graph
        of every earlier batch, which grows without bound and is never what is wanted: the dual
        ascends on a *measurement*, not on a differentiable path back to the primal parameters.
        """
        current = violation.detach()
        if self._average is None:
            seed = current if self.init is None else torch.full_like(current, float(self.init))
            self._average = seed.clone()
        else:
            self._average = self.decay * self._average + (1.0 - self.decay) * current
        return self._average.clone()

    def constraint_state(self, violation: torch.Tensor, **kwargs: Any) -> ConstraintState:
        """A :py:class:`~cooper.constraints.ConstraintState` with a raw primal and a smoothed dual.

        Args:
            violation: The raw, **differentiable** constraint violation, sign-conventioned as
                Cooper expects (``<= 0`` means satisfied for an inequality constraint).
            **kwargs: Forwarded to :py:class:`~cooper.constraints.ConstraintState` --
                ``constraint_features``, ``contributes_to_primal_update`` and so on.

        Raises:
            ValueError: If ``strict_violation`` is passed, since supplying it would silently
                discard the smoothing this object exists to apply.
        """
        if "strict_violation" in kwargs:
            raise ValueError(
                "`strict_violation` is what EMAViolation supplies; passing it here would discard "
                "the smoothing. Use a plain ConstraintState if that is what you want."
            )
        return ConstraintState(violation=violation, strict_violation=self.update(violation), **kwargs)

    def state_dict(self) -> dict[str, float | torch.Tensor | None]:
        """Dynamic state, for checkpointing alongside the multiplier."""
        return {"decay": self.decay, "init": self.init, "average": self._average}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore from :py:meth:`state_dict`."""
        self.decay = float(state["decay"])
        self.init = state["init"]
        average = state["average"]
        self._average = None if average is None else torch.as_tensor(average).clone()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(decay={self.decay}, init={self.init})"
