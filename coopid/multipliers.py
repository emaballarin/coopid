"""Lagrange multipliers with a ceiling, not only a floor."""

from typing import Optional

import torch
from cooper.multipliers import DenseMultiplier
from cooper.utils import ConstraintType

__all__ = ["BoundedMultiplier"]


class BoundedMultiplier(DenseMultiplier):
    r"""A :py:class:`~cooper.multipliers.DenseMultiplier` projected onto a box, not a half-line.

    Cooper's :py:meth:`~cooper.multipliers.ExplicitMultiplier.post_step_` is exactly
    ``weight.data = relu(weight.data)``: it enforces :math:`\mu \geq 0` for inequality
    constraints and imposes no ceiling. That is the right default for a solver, and the wrong
    one whenever the constraint statistic is small and the ascent rate is absolute, because the
    multiplier then has to grow by orders of magnitude before its term competes -- and nothing
    stops it growing further.

    An unbounded multiplier is a **documented failure mode, not a hypothetical**: once
    :math:`\mu \cdot (\text{typical violation})` dwarfs the other loss terms, the remaining terms
    stop constraining anything and the objective becomes ill-posed rather than merely badly
    conditioned. The symptom is a mechanism that "fails" while its own diagnostic looks healthy.

    The right ``upper`` is problem-dependent and cannot be guessed by this class, because it
    depends on the scale of the *other* terms in the loss, which the multiplier cannot see. Set
    it so that ``upper * (typical violation)`` is comparable to the terms it must compete with.

    Args:
        num_constraints: Number of constraints, as in the base class.
        init: Initial multiplier values, as in the base class.
        lower: Floor for the multiplier. ``None`` (the default) keeps Cooper's behaviour --
            :math:`0` for an inequality constraint and unbounded below for an equality one.
            The constraint type is only known once the multiplier is attached to a
            :py:class:`~cooper.constraints.Constraint`, so this is resolved at projection time.
        upper: Ceiling for the multiplier. ``None`` leaves it unbounded above, which is exactly
            stock Cooper.
        device: As in the base class.
        dtype: As in the base class.

    Raises:
        ValueError: If ``lower`` and ``upper`` are both given and ``upper < lower``.
        ValueError: If ``init`` is given and falls outside an explicitly requested bound.
    """

    def __init__(
        self,
        num_constraints: int | None = None,
        init: torch.Tensor | None = None,
        *,
        lower: float | None = None,
        upper: float | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(num_constraints=num_constraints, init=init, device=device, dtype=dtype)
        if lower is not None and upper is not None and upper < lower:
            raise ValueError(f"upper ({upper}) must be >= lower ({lower})")
        if init is not None:
            if lower is not None and torch.any(init < lower):
                raise ValueError(f"`init` has entries below the requested lower bound {lower}")
            if upper is not None and torch.any(init > upper):
                raise ValueError(f"`init` has entries above the requested upper bound {upper}")
        self.lower = lower
        self.upper = upper

    def resolved_lower(self) -> float | None:
        """The floor actually applied, once the constraint type is known.

        An explicit ``lower`` always wins. Otherwise an inequality multiplier keeps Cooper's
        implicit floor of zero and an equality multiplier stays unbounded below.
        """
        if self.lower is not None:
            return self.lower
        if getattr(self, "constraint_type", None) == ConstraintType.INEQUALITY:
            return 0.0
        return None

    @torch.no_grad()
    def post_step_(self) -> None:
        """Project the multiplier onto ``[resolved_lower, upper]`` after each dual step."""
        lower = self.resolved_lower()
        # `torch.clamp` raises when both bounds are None, which is the legitimate configuration
        # of an unbounded equality multiplier -- projection is then simply a no-op.
        if lower is None and self.upper is None:
            return
        self.weight.data = self.weight.data.clamp(min=lower, max=self.upper)

    def is_saturated(self) -> torch.Tensor:
        """Per-constraint mask of multipliers sitting on the ceiling.

        Worth logging every run. A multiplier pinned at ``upper`` means the bound is binding, and
        a bound that binds is a **weighting decision being made by the cap** rather than by the
        constraint -- which may be exactly what is wanted, but should never be discovered late.
        """
        if self.upper is None:
            return torch.zeros_like(self.weight.data, dtype=torch.bool)
        return self.weight.data >= self.upper

    def __repr__(self) -> str:
        return f"{type(self).__name__}(num_constraints={self.weight.shape[0]}, lower={self.lower}, upper={self.upper})"
