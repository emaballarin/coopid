"""A derivative term, completing PID over Cooper's `nuPI`."""

from collections.abc import Callable
from collections.abc import Iterable

import torch
from cooper.optim import nuPI
from cooper.optim.torch_optimizers.nupi_optimizer import nuPIInitType

__all__ = ["nuPID"]


class nuPID(nuPI):
    r"""`nuPI` plus a derivative term: :math:`K_d` on the second difference of the filtered error.

    Cooper's :py:class:`~cooper.optim.nuPI` is a proportional-integral controller in **velocity
    form** -- it emits the *increment* of the control signal rather than the signal itself:

    .. math::
        \Delta \mu_t = \eta \, ( K_i e_t + K_p (\xi_t - \xi_{t-1}) ),
        \qquad \xi_t = \nu \xi_{t-1} + (1 - \nu) e_t

    In that form the textbook PID adds a **second** difference, and this class adds exactly that:

    .. math::
        \Delta \mu_t \mathrel{+}= \eta \, K_d \, (\xi_t - 2 \xi_{t-1} + \xi_{t-2})

    **Why the derivative must act on a filtered signal.** :math:`e_t` here is a per-batch
    constraint violation, i.e. a noisy estimate. The first difference of noise is noise amplified
    by :math:`\sqrt{2}`; the *second* difference amplifies it by 2. A raw D term on a stochastic
    constraint is a noise generator, which is why this one is defined on :math:`\xi` and why
    ``ema_nu = 0`` (no filtering) with a non-zero ``Kd`` emits a warning.

    **Implementation.** The PI part is delegated to :py:class:`~cooper.optim.nuPI` **verbatim**,
    and the derivative increment is applied afterwards from independent state. So ``Kd = 0`` is
    bit-identical to stock ``nuPI`` *by construction* rather than by numerical coincidence -- and
    ``tests/test_optim.py`` checks it anyway, across the init schemes and gain settings.

    The D term contributes nothing on the first two steps, because a second difference is not
    defined before two first differences exist. Its EMA is seeded at the first observed error, so
    turning ``Kd`` on cannot change where a run starts.

    Args:
        params: Parameters to optimise, as in :py:class:`~cooper.optim.nuPI`. In practice, the
            Lagrange multipliers.
        lr: Learning rate :math:`\eta`.
        weight_decay: As in :py:class:`~cooper.optim.nuPI`; applies to the PI part only.
        Kp: Proportional gain.
        Ki: Integral gain.
        Kd: Derivative gain. ``0`` (the default) disables the term entirely.
        ema_nu: EMA coefficient :math:`\nu`, shared with the proportional path.
        init_type: As in :py:class:`~cooper.optim.nuPI`.
        maximize: ``True`` for a dual optimiser, which Cooper requires.

    Raises:
        ValueError: If ``Kd`` is negative.
        NotImplementedError: If a sparse gradient is seen while ``Kd`` is non-zero.

    Warns:
        UserWarning: If ``Kd`` is non-zero while ``ema_nu`` is zero, i.e. a derivative taken on
            an unfiltered stochastic signal.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float,
        weight_decay: float = 0.0,
        Kp: float | torch.Tensor = 0.0,
        Ki: float | torch.Tensor = 1.0,
        Kd: float | torch.Tensor = 0.0,
        ema_nu: float = 0.0,
        init_type: nuPIInitType = nuPIInitType.SGD,
        maximize: bool = False,
    ) -> None:
        super().__init__(
            params,
            lr=lr,
            weight_decay=weight_decay,
            Kp=Kp,
            Ki=Ki,
            ema_nu=ema_nu,
            init_type=init_type,
            maximize=maximize,
        )
        if not isinstance(Kd, torch.Tensor):
            Kd = torch.tensor(Kd)
        if torch.any(Kd < 0.0):
            raise ValueError(f"Kd must be non-negative, got {Kd}")
        # `.ne(0)` rather than `!= 0`: a gain is exactly zero (disabled) or it is not, and
        # this matches the idiom Cooper itself uses for Kp and Ki.
        if Kd.ne(0.0).any() and ema_nu <= 0.0:
            import warnings

            warnings.warn(
                "nuPID with Kd != 0 and ema_nu == 0 differentiates an unfiltered signal twice. "
                "On a per-batch constraint violation that amplifies noise by a factor of 2; set "
                "ema_nu > 0 unless the constraint is measured exactly.",
                stacklevel=2,
            )
        self.defaults["Kd"] = Kd
        for group in self.param_groups:
            group.setdefault("Kd", Kd)

    @torch.no_grad()
    def step(self, closure: Callable | None = None) -> float | None:
        """One PI step (delegated, unchanged) followed by the derivative increment."""
        errors = {}
        for group in self.param_groups:
            if group["Kd"].eq(0.0).all():
                continue
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise NotImplementedError("nuPID does not support sparse gradients; use nuPI with Kd = 0.")
                # Snapshot before `nuPI.step` mutates the parameter: the D term is a function of
                # the ERROR sequence, and `p.grad` is that error.
                errors[p] = p.grad.clone().detach()

        loss = super().step(closure)

        for group in self.param_groups:
            Kd, ema_nu = group["Kd"], group["ema_nu"]
            if Kd.eq(0.0).all():
                continue
            alpha = group["lr"] if group["maximize"] else -group["lr"]
            for p in group["params"]:
                if p not in errors:
                    continue
                self._apply_derivative(p, errors[p], Kd=Kd, ema_nu=ema_nu, alpha=alpha)

        return loss

    def _apply_derivative(
        self, param: torch.Tensor, error: torch.Tensor, *, Kd: torch.Tensor, ema_nu: float, alpha: float
    ) -> None:
        """Fold one error into the D path and apply `Kd * (xi_t - 2 xi_{t-1} + xi_{t-2})`."""
        state = self.state[param]
        if "d_xi" not in state:
            # Seed at the first observation, so enabling Kd does not move the first step.
            state["d_xi"] = error.clone()
            state["d_diff"] = None
            return
        previous = state["d_xi"]
        current = previous.mul(ema_nu).add(error, alpha=1.0 - ema_nu)
        difference = current - previous
        if state["d_diff"] is not None:
            param.add_((difference - state["d_diff"]).mul(Kd), alpha=alpha)
        state["d_xi"] = current
        state["d_diff"] = difference
