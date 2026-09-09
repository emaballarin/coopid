"""Constraint levels derived from a measured null, rather than guessed.

A constraint ``statistic <= level`` needs a number for ``level``, and for any statistic estimated
from a finite batch there is no defensible constant. The statistic has a **floor** -- the value it
takes when the constrained property already holds exactly -- and that floor is not zero, is
typically :math:`O(1/N)`, and moves with the sample size, the dimension, the preprocessing and
the target. A level set below the floor is unreachable; a level set well above it does not
constrain. Neither failure announces itself.

The fix is to measure. Push draws from the null through the **same pipeline** the real statistic
goes through, record where it lands and how much it moves from batch to batch, and place the
level a stated number of spreads above that floor.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

__all__ = ["Calibration", "calibrate"]


@dataclass(frozen=True, slots=True)
class Calibration:
    """Where a statistic sits under the null, and the units in which to talk about it.

    Args:
        floor_mean: Mean of the statistic over draws from the null.
        per_batch_sd: Standard deviation of the statistic **across single draws**. This, not the
            standard error of the mean, is the unit a threshold and an ascent rate need.
        n_repeats: How many null draws produced the estimate.

    Raises:
        ValueError: If ``per_batch_sd`` is not positive or ``n_repeats`` is below 2.
    """

    floor_mean: float
    per_batch_sd: float
    n_repeats: int

    def __post_init__(self) -> None:
        """Reject a calibration that cannot be divided by."""
        if self.n_repeats < 2:
            raise ValueError(f"a spread needs at least 2 draws, got n_repeats={self.n_repeats}")
        if not self.per_batch_sd > 0.0:
            raise ValueError(
                f"per_batch_sd must be positive, got {self.per_batch_sd}. It is the unit every "
                "quantity here divides by; a non-positive spread usually means the statistic is "
                "deterministic under the null, or that n_repeats is far too small."
            )

    @property
    def standard_error(self) -> float:
        r"""Standard error **of the floor mean**, i.e. ``per_batch_sd / sqrt(n_repeats)``.

        Kept distinct on purpose. Confusing the two is the classic error here and it fails in
        both directions at once: using the standard error as the spread sets the level too tight
        by a factor of :math:`\\sqrt{n}` (so the constraint is unreachable) *and* the derived
        ascent rate too slow by the same factor (so the multiplier never arrives). The result is
        a run that silently ignores its own constraint.
        """
        return self.per_batch_sd / math.sqrt(self.n_repeats)

    def level(self, n_sd: float = 3.0) -> float:
        """A constraint level standing ``n_sd`` per-batch spreads above the measured floor.

        ``n_sd`` trades reachability against strictness: at ``0`` the level is the floor itself
        and a typical batch violates it half the time by construction, so the multiplier ratchets
        on noise. Three spreads is a reasonable default for a statistic that is roughly symmetric
        under the null; a heavy-tailed one wants more.
        """
        return self.floor_mean + n_sd * self.per_batch_sd

    def margin_sd(self, statistic: float) -> float:
        """How far ``statistic`` sits above the floor, in per-batch spreads.

        **Measured from the floor mean, never from a level.** A level already stands ``n_sd``
        spreads above the floor, so measuring from it under-reports the margin by exactly that
        offset.

        The denominator is the spread rather than the level because a level is a *difference of
        two quantities of similar size* and can sit arbitrarily close to zero. Dividing by it is
        a monotone rescale within one configuration -- orderings survive -- but it **inverts
        comparisons across** configurations, whose floors differ by more than their spreads do.
        The spread is the unit that transfers.
        """
        return (statistic - self.floor_mean) / self.per_batch_sd

    def n_sd_of(self, level: float) -> float:
        """The ``n_sd`` that would produce ``level``; the exact inverse of :py:meth:`level`."""
        return self.margin_sd(level)

    def dual_lr(self, gain: float = 0.1) -> float:
        """An ascent rate commensurate with the statistic, ``gain / per_batch_sd``.

        An absolute dual learning rate is the same category error as an absolute threshold. On an
        :math:`O(1/N)` statistic a violation of ``4e-3`` under a rate of ``1e-2`` moves the
        multiplier by ``4e-5`` per step, while the weight at which the constrained term competes
        with the rest of the loss may be of order ``1e3`` -- so the constraint cannot bind within
        any realistic budget, and it fails silently.

        Dividing by the spread makes the rate dimensionless: the multiplier gains ``gain`` per
        standard deviation of violation per step, which transfers across sample size, dimension,
        target and preprocessing.

        Raises:
            ValueError: If ``gain`` is not positive.
        """
        if gain <= 0.0:
            raise ValueError(f"gain must be positive, got {gain}")
        return gain / self.per_batch_sd

    @classmethod
    def from_samples(cls, values: torch.Tensor) -> Calibration:
        """Build from null draws already in hand, without re-running the pipeline."""
        flat = torch.as_tensor(values, dtype=torch.float64).flatten()
        if flat.numel() < 2:
            raise ValueError(f"a spread needs at least 2 draws, got {flat.numel()}")
        return cls(
            floor_mean=flat.mean().item(),
            per_batch_sd=flat.std(correction=1).item(),
            n_repeats=int(flat.numel()),
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(floor_mean={self.floor_mean:.6g}, "
            f"per_batch_sd={self.per_batch_sd:.6g}, n_repeats={self.n_repeats})"
        )


def calibrate(
    statistic_under_null: Callable[[torch.Generator], torch.Tensor | float],
    *,
    n_repeats: int = 200,
    seed: int = 0,
    device: torch.device | str | None = None,
) -> Calibration:
    r"""Measure a statistic's floor by evaluating it on draws from its own null.

    Args:
        statistic_under_null: Called once per repeat with a seeded
            :py:class:`torch.Generator`, and must return **one scalar** value of the statistic,
            computed on data drawn from the null **through the identical pipeline** the real
            statistic will go through. Everything that changes the statistic's distribution --
            sample size, dimension, standardisation, the number of projections, the dtype --
            must be the same here as in training, or the floor measured is a different quantity's
            floor. This callable is the whole contract, and it is the caller's because only the
            caller knows the pipeline.
        n_repeats: Number of null draws. The floor mean converges as
            :math:`1/\\sqrt{n}`; the per-batch spread, which is what the level and the rate
            actually need, converges more slowly and is the reason to be generous here.
        seed: Base seed. Repeat ``i`` uses ``seed + i``, so a calibration is reproducible and two
            calibrations with different bases are independent.
        device: Device for the generator handed to the callable.

    Returns:
        A :py:class:`Calibration` recording the floor, the per-batch spread and the repeat count.

    Raises:
        ValueError: If ``n_repeats`` is below 2, or if a call returns a non-scalar.
    """
    if n_repeats < 2:
        raise ValueError(f"a spread needs at least 2 draws, got n_repeats={n_repeats}")
    values = []
    for offset in range(n_repeats):
        generator = torch.Generator(device=device or "cpu").manual_seed(seed + offset)
        value = statistic_under_null(generator)
        tensor = torch.as_tensor(value)
        if tensor.numel() != 1:
            raise ValueError(f"statistic_under_null must return one scalar per call, got shape {tuple(tensor.shape)}")
        values.append(tensor.detach().reshape(()).to(torch.float64))
    return Calibration.from_samples(torch.stack(values))
