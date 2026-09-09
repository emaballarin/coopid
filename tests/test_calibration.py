import math

import cooper
import pytest
import torch
from coopid import BoundedMultiplier, Calibration, calibrate


def test_recovers_a_null_with_a_known_floor_and_spread():
    """Mean of `n` standard normals: floor 0, per-batch sd `1/sqrt(n)`."""
    n = 64
    c = calibrate(lambda g: torch.randn(n, generator=g).mean(), n_repeats=4000)
    assert c.floor_mean == pytest.approx(0.0, abs=0.01)
    assert c.per_batch_sd == pytest.approx(1.0 / math.sqrt(n), rel=0.05)
    assert c.n_repeats == 4000


def test_recovers_a_null_whose_floor_is_not_zero():
    """Mean of `n` squared normals: floor 1, per-batch sd `sqrt(2/n)`. The floor is the point."""
    n = 256
    c = calibrate(lambda g: torch.randn(n, generator=g).square().mean(), n_repeats=4000)
    assert c.floor_mean == pytest.approx(1.0, abs=0.01)
    assert c.per_batch_sd == pytest.approx(math.sqrt(2.0 / n), rel=0.05)


@pytest.mark.parametrize("n_sd", [0.0, 1.0, 3.0, 7.5])
def test_level_and_margin_sd_are_exact_inverses(n_sd):
    c = Calibration(floor_mean=0.37, per_batch_sd=0.012, n_repeats=200)
    assert c.level(n_sd) == pytest.approx(0.37 + n_sd * 0.012, rel=1e-12)
    assert c.margin_sd(c.level(n_sd)) == pytest.approx(n_sd, rel=1e-9, abs=1e-12)
    assert c.n_sd_of(c.level(n_sd)) == pytest.approx(n_sd, rel=1e-9, abs=1e-12)


def test_margin_is_measured_from_the_floor_not_from_a_level():
    """Measuring from a level would under-report every margin by exactly `n_sd`."""
    c = Calibration(floor_mean=1.0, per_batch_sd=0.1, n_repeats=100)
    assert c.margin_sd(1.0) == pytest.approx(0.0), "a statistic AT the floor has zero margin"
    assert c.margin_sd(c.level(3.0)) == pytest.approx(3.0)


def test_dual_lr_is_dimensionless_in_spreads():
    c = Calibration(floor_mean=0.0, per_batch_sd=4e-3, n_repeats=200)
    assert c.dual_lr(0.1) == pytest.approx(0.1 / 4e-3)
    # One spread of violation moves the multiplier by exactly `gain`, whatever the scale.
    assert c.dual_lr(0.1) * c.per_batch_sd == pytest.approx(0.1)
    with pytest.raises(ValueError, match="gain must be positive"):
        c.dual_lr(0.0)


def test_standard_error_is_distinct_from_the_per_batch_spread():
    """The classic error: they differ by sqrt(n_repeats), and confusing them fails both ways."""
    c = Calibration(floor_mean=0.0, per_batch_sd=0.2, n_repeats=400)
    assert c.standard_error == pytest.approx(0.2 / 20.0)
    assert c.per_batch_sd / c.standard_error == pytest.approx(math.sqrt(400))


def test_calibration_is_reproducible_and_seed_dependent():
    def statistic(g):
        return torch.randn(32, generator=g).mean()

    a = calibrate(statistic, n_repeats=200, seed=0)
    b = calibrate(statistic, n_repeats=200, seed=0)
    c = calibrate(statistic, n_repeats=200, seed=1000)
    assert a == b
    assert a != c
    assert a.per_batch_sd == pytest.approx(c.per_batch_sd, rel=0.2)


def test_from_samples_round_trip():
    values = torch.tensor([1.0, 2.0, 3.0, 4.0])
    c = Calibration.from_samples(values)
    assert c.floor_mean == pytest.approx(2.5)
    assert c.per_batch_sd == pytest.approx(values.std(correction=1).item())
    assert c.n_repeats == 4


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"floor_mean": 0.0, "per_batch_sd": 0.0, "n_repeats": 10}, "per_batch_sd must be positive"),
        ({"floor_mean": 0.0, "per_batch_sd": -1.0, "n_repeats": 10}, "per_batch_sd must be positive"),
        ({"floor_mean": 0.0, "per_batch_sd": 1.0, "n_repeats": 1}, "at least 2 draws"),
    ],
)
def test_invalid_calibrations_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Calibration(**kwargs)


def test_calibrate_rejects_bad_input():
    with pytest.raises(ValueError, match="at least 2 draws"):
        calibrate(lambda g: torch.randn(4, generator=g).mean(), n_repeats=1)
    with pytest.raises(ValueError, match="one scalar per call"):
        calibrate(lambda g: torch.randn(4, generator=g), n_repeats=5)
    with pytest.raises(ValueError, match="per_batch_sd must be positive"):
        calibrate(lambda g: torch.zeros(()), n_repeats=5)


class _CMP(cooper.ConstrainedMinimizationProblem):
    """`min 0.5 t^2 - 3t` s.t. `mean((t + eps)^2) <= level`.

    The statistic's floor is 1, not 0 -- it is a mean of squares. A level of 0 is therefore
    **unreachable by construction**, which is exactly the situation a hardcoded threshold walks
    into, and it does not announce itself: the run simply pins its multiplier and carries on.
    """

    def __init__(self, multiplier, level):
        super().__init__()
        self.level = level
        self.constraint = cooper.Constraint(
            constraint_type=cooper.ConstraintType.INEQUALITY,
            formulation_type=cooper.formulations.Lagrangian,
            multiplier=multiplier,
        )

    def compute_cmp_state(self, theta, noise):
        statistic = (theta + noise).square().mean()
        return cooper.CMPState(
            loss=(0.5 * theta.square() - 3.0 * theta).squeeze(),
            observed_constraints={self.constraint: cooper.ConstraintState(violation=statistic - self.level)},
        )


def _solve(level, dual_lr, n=256, steps=4000, cap=20.0):
    torch.manual_seed(0)
    multiplier = BoundedMultiplier(num_constraints=1, upper=cap)
    cmp = _CMP(multiplier, level)
    theta = torch.nn.Parameter(torch.zeros(1))
    opt = cooper.optim.SimultaneousOptimizer(
        cmp=cmp,
        primal_optimizers=torch.optim.SGD([theta], lr=1e-3),
        dual_optimizers=torch.optim.SGD(multiplier.parameters(), lr=dual_lr, maximize=True),
    )
    for _ in range(steps):
        opt.roll(compute_cmp_state_kwargs={"theta": theta, "noise": torch.randn(n)})
    torch.manual_seed(99)
    expected = torch.stack([(theta.detach() + torch.randn(n)).square().mean() for _ in range(200)]).mean()
    return theta.item(), multiplier.weight.item(), multiplier.is_saturated().item(), expected.item()


def test_end_to_end_a_hardcoded_level_is_unreachable_and_a_calibrated_one_is_met():
    n = 256
    cal = calibrate(lambda g: torch.randn(n, generator=g).square().mean(), n_repeats=2000)
    assert cal.floor_mean == pytest.approx(1.0, abs=0.02), "the floor is 1, which is the whole point"

    _, mu_naive, saturated_naive, stat_naive = _solve(0.0, cal.dual_lr(0.1), n=n)
    assert saturated_naive, "an unreachable level must pin the multiplier at its ceiling"
    assert mu_naive == pytest.approx(20.0)
    assert stat_naive - 0.0 > 1.0, "and the constraint is still violated, by more than the whole floor"

    level = cal.level(3.0)
    theta, mu_cal, saturated_cal, stat_cal = _solve(level, cal.dual_lr(0.1), n=n)
    assert not saturated_cal, "a calibrated level must not need the ceiling"
    assert 0.0 < mu_cal < 20.0
    assert stat_cal <= level + 1e-2, f"expected statistic {stat_cal:.4f} should meet level {level:.4f}"
    assert theta > 0.4, "and the primal is not dragged to a useless point"
