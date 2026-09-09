import itertools
import warnings

import cooper
import pytest
import torch
from cooper.optim import nuPI
from cooper.optim.torch_optimizers.nupi_optimizer import nuPIInitType

from coopid import nuPID

GRID = [
    (Kp, Ki, nu, init)
    for Kp, Ki in [(0.0, 1.0), (1.0, 1.0), (2.0, 0.5), (1.0, 0.0)]
    for nu in [0.0, 0.5, 0.9]
    for init in [nuPIInitType.SGD, nuPIInitType.ZEROS]
]


@pytest.mark.parametrize(("Kp", "Ki", "nu", "init"), GRID)
@pytest.mark.parametrize("maximize", [False, True])
def test_kd_zero_is_bit_identical_to_stock_nupi(Kp, Ki, nu, init, maximize):
    """The load-bearing safety property: opting into `nuPID` must change nothing until `Kd > 0`."""
    torch.manual_seed(0)
    errors = [torch.randn(3, dtype=torch.float64) for _ in range(50)]

    def drive(optimizer_cls, **extra):
        p = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            opt = optimizer_cls([p], lr=0.1, Kp=Kp, Ki=Ki, ema_nu=nu, init_type=init, maximize=maximize, **extra)
        trajectory = []
        for error in errors:
            p.grad = error.clone()
            opt.step()
            trajectory.append(p.detach().clone())
        return torch.stack(trajectory)

    assert torch.equal(drive(nuPI), drive(nuPID, Kd=0.0)), "Kd=0 must reproduce nuPI exactly"


@pytest.mark.parametrize("maximize", [False, True])
def test_derivative_increment_is_the_second_difference_of_the_filtered_error(maximize):
    """With Kp = Ki = 0 the only motion is the D term, so it can be checked in closed form."""
    lr, Kd, nu = 0.1, 3.0, 0.8
    errors = [
        torch.tensor([1.0], dtype=torch.float64),
        torch.tensor([4.0], dtype=torch.float64),
        torch.tensor([2.0], dtype=torch.float64),
        torch.tensor([-3.0], dtype=torch.float64),
    ]

    p = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        opt = nuPID([p], lr=lr, Kp=0.0, Ki=0.0, Kd=Kd, ema_nu=nu, maximize=maximize)

    xi = None
    previous_difference = None
    alpha = lr if maximize else -lr
    expected = 0.0
    for step, error in enumerate(errors):
        p.grad = error.clone()
        opt.step()
        if xi is None:
            xi = error.item()  # seeded at the first observation
        else:
            new_xi = nu * xi + (1.0 - nu) * error.item()
            difference = new_xi - xi
            if previous_difference is not None:
                expected += alpha * Kd * (difference - previous_difference)
            previous_difference = difference
            xi = new_xi
        if step < 2:
            assert p.item() == pytest.approx(0.0, abs=1e-15), "no D before two differences exist"
        assert p.item() == pytest.approx(expected, rel=1e-12)


def test_negative_kd_is_rejected():
    p = torch.nn.Parameter(torch.zeros(1))
    with pytest.raises(ValueError, match="Kd must be non-negative"):
        nuPID([p], lr=0.1, Kd=-1.0)


def test_unfiltered_derivative_warns():
    p = torch.nn.Parameter(torch.zeros(1))
    with pytest.warns(UserWarning, match="unfiltered signal twice"):
        nuPID([p], lr=0.1, Kd=1.0, ema_nu=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no warning when filtered, or when Kd is off
        nuPID([p], lr=0.1, Kd=1.0, ema_nu=0.9)
        nuPID([p], lr=0.1, Kd=0.0, ema_nu=0.0)


def test_sparse_gradients_are_refused_when_the_d_term_is_active():
    p = torch.nn.Parameter(torch.zeros(4))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        opt = nuPID([p], lr=0.1, Kd=1.0, ema_nu=0.9)
    indices = torch.tensor([[0, 2]])
    p.grad = torch.sparse_coo_tensor(indices, torch.tensor([1.0, 2.0]), size=(4,))
    with pytest.raises(NotImplementedError, match="sparse gradients"):
        opt.step()


class _CMP(cooper.ConstrainedMinimizationProblem):
    """`min 0.5 x^2 - 9x` s.t. `x <= 1`; equilibrium `x* = 1`, `mu* = 8`."""

    def __init__(self, multiplier):
        super().__init__()
        self.constraint = cooper.Constraint(
            constraint_type=cooper.ConstraintType.INEQUALITY,
            formulation_type=cooper.formulations.Lagrangian,
            multiplier=multiplier,
        )

    def compute_cmp_state(self, x):
        return cooper.CMPState(
            loss=(0.5 * x.square() - 9.0 * x).squeeze(),
            observed_constraints={self.constraint: cooper.ConstraintState(violation=x - 1.0)},
        )


def _overshoot(Kd, steps=800, lr=0.3):
    multiplier = cooper.multipliers.DenseMultiplier(num_constraints=1)
    cmp = _CMP(multiplier)
    x = torch.nn.Parameter(torch.zeros(1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dual = nuPID(multiplier.parameters(), lr=lr, Kp=0.0, Ki=1.0, Kd=Kd, ema_nu=0.9, maximize=True)
    opt = cooper.optim.SimultaneousOptimizer(
        cmp=cmp, primal_optimizers=torch.optim.SGD([x], lr=lr), dual_optimizers=dual
    )
    peak = -float("inf")
    for _ in range(steps):
        opt.roll(compute_cmp_state_kwargs={"x": x})
        peak = max(peak, multiplier.weight.item())
    return peak - 8.0, x.item(), multiplier.weight.item()


def test_the_derivative_term_damps_dual_overshoot_without_moving_the_equilibrium():
    """Measured, not assumed: overshoot falls ~9x from Kd=0 to Kd=20, monotonically."""
    results = {Kd: _overshoot(Kd) for Kd in (0.0, 2.0, 5.0, 10.0, 20.0)}
    overshoots = [results[Kd][0] for Kd in (0.0, 2.0, 5.0, 10.0, 20.0)]

    assert all(a > b for a, b in itertools.pairwise(overshoots)), (
        f"overshoot must fall monotonically in Kd, got {overshoots}"
    )
    assert overshoots[0] > 2.0
    assert overshoots[-1] < 0.5
    for Kd, (_, x, mu) in results.items():
        assert x == pytest.approx(1.0, abs=1e-3), f"Kd={Kd} moved the primal equilibrium"
        assert mu == pytest.approx(8.0, abs=1e-3), f"Kd={Kd} moved the dual equilibrium"
