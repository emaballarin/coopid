import cooper
import pytest
import torch

from coopid import BoundedMultiplier


def test_upper_bound_is_enforced_by_projection():
    m = BoundedMultiplier(num_constraints=2, upper=3.0)
    m.set_constraint_type(cooper.ConstraintType.INEQUALITY)
    m.weight.data = torch.tensor([10.0, -5.0])
    m.post_step_()
    assert torch.equal(m.weight.data, torch.tensor([3.0, 0.0]))


def test_default_matches_stock_cooper_for_inequality():
    """With no explicit bounds this must be exactly relu, i.e. unchanged behaviour."""
    m = BoundedMultiplier(num_constraints=3)
    m.set_constraint_type(cooper.ConstraintType.INEQUALITY)
    m.weight.data = torch.tensor([-1.0, 0.5, 1e9])
    m.post_step_()
    assert torch.equal(m.weight.data, torch.tensor([0.0, 0.5, 1e9]))


def test_equality_constraint_is_not_floored_at_zero():
    m = BoundedMultiplier(num_constraints=1)
    m.set_constraint_type(cooper.ConstraintType.EQUALITY)
    m.weight.data = torch.tensor([-4.0])
    m.post_step_()
    assert torch.equal(m.weight.data, torch.tensor([-4.0]))


def test_explicit_lower_overrides_the_implicit_one():
    m = BoundedMultiplier(num_constraints=1, lower=-2.0, upper=2.0)
    m.set_constraint_type(cooper.ConstraintType.INEQUALITY)
    m.weight.data = torch.tensor([-9.0])
    m.post_step_()
    assert torch.equal(m.weight.data, torch.tensor([-2.0]))


def test_saturation_flag():
    m = BoundedMultiplier(num_constraints=2, upper=1.0)
    m.set_constraint_type(cooper.ConstraintType.INEQUALITY)
    m.weight.data = torch.tensor([1.0, 0.25])
    assert m.is_saturated().tolist() == [True, False]
    assert not BoundedMultiplier(num_constraints=1).is_saturated().any()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"lower": 5.0, "upper": 1.0}, "must be >="),
        ({"init": torch.tensor([9.0]), "upper": 1.0}, "above the requested upper"),
        ({"init": torch.tensor([-9.0]), "lower": 0.0}, "below the requested lower"),
    ],
)
def test_invalid_configurations_are_rejected(kwargs, match):
    num = None if "init" in kwargs else 1
    with pytest.raises(ValueError, match=match):
        BoundedMultiplier(num_constraints=num, **kwargs)


class _BoxCMP(cooper.ConstrainedMinimizationProblem):
    """`min 0.5 x^2 - g x` s.t. `x <= 1`.

    Strongly convex in the primal, so the saddle dynamics are stable and both coordinates have
    a closed form: `x* = g - mu` and, when the constraint is active, `mu* = g - 1`. That makes
    the ceiling's effect exactly predictable -- capping `mu` at `c < g - 1` must leave the
    constraint violated at `x = g - c`.
    """

    def __init__(self, multiplier, gain):
        super().__init__()
        self.gain = gain
        self.constraint = cooper.Constraint(
            constraint_type=cooper.ConstraintType.INEQUALITY,
            formulation_type=cooper.formulations.Lagrangian,
            multiplier=multiplier,
        )

    def compute_cmp_state(self, x):
        return cooper.CMPState(
            loss=(0.5 * x.square() - self.gain * x).squeeze(),
            observed_constraints={self.constraint: cooper.ConstraintState(violation=x - 1.0)},
        )


def _run(multiplier, gain=9.0, steps=3000, lr=5e-2):
    x = torch.nn.Parameter(torch.zeros(1))
    cmp = _BoxCMP(multiplier, gain)
    opt = cooper.optim.SimultaneousOptimizer(
        cmp=cmp,
        primal_optimizers=torch.optim.SGD([x], lr=lr),
        dual_optimizers=torch.optim.SGD(multiplier.parameters(), lr=lr, maximize=True),
    )
    for _ in range(steps):
        opt.roll(compute_cmp_state_kwargs={"x": x})
    return x.item(), multiplier.weight.item()


def test_end_to_end_unbounded_reaches_the_analytic_equilibrium():
    """Without a ceiling this must solve the problem: `mu* = g - 1 = 8`, `x* = 1`."""
    free = BoundedMultiplier(num_constraints=1)
    x, mu = _run(free)
    assert mu == pytest.approx(8.0, abs=1e-3)
    assert x == pytest.approx(1.0, abs=1e-3)
    assert not free.is_saturated().any()


def test_end_to_end_the_ceiling_binds_and_costs_feasibility():
    """A ceiling below the equilibrium multiplier pins it and leaves `x` at `g - cap = 7`.

    This is the failure mode the class exists to make visible rather than to prevent: the cap
    IS a weighting decision, and `is_saturated` is how a run reports that the cap, not the
    constraint, is setting the trade-off.
    """
    capped = BoundedMultiplier(num_constraints=1, upper=2.0)
    x, mu = _run(capped)
    assert mu == pytest.approx(2.0, abs=1e-6)
    assert x == pytest.approx(7.0, abs=1e-3)
    assert capped.is_saturated().all()
