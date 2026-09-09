import cooper
import pytest
import torch

from coopid import EMAViolation


def test_recurrence_is_exact_and_dtype_is_preserved():
    f = EMAViolation(decay=0.9)
    expected = None
    for x in (1.0, 2.0, 3.0, 4.0):
        got = f.update(torch.tensor([x], dtype=torch.float64))
        expected = x if expected is None else 0.9 * expected + 0.1 * x
        assert got.dtype == torch.float64
        assert got.item() == pytest.approx(expected, rel=1e-12)


def test_seeds_at_the_first_observation_so_smoothing_changes_no_first_step():
    """Enabling smoothing must not move where a run starts."""
    for decay in (0.0, 0.5, 0.99, 0.999):
        assert EMAViolation(decay=decay).update(torch.tensor([7.5])).item() == pytest.approx(7.5)


def test_explicit_init_reintroduces_warm_up_bias():
    f = EMAViolation(decay=0.9, init=0.0)
    assert f.update(torch.tensor([10.0])).item() == pytest.approx(0.0)


def test_zero_decay_is_exactly_a_no_op():
    f = EMAViolation(decay=0.0)
    for x in (3.0, -1.0, 8.0):
        assert f.update(torch.tensor([x])).item() == pytest.approx(x)


def test_state_is_detached_and_does_not_retain_a_graph():
    f = EMAViolation(decay=0.5)
    x = torch.tensor([2.0], requires_grad=True)
    for _ in range(3):
        out = f.update(x * 3.0)
        assert not out.requires_grad
    assert not f.average.requires_grad


def test_constraint_state_keeps_the_primal_exact_and_smooths_only_the_dual():
    f = EMAViolation(decay=0.9)
    x = torch.tensor([1.0], requires_grad=True)
    f.update(torch.tensor([0.0]))  # seed away from the next observation
    violation = x * 5.0
    state = f.constraint_state(violation)
    assert state.violation.requires_grad, "the primal must see the raw differentiable violation"
    assert torch.equal(state.violation, violation)
    assert not state.strict_violation.requires_grad
    assert state.strict_violation.item() == pytest.approx(0.5)  # 0.9*0 + 0.1*5


def test_constraint_state_forwards_kwargs_and_rejects_a_supplied_strict_violation():
    f = EMAViolation()
    state = f.constraint_state(torch.tensor([1.0]), contributes_to_primal_update=False)
    assert state.contributes_to_primal_update is False
    with pytest.raises(ValueError, match="discard the smoothing"):
        f.constraint_state(torch.tensor([1.0]), strict_violation=torch.tensor([0.0]))


def test_reset_and_state_dict_round_trip():
    f = EMAViolation(decay=0.8)
    for x in (1.0, 2.0, 3.0):
        f.update(torch.tensor([x]))
    saved = f.state_dict()
    reference = f.update(torch.tensor([9.0])).item()

    g = EMAViolation(decay=0.1)
    g.load_state_dict(saved)
    assert g.update(torch.tensor([9.0])).item() == pytest.approx(reference)

    f.reset()
    assert f.average is None
    assert f.update(torch.tensor([42.0])).item() == pytest.approx(42.0)


@pytest.mark.parametrize("decay", [-0.1, 1.0, 1.5])
def test_invalid_decay_rejected(decay):
    with pytest.raises(ValueError, match="decay must be in"):
        EMAViolation(decay=decay)


class _CMP(cooper.ConstrainedMinimizationProblem):
    def __init__(self, multiplier, smoother):
        super().__init__()
        self.smoother = smoother
        self.constraint = cooper.Constraint(
            constraint_type=cooper.ConstraintType.INEQUALITY,
            formulation_type=cooper.formulations.Lagrangian,
            multiplier=multiplier,
        )

    def compute_cmp_state(self, x, noise):
        violation = x - 1.0 + noise
        return cooper.CMPState(
            loss=(0.5 * x.square() - 9.0 * x).squeeze(),
            observed_constraints={self.constraint: self.smoother.constraint_state(violation)},
        )


def test_the_dual_sees_a_much_quieter_signal_than_the_primal():
    """The direct, mechanistic claim: `strict_violation` has far less variance than `violation`.

    This is what the filter does. It is deliberately NOT phrased as "the multiplier is quieter":
    a pure-integral dual is itself a low-pass filter, so smoothing its input changes its output
    variance very little. See NOTES.md -- GECO's averaging earns its keep on a PROPORTIONAL path
    and in gating decisions, not by damping an integrator.
    """
    torch.manual_seed(0)
    f = EMAViolation(decay=0.99)
    raw, smoothed = [], []
    for _ in range(4000):
        violation = torch.randn(1, dtype=torch.float64)
        state = f.constraint_state(violation)
        raw.append(state.violation.item())
        smoothed.append(state.strict_violation.item())

    def variance(values):
        tail = values[-2000:]
        mean = sum(tail) / len(tail)
        return sum((v - mean) ** 2 for v in tail) / len(tail)

    assert variance(smoothed) < 0.05 * variance(raw), f"raw {variance(raw):.4f} vs smoothed {variance(smoothed):.4f}"


def test_end_to_end_smoothing_still_reaches_the_analytic_equilibrium():
    """Correctness is preserved: `mu* = 8`, `x* = 1`, with the dual driven by the average."""
    torch.manual_seed(0)
    noises = [torch.randn(1) * 0.2 for _ in range(8000)]
    multiplier = cooper.multipliers.DenseMultiplier(num_constraints=1)
    cmp = _CMP(multiplier, EMAViolation(decay=0.9))
    x = torch.nn.Parameter(torch.zeros(1))
    opt = cooper.optim.SimultaneousOptimizer(
        cmp=cmp,
        primal_optimizers=torch.optim.SGD([x], lr=5e-2),
        dual_optimizers=torch.optim.SGD(multiplier.parameters(), lr=5e-2, maximize=True),
    )
    for noise in noises:
        opt.roll(compute_cmp_state_kwargs={"x": x, "noise": noise})
    assert x.item() == pytest.approx(1.0, abs=0.2)
    assert multiplier.weight.item() == pytest.approx(8.0, abs=0.5)
