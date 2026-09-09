import math

import pytest
from coopid import GatedLevel, OpenLoopLevel


def test_linear_endpoints_and_midpoint():
    s = OpenLoopLevel(start=10.0, end=2.0, steps=4)
    assert s.level == pytest.approx(10.0)
    assert [s.step() for _ in range(4)] == pytest.approx([8.0, 6.0, 4.0, 2.0])
    assert s.step() == pytest.approx(2.0), "must hold at `end` afterwards"


def test_exponential_interpolates_in_the_log():
    s = OpenLoopLevel(start=1e-1, end=1e-5, steps=4, mode="exponential")
    got = [s.level] + [s.step() for _ in range(4)]
    assert got == pytest.approx([1e-1, 1e-2, 1e-3, 1e-4, 1e-5], rel=1e-9)


def test_zero_steps_jumps_immediately():
    assert OpenLoopLevel(start=5.0, end=1.0, steps=0).level == pytest.approx(1.0)


def test_open_loop_state_dict_round_trip():
    s = OpenLoopLevel(start=0.0, end=1.0, steps=10)
    for _ in range(3):
        s.step()
    other = OpenLoopLevel(start=0.0, end=1.0, steps=10)
    other.load_state_dict(s.state_dict())
    assert other.level == pytest.approx(s.level)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"steps": -1}, "steps must be"),
        ({"steps": 4, "mode": "quadratic"}, "mode must be"),
        ({"steps": 4, "mode": "exponential", "start": -1.0}, "positive endpoints"),
    ],
)
def test_open_loop_rejects_invalid_configurations(kwargs, match):
    base = {"start": 1.0, "end": 2.0}
    with pytest.raises(ValueError, match=match):
        OpenLoopLevel(**{**base, **kwargs})


def test_gated_holds_until_patience_is_met():
    g = GatedLevel(start=1.0, end=0.0, factor=0.5, patience=3)
    assert g.step(-1.0) == pytest.approx(1.0)
    assert g.step(-1.0) == pytest.approx(1.0)
    assert g.step(-1.0) == pytest.approx(0.5), "third consecutive success tightens"
    assert g.n_tightenings == 1


def test_gated_streak_resets_on_a_violation():
    g = GatedLevel(start=1.0, end=0.0, factor=0.5, patience=3)
    g.step(-1.0)
    g.step(-1.0)
    g.step(+1.0)  # breaks the streak
    g.step(-1.0)
    g.step(-1.0)
    assert g.level == pytest.approx(1.0), "must not have tightened yet"
    assert g.step(-1.0) == pytest.approx(0.5)


def test_gated_approaches_end_geometrically_and_never_overshoots():
    g = GatedLevel(start=8.0, end=0.0, factor=0.5, patience=1)
    levels = [g.step(-1.0) for _ in range(5)]
    assert levels == pytest.approx([4.0, 2.0, 1.0, 0.5, 0.25])
    for _ in range(200):
        g.step(-1.0)
    assert 0.0 <= g.level < 1e-6


def test_gated_factor_one_jumps_straight_to_end():
    g = GatedLevel(start=5.0, end=1.0, factor=1.0, patience=1)
    assert g.step(-1.0) == pytest.approx(1.0)


def test_margin_demands_room_to_spare():
    g = GatedLevel(start=1.0, end=0.0, factor=0.5, patience=1, margin=0.5)
    assert g.step(-0.1) == pytest.approx(1.0), "satisfied, but not by the margin"
    assert g.step(-0.6) == pytest.approx(0.5), "satisfied with room to spare"


def test_gated_state_dict_round_trip():
    g = GatedLevel(start=1.0, end=0.0, patience=2)
    g.step(-1.0)
    g.step(-1.0)
    other = GatedLevel(start=1.0, end=0.0, patience=2)
    other.load_state_dict(g.state_dict())
    assert other.level == pytest.approx(g.level)
    assert other.n_tightenings == g.n_tightenings


@pytest.mark.parametrize(("kwargs", "match"), [({"factor": 0.0}, "factor"), ({"patience": 0}, "patience")])
def test_gated_rejects_invalid_configurations(kwargs, match):
    with pytest.raises(ValueError, match=match):
        GatedLevel(start=1.0, end=0.0, **kwargs)
