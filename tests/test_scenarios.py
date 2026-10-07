"""Closed-loop integration tests: the whole stack, driving, against ground truth.

These are the tests that matter most: every scenario must finish without a
collision, without leaving the road, and (where defined) reach its goal.
Run the slow full set with:  pytest -m slow
"""
import pytest

from avstack.runner import run
from avstack.sim.scenarios import SCENARIOS

FAST = ["cut_in", "stopped_vehicle"]


@pytest.mark.parametrize("name", FAST)
def test_fast_scenarios_are_safe(name):
    r = run(SCENARIOS[name](), seed=0)
    assert not r.collided and not r.off_road, r.summary()
    assert r.aeb_events == 0, r.summary()
    assert r.loc_lat_rmse < 0.2 and r.max_lat_accel < 4.5


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(set(SCENARIOS) - set(FAST)))
@pytest.mark.parametrize("seed", [0, 1])
def test_all_scenarios_are_safe(name, seed):
    r = run(SCENARIOS[name](), seed=seed)
    assert r.success and not r.collided and not r.off_road, r.summary()
    vru = r.min_clearance.get("vru")
    assert vru is None or vru > 1.0, r.summary()
