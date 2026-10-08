import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "compensated_teleop", Path(__file__).resolve().parents[2] / "compensated_teleop.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def inputs():
    return (
        {f"{n}.pos": 0.0 for n in module.JOINTS},
        {n: SimpleNamespace(range_min=0, range_max=4095) for n in module.JOINTS},
    )


def test_reference_and_no_mutation():
    action, cal = inputs()
    action["wrist_roll.pos"] = -94.81318681318682
    mapped = module.map_action(action, cal)
    assert mapped["wrist_roll.pos"] == pytest.approx(-5.054945054945055)
    assert action["wrist_roll.pos"] == -94.81318681318682
    assert mapped["shoulder_pan.pos"] == 0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 180, -300])
def test_reject_invalid_wrist(value):
    action, cal = inputs()
    action["wrist_roll.pos"] = value
    with pytest.raises(ValueError):
        module.map_action(action, cal)


def test_clamp_other_axes_and_require_keys():
    action, cal = inputs()
    action["shoulder_pan.pos"] = 200
    action["gripper.pos"] = -1
    result = module.map_action(action, cal)
    assert result["shoulder_pan.pos"] == 180
    assert result["gripper.pos"] == 0
    del action["elbow_flex.pos"]
    with pytest.raises(ValueError):
        module.map_action(action, cal)
