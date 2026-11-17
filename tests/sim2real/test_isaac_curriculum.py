"""Curriculum progression and prevention of synthetic prerequisite rewards."""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

EXAMPLE = Path(__file__).resolve().parents[2] / "scripts/simulation/so101_tape_leisaac"
with patch.object(sys, "path", [str(EXAMPLE), *sys.path]):
    from grpo_curriculum import CurriculumSchedule, tracker_from_prefix, validate_anchor


def source_trace():
    config = {
        "tape": {"outer_radius_m": 0.01, "height_m": 0.01},
        "holder": {"inner_radius_m": 0.05, "height_m": 0.1, "bottom_m": 0},
        "generation": {"open_percent": 25},
    }
    physics = {
        "state": np.array([[0, 0, 0, 0, 0, opening] for opening in [25, 10, 10, 10]], dtype=float),
        "tape": np.array(
            [
                [*position, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                for position in [[0, 0, 0.02], [0, 0, 0.03], [0, 0, 0.06], [0.04, 0, 0.07]]
            ]
        ),
        "holder": np.tile([0.04, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], (4, 1)),
        "forces": np.array([[0, 0, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]]),
    }
    return config, physics


def test_prerequisites_are_derived_from_physical_history_and_start_without_stable_credit():
    config, physics = source_trace()
    for stage, tick in [("approach", 0), ("lift", 1), ("transport", 2), ("release", 3)]:
        validate_anchor(config, physics, [0, 0, 0.02], stage, tick)
    tracker = tracker_from_prefix(config, physics, [0, 0, 0.02], 3)
    assert tracker.grasped and tracker.lifted and tracker.carried and tracker.stable == 0
    physics["forces"][:] = 0
    with pytest.raises(ValueError, match="history"):
        validate_anchor(config, physics, [0, 0, 0.02], "release", 3)


def test_a_completed_source_task_cannot_be_used_as_a_free_success_initialization():
    config, physics = source_trace()
    for key in physics:
        physics[key] = np.concatenate([physics[key], np.repeat(physics[key][-1:], 30, axis=0)])
    physics["state"][4:, 5] = 25
    physics["forces"][4:] = [0, 0, 1]
    with pytest.raises(ValueError, match="already satisfy"):
        tracker_from_prefix(config, physics, [0, 0, 0.02], len(physics["state"]) - 1)


def test_stage_advances_only_after_measured_suffix_success_and_failed_stage_stops():
    schedule = CurriculumSchedule()
    for _ in range(3):
        assert schedule.observe([1] * 8)["decision"] == "continue"
    assert schedule.observe([1] * 8)["decision"] == "advance"
    assert schedule.stage == "transport"
    for _ in range(9):
        assert schedule.observe([0] * 8)["decision"] == "continue"
    assert schedule.observe([0] * 8)["decision"] == "stalled"
    assert schedule.stage == "transport"


def test_late_single_success_does_not_trigger_a_stage_advance():
    schedule = CurriculumSchedule()
    for _ in range(3):
        schedule.observe([0] * 8)
    assert schedule.observe([1] + [0] * 7)["decision"] == "continue"
    assert schedule.stage == "release"


def test_open_hold_still_requires_an_action_and_supported_containment():
    config, physics = source_trace()
    physics["forces"][3] = [0, 0, 1]
    physics["state"][3, 5] = 18
    validate_anchor(config, physics, [0, 0, 0.02], "open_hold", 3)
    assert tracker_from_prefix(config, physics, [0, 0, 0.02], 3).stable == 0
    physics["state"][3, 5] = 25
    with pytest.raises(ValueError, match="require a new opening"):
        validate_anchor(config, physics, [0, 0, 0.02], "open_hold", 3)
    physics["state"][3, 5] = 18
    physics["forces"][3, 2] = 0
    with pytest.raises(ValueError, match="containment and support"):
        validate_anchor(config, physics, [0, 0, 0.02], "open_hold", 3)


def test_resume_keeps_group_outcomes_and_identical_next_stage_decisions():
    schedule = CurriculumSchedule(stages=["open_hold", "release", "full"], max_groups=6)
    for _ in range(3):
        schedule.observe([1, 1] + [0] * 6)
    state = schedule.state_dict()
    restored = CurriculumSchedule.from_state_dict(state)
    assert restored.stage == "open_hold" and len(restored.groups) == 3
    outcomes = [1, 1] + [0] * 6
    assert schedule.observe(outcomes) == restored.observe(outcomes)
    assert schedule.stage == restored.stage == "release"
    assert len(state["groups"]) == 3
    assert schedule.state_dict() == restored.state_dict()


def test_stalled_schedule_cannot_be_resumed_as_if_it_were_healthy():
    schedule = CurriculumSchedule(max_groups=6)
    for _ in range(6):
        schedule.observe([0] * 8)
    with pytest.raises(ValueError, match="revised curriculum"):
        CurriculumSchedule.from_state_dict(schedule.state_dict())
