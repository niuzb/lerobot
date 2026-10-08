import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location(
    "check_recorded_dataset", Path(__file__).resolve().parents[2] / "check_recorded_dataset.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def frames(state, action):
    return pd.DataFrame(
        {
            "episode_index": [0] * len(state),
            "frame_index": range(len(state)),
            "observation.state": list(state),
            "action": list(action),
        }
    )


def test_frozen_state_with_varying_action_is_rejected():
    state = np.zeros((90, 6))
    action = np.zeros_like(state)
    action[:, 2] = np.linspace(0, 30, len(state))
    assert not audit.audit_frames(frames(state, action), 30)["passed"]


def test_stationary_hold_and_real_motion_are_allowed():
    state = np.zeros((90, 6))
    assert audit.audit_frames(frames(state, state), 30)["passed"]
    state[:, 2] = np.linspace(0, 30, len(state))
    assert audit.audit_frames(frames(state, state + 1), 30)["passed"]


def test_nonfinite_values_are_rejected():
    state = np.zeros((90, 6))
    state[10, 3] = np.nan
    assert not audit.audit_frames(frames(state, np.zeros_like(state)), 30)["passed"]
