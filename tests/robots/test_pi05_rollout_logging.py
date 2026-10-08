"""Verify prediction/driver logging without connecting any hardware."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import torch


def test_logs_requested_and_executed_actions_separately(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        'pi05_logging_test', Path(__file__).parents[2] / 'pi05_remote_rollout.py'
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    client = object.__new__(module.LoggedClient)
    client.out = tmp_path
    client.chunk_count = client.action_count = 0
    client.first_action = None
    client.latest_action = 10
    client.robot = SimpleNamespace(_last_stable_present_pos={'gripper': 2.0})
    monkeypatch.setattr(module.RobotClient, '_aggregate_action_queues', lambda *args: None)
    monkeypatch.setattr(module.RobotClient, '_action_tensor_to_action_dict',
                        lambda self, tensor: {'gripper.pos': tensor[-1].item()})
    tensor = torch.tensor([0., 1., 2., 3., 4., 90.])
    incoming = SimpleNamespace(get_action=lambda: tensor, get_timestep=lambda: 10,
                               get_timestamp=lambda: 123.0)
    client._aggregate_action_queues([incoming])

    def fake_control(self, verbose):
        self._action_tensor_to_action_dict(tensor)
        return {'gripper.pos': 2.1}

    monkeypatch.setattr(module.RobotClient, 'control_loop_action', fake_control)
    assert client.control_loop_action() == {'gripper.pos': 2.1}
    raw = json.loads((tmp_path / 'raw_predictions.jsonl').read_text())
    executed = json.loads((tmp_path / 'actions.jsonl').read_text())
    assert raw['actions'][0]['action'][-1] == 90
    assert executed['requested_robot_action']['gripper.pos'] == 90
    assert executed['sent']['gripper.pos'] == 2.1
    assert executed['measured']['gripper'] == 2
    assert executed['before_write'] <= executed['time']
