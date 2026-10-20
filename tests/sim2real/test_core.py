"""No-hardware regression tests for data admission, calibration and paired evaluation."""

import os
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from lerobot.sim2real.calibration import fit_joint_map, projection_errors
from lerobot.sim2real.config import JointMap, fingerprint, load_config, require_gate, write_json
from lerobot.sim2real.pipeline import compare_trials, generated_attempt, generation_gates, trial_plan
from lerobot.sim2real.simulation import SuccessTracker
from lerobot.sim2real.training import batch_indices


def scene():
    root = Path(__file__).resolve().parents[2]
    cfg = load_config(root / "configs/so101_r2s2r.json")
    if os.environ.get("R2S2R_ASSET"):
        cfg["asset"] = os.environ["R2S2R_ASSET"]
    cfg["joint_map"] = {
        "sign": [1, -1, 1, -1, 1],
        "offset_rad": [0.1, -0.1, 0.2, -0.2, 0],
        "gripper_knots": [[0, 0], [40, 0.4], [100, 1.2]],
    }
    cfg["tape"]["position_m"] = [0, -0.23, 0.0095]
    cfg["holder"].update(position_m=[0.13, -0.23, 0], wall_m=0.001)
    for camera in cfg["cameras"].values():
        camera.update(K=[[600, 0, 320], [0, 600, 240], [0, 0, 1]], T_parent_cv=np.eye(4).tolist())
    return cfg


class MappingTests(unittest.TestCase):
    def test_signed_mapping_and_gripper_roundtrip(self):
        mapping = JointMap(scene()["joint_map"])
        for value in [0, 20, 40, 85, 100]:
            state = [10, -80, 90, 60, -40, value]
            np.testing.assert_allclose(mapping.to_robot(mapping.to_sim(state)), state, atol=1e-5)

    def test_decreasing_gripper_mapping(self):
        cfg = scene()["joint_map"]
        cfg["gripper_knots"] = [[0, 1.5], [100, -0.1]]
        mapping = JointMap(cfg)
        state = [0, 1, 2, 3, 4, 25]
        np.testing.assert_allclose(mapping.to_robot(mapping.to_sim(state)), state, atol=1e-5)

    def test_rejects_unmeasured_and_invalid_values(self):
        cfg = scene()["joint_map"]
        cfg["offset_rad"][0] = None
        with self.assertRaises(ValueError):
            JointMap(cfg)
        mapping = JointMap(scene()["joint_map"])
        for state in [[0, 0, 0, 0, 0, -1], [0, 0, 0, 0, 0, 101], [np.nan] * 6, [0] * 5]:
            with self.assertRaises(ValueError):
                mapping.to_sim(state)

    def test_fit_signs_offsets_from_physical_reference(self):
        truth = JointMap(scene()["joint_map"])
        poses = []
        for i in range(4):
            state = np.array([i * 12, i * -15, i * 13, i * 14, i * -16, 30])
            poses.append({"robot_state": state, "sim_q_rad": truth.to_sim(state)})
        signs, offsets = fit_joint_map(poses)
        np.testing.assert_allclose(signs, truth.sign)
        np.testing.assert_allclose(offsets, truth.offset)

    def test_heldout_projection_detects_bad_camera(self):
        k = np.array([[600.0, 0, 320], [0, 600, 240], [0, 0, 1]])
        xyz = np.array([[0.0, 0, 1], [0.1, 0.1, 1]])
        pixels = np.array([[320, 240], [380, 300]])
        np.testing.assert_allclose(projection_errors(k, np.zeros(5), np.eye(4), xyz, pixels), 0)
        bad = np.eye(4)
        bad[0, 3] = 0.1
        self.assertGreater(projection_errors(k, np.zeros(5), bad, xyz, pixels).mean(), 50)


class SuccessTests(unittest.TestCase):
    def setUp(self):
        self.cfg = scene()
        self.tracker = SuccessTracker(self.cfg, self.cfg["tape"]["position_m"])

    def update(self, pos, grip=80, fixed=False, moving=False, support=True, velocity=None):
        return self.tracker.update(
            pos, np.eye(3), np.zeros(6) if velocity is None else velocity, grip, fixed, moving, support
        )

    def prepare_lift(self):
        self.update([0, -0.23, 0.06], 0, True, True, False)
        self.update([0.13, -0.23, 0.15], 0, True, True, False)

    def test_requires_grasp_lift_release_and_one_second(self):
        self.prepare_lift()
        pos = [0.13, -0.23, 0.004 + 0.019 / 2]
        for _ in range(29):
            self.assertFalse(self.update(pos))
        self.assertTrue(self.update(pos))

    def test_empty_grasp_and_teleported_object_do_not_pass(self):
        for _ in range(40):
            self.assertFalse(self.update([0.13, -0.23, 0.0135]))

    def test_closed_gripper_rim_and_slip_do_not_pass(self):
        self.prepare_lift()
        for pos, grip, velocity, contact in [
            ([0.13, -0.23, 0.0135], 0, np.zeros(6), True),
            ([0.16, -0.23, 0.0135], 80, np.zeros(6), False),
            ([0.13, -0.23, 0.10], 80, np.zeros(6), False),
            ([0.13, -0.23, 0.0135], 80, np.ones(6), False),
        ]:
            for _ in range(40):
                self.assertFalse(self.update(pos, grip, contact, contact, velocity=velocity))

    def test_missing_holder_support_never_passes(self):
        self.prepare_lift()
        for _ in range(40):
            self.assertFalse(self.update([0.13, -0.23, 0.0135], support=False))

    def test_settling_timer_resets_after_failure(self):
        self.prepare_lift()
        for _ in range(29):
            self.update([0.13, -0.23, 0.0135])
        self.update([0, 0, 0.2])
        self.assertFalse(self.update([0.13, -0.23, 0.0135]))


class GateTests(unittest.TestCase):
    def test_generation_rejects_a_late_drop(self):
        cfg = scene()
        sim = Mock()
        sim.randomization = {}
        sim.config = cfg
        sim.site, sim.object_id, sim.free_q = 0, 0, 0
        sim.data = SimpleNamespace(
            site_xpos=np.array([[0, -0.23, 0.02]]),
            site_xmat=np.eye(3).reshape(1, 9),
            xpos=np.array([cfg["tape"]["position_m"]]),
            qpos=np.r_[np.zeros(3), 1, 0, 0, 0],
        )
        sim.model.body.return_value.id = 0
        sim.state.return_value = np.zeros(6)
        sim.ik.return_value = np.zeros(6)
        sim.tracker.lifted = True
        sim.step.side_effect = [{"success": True}] * 284 + [{"success": False}]
        result = generated_attempt(sim, np.zeros((2, 6)), 1, 1000)
        self.assertFalse(result["summary"]["success"])
        self.assertEqual(result["summary"]["reason"], "no_verified_containment")

    def test_changed_scene_and_state_replay_cannot_admit_data(self):
        cfg = scene()
        with tempfile.TemporaryDirectory() as tmp:
            calibration, replay = Path(tmp) / "cal.json", Path(tmp) / "replay.json"
            write_json(calibration, {"passed": True, "config_sha256": fingerprint(cfg)})
            changed = deepcopy(cfg)
            changed["tape"]["mass_kg"] *= 2
            with self.assertRaises(ValueError):
                require_gate(changed, calibration)
            write_json(
                replay, {"passed": True, "kind": "replay", "mode": "state", "config_sha256": fingerprint(cfg)}
            )
            with self.assertRaises(ValueError):
                generation_gates(cfg, calibration, replay)

    def test_exact_balanced_batches_and_reproducibility(self):
        batches = list(batch_indices(7, 3, 12, 123, True))
        self.assertEqual(batches, list(batch_indices(7, 3, 12, 123, True)))
        for batch in batches:
            self.assertEqual(sum(i < 7 for i in batch), 4)
            self.assertEqual(sum(7 <= i < 10 for i in batch), 4)
        with self.assertRaises(ValueError):
            list(batch_indices(7, 0, 1, 123, True))


class EvaluationTests(unittest.TestCase):
    def results(self):
        plan = trial_plan(scene(), "base", "mixed")
        rows = [
            {
                "candidate": name,
                "layout_id": i,
                "grasped": True,
                "placed": i < (10 if name == "baseline" else 16),
                "fault": False,
                "seconds": 15.0,
                "evidence": f"{name}_{i}.mp4",
            }
            for name in ("baseline", "mixed")
            for i in range(30)
        ]
        return plan, {"plan_sha256": fingerprint(plan), "results": rows}

    def test_complete_paired_acceptance(self):
        plan, results = self.results()
        report = compare_trials(plan, results)
        self.assertTrue(report["passed"])
        self.assertEqual(report["extra_successes"], 6)

    def test_missing_duplicate_or_unverified_trials_fail(self):
        for mutation in ("missing", "duplicate", "unverified", "wrong_plan"):
            plan, results = self.results()
            if mutation == "missing":
                results["results"].pop()
            elif mutation == "duplicate":
                results["results"].append(results["results"][0])
            elif mutation == "unverified":
                results["results"][0]["placed"] = None
            else:
                results["plan_sha256"] = "wrong"
            with self.assertRaises(ValueError):
                compare_trials(plan, results)

    def test_more_faults_prevent_acceptance(self):
        plan, results = self.results()
        results["results"][-1]["fault"] = True
        self.assertFalse(compare_trials(plan, results)["passed"])


if __name__ == "__main__":
    unittest.main()
