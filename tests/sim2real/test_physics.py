"""MuJoCo asset tests; run on the configured Linux host when Windows DLLs are unavailable."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from test_core import scene

from lerobot.sim2real.calibration import calibrate
from lerobot.sim2real.simulation import Simulator


class PhysicsTests(unittest.TestCase):
    def test_renderer_matches_opencv_noncentral_intrinsics(self):
        import mujoco

        cfg = scene()
        cfg["joint_map"] = {"sign": [1] * 5, "offset_rad": [0] * 5, "gripper_knots": [[0, 0], [100, 1.2]]}
        transform = np.eye(4)
        transform[:3, 3] = [1, 1, 1]
        cfg["cameras"]["side"].update(
            K=[[600, 0, 300], [0, 550, 230], [0, 0, 1]], T_parent_cv=transform.tolist()
        )
        with tempfile.TemporaryDirectory() as tmp:
            sim = Simulator(cfg, Path(tmp) / "scene.xml")
            try:
                sim.reset([0, 0, 0, 0, 0, 50])
                sim.images()
                renderer = sim.renderer
                renderer.update_scene(sim.data, camera="side")
                marker = renderer.scene.geoms[renderer.scene.ngeom]
                mujoco.mjv_initGeom(
                    marker,
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([0.003, 0, 0]),
                    np.array([1.01, 1.02, 1.3]),
                    np.eye(3).ravel(),
                    np.array([0.0, 1.0, 0.0, 1.0]),
                )
                renderer.scene.ngeom += 1
                frame = renderer.render()
                mask = (
                    (frame[..., 1] > 80)
                    & (frame[..., 1] > frame[..., 0] * 1.5)
                    & (frame[..., 1] > frame[..., 2] * 1.5)
                )
                y, x = np.where(mask)
                self.assertGreater(len(x), 5)
                np.testing.assert_allclose([x.mean(), y.mean()], [320, 230 + 550 * 0.02 / 0.3], atol=2)
            finally:
                sim.close()

    def test_calibration_fit_and_independent_heldout_views(self):
        import cv2
        import mujoco

        cfg = scene()
        cfg["joint_map"] = {"sign": [1] * 5, "offset_rad": [0] * 5, "gripper_knots": [[0, 0], [100, 1.2]]}
        model = mujoco.MjModel.from_xml_path(cfg["asset"])
        data = mujoco.MjData(model)
        qidx = [model.jnt_qposadr[model.joint(n).id] for n in cfg["joint_names"]]
        observations = {"joint_reference_poses": [], "cameras": {}}
        for i in range(6):
            state = [i * 3.0] * 5 + [50.0]
            q = np.r_[np.deg2rad(state[:5]), 0.6]
            data.qpos[qidx] = q
            mujoco.mj_forward(model, data)
            fixed, moving = (model.geom(n).id for n in ("fixed_jaw_sph_tip1", "moving_jaw_sph_tip1"))
            gap = max(
                0.0,
                np.linalg.norm(data.geom_xpos[fixed] - data.geom_xpos[moving])
                - model.geom_size[fixed, 0]
                - model.geom_size[moving, 0],
            )
            observations["joint_reference_poses"].append(
                {"split": "check", "robot_state": state, "sim_q_rad": q.tolist(), "gripper_gap_m": float(gap)}
            )
        xyz = np.array(
            [
                [-0.04, -0.04, 0.3],
                [0.04, -0.04, 0.3],
                [-0.04, 0.04, 0.3],
                [0.04, 0.04, 0.3],
                [-0.03, -0.03, 0.35],
                [0.03, -0.03, 0.35],
                [-0.03, 0.03, 0.35],
                [0.03, 0.03, 0.35],
            ]
        )
        for name, camera in cfg["cameras"].items():
            views = []
            k = np.asarray(camera["K"], dtype=float)
            pixels = cv2.projectPoints(xyz, np.zeros(3), np.zeros(3), k, np.zeros(5))[0].reshape(-1, 2)
            for i in range(6):
                ref = observations["joint_reference_poses"][i]
                data.qpos[qidx] = ref["sim_q_rad"]
                mujoco.mj_forward(model, data)
                if camera["parent"] == "world":
                    world = xyz
                else:
                    parent = model.body(camera["parent"]).id
                    world = xyz @ data.xmat[parent].reshape(3, 3).T + data.xpos[parent]
                views.append(
                    {
                        "view_id": f"{name}_{i}",
                        "split": "fit" if i < 3 else "check",
                        "robot_state": ref["robot_state"],
                        "world_points_m": world.tolist(),
                        "image_points_px": pixels.tolist(),
                    }
                )
            observations["cameras"][name] = {"views": views}
        with tempfile.TemporaryDirectory() as tmp:
            report = calibrate(cfg, observations, Path(tmp))
            self.assertTrue(report["passed"])
            self.assertLess(report["cameras"]["front"]["p95_px"], 0.01)
            for row in observations["cameras"]["side"]["views"][3:]:
                row["image_points_px"] = (np.asarray(row["image_points_px"]) + 30).tolist()
            self.assertFalse(calibrate(cfg, observations, Path(tmp))["passed"])

    def test_compound_hole_and_independent_free_objects(self):
        import mujoco

        cfg = scene()
        cfg["joint_map"] = {"sign": [1] * 5, "offset_rad": [0] * 5, "gripper_knots": [[0, 0], [100, 1.2]]}
        with tempfile.TemporaryDirectory() as tmp:
            sim = Simulator(cfg, Path(tmp) / "scene.xml")
            try:
                sim.reset([0, 0, 0, 0, 0, 50])
                self.assertGreater(sim.model.body_mass[sim.model.body("holder").id], 0)
                self.assertEqual(sim.model.neq, 0)
                self.assertEqual(
                    sum(
                        (mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("tape_")
                        for i in range(sim.model.ngeom)
                    ),
                    32,
                )
                pos = sim.data.xpos[sim.object_id]
                geom = np.zeros(1, dtype=np.int32)
                distance = mujoco.mj_ray(
                    sim.model,
                    sim.data,
                    np.array([pos[0], pos[1], 0.1]),
                    np.array([0.0, 0.0, -1.0]),
                    None,
                    1,
                    -1,
                    geom,
                )
                self.assertNotIn(
                    "tape_", mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_GEOM, geom[0]) or ""
                )
                self.assertGreater(distance, 0.08)
                before = sim.data.qpos.copy()
                sim.ik(
                    sim.data.site_xpos[sim.site].copy(),
                    sim.data.site_xmat[sim.site].reshape(3, 3).copy(),
                    50,
                    sim.state(),
                )
                np.testing.assert_array_equal(before, sim.data.qpos)
            finally:
                sim.close()


if __name__ == "__main__":
    unittest.main()
