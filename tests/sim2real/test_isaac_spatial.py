"""Coordinate and camera-feature regression checks independent of the Isaac SDK."""

import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

EXAMPLE = Path(__file__).resolve().parents[2] / "scripts/simulation/so101_tape_leisaac"


def load_example(name, filename):
    spec = importlib.util.spec_from_file_location(name, EXAMPLE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = load_example("_tape_spatial_common", "common.py")
with patch.dict(sys.modules, {"common": common}):
    spatial = load_example("_tape_spatial_geometry", "spatial.py")
rotation, sha256, transform = common.rotation, common.sha256, common.transform
augmented_scene, base_position = spatial.augmented_scene, spatial.base_position
episode_features, frame_features = spatial.episode_features, spatial.frame_features
pose_vector, rendered_cameras = spatial.pose_vector, spatial.rendered_cameras
rpy_rotation, state_names = spatial.rpy_rotation, spatial.state_names
table_height, validate_pose = spatial.table_height, spatial.validate_pose


class SpatialGeometryTest(unittest.TestCase):
    def setUp(self):
        self.scene = {
            "tape": {"position_m": [0.23, -0.12, 0.0095]},
            "holder": {"position_m": [0.29, 0, 0]},
            "table": {"thickness_m": 0.02},
        }
        self.cameras = {
            "cameras": {
                "front": {
                    "parent": "gripper",
                    "T_parent_cv": transform([0.01, 0.02, 0.03]).tolist(),
                    "K": np.eye(3).tolist(),
                    "distortion": [0] * 5,
                },
                "side": {
                    "parent": "world",
                    "T_parent_cv": transform([0.6, -0.2, 0.4]).tolist(),
                    "position_m": [0.6, -0.2, 0.4],
                    "K": np.eye(3).tolist(),
                    "distortion": [0] * 5,
                },
            }
        }

    def variant(self, height):
        return augmented_scene(
            self.scene,
            self.cameras,
            height_delta_m=height,
            seed=17,
            wrist_translation_m=0,
            wrist_rotation_deg=0,
            external_translation_m=0,
            external_rotation_deg=0,
        )

    def test_workspace_translation_preserves_mount_and_object_clearance(self):
        scene = self.variant(0.02)
        self.assertEqual(table_height(scene), 0.02)
        np.testing.assert_allclose(base_position(scene), [0, 0, 0.02])
        self.assertAlmostEqual(scene["tape"]["position_m"][2] - table_height(scene), 0.0095)
        self.assertAlmostEqual(scene["holder"]["position_m"][2] - table_height(scene), 0)
        self.assertEqual(self.scene["holder"]["position_m"], [0.29, 0, 0])

    def test_height_changes_external_view_not_just_metadata(self):
        low, high = self.variant(-0.02), self.variant(0.02)
        values = []
        for scene in (low, high):
            offset = base_position(scene)
            aux, extras = frame_features(scene, transform(offset), transform(offset + [0.2, 0, 0.1]))
            self.assertEqual(len(aux), 19)
            values.append(extras)
        np.testing.assert_allclose(
            values[0]["observation.camera_pose.front"], values[1]["observation.camera_pose.front"], atol=1e-7
        )
        self.assertAlmostEqual(
            float(
                values[0]["observation.camera_pose.side"][2] - values[1]["observation.camera_pose.side"][2]
            ),
            0.04,
            places=6,
        )

    def test_dynamic_wrist_pose_comes_from_recorded_gripper(self):
        scene = self.variant(0)
        arrays = {
            "robot_links": np.array(
                [
                    [[0, 0, 0, 1, 0, 0, 0], [0.2, 0, 0.1, 1, 0, 0, 0]],
                    [[0, 0, 0, 1, 0, 0, 0], [0.3, 0, 0.1, 1, 0, 0, 0]],
                ]
            )
        }
        auxiliary, extras = episode_features(scene, {"body_names": ["base", "gripper"]}, arrays)
        self.assertEqual(auxiliary.shape, (2, 19))
        self.assertEqual(len(state_names()), 25)
        np.testing.assert_allclose(extras["observation.camera_pose.front"][:, 0], [0.21, 0.31])
        np.testing.assert_array_equal(
            extras["observation.camera_pose.side"][0], extras["observation.camera_pose.side"][1]
        )

    def test_seeded_camera_changes_are_proper_rotations(self):
        original = copy.deepcopy(self.cameras)
        a = augmented_scene(self.scene, self.cameras, height_delta_m=0, seed=23)
        b = augmented_scene(self.scene, self.cameras, height_delta_m=0, seed=23)
        c = augmented_scene(self.scene, self.cameras, height_delta_m=0, seed=24)
        self.assertEqual(a, b)
        self.assertNotEqual(a["cameras"], c["cameras"])
        self.assertEqual(self.cameras, original)
        for name in ("front", "side"):
            validate_pose(a["cameras"][name]["T_parent_cv"])
            self.assertEqual(a["cameras"][name]["K"], original["cameras"][name]["K"])

    def test_quaternion_round_trip_including_half_turn(self):
        for angles in ([0, 0, 0], [np.pi, 0, 0], [0, 0, np.pi / 2], [0.7, -0.3, 1.1]):
            pose = np.eye(4)
            pose[:3, :3] = rpy_rotation(angles)
            vector = pose_vector(pose)
            np.testing.assert_allclose(rotation(vector[3:]), pose[:3, :3], atol=1e-6)

    def test_mismatched_physical_base_is_rejected(self):
        scene = self.variant(0.02)
        arrays = {"robot_links": np.array([[[0, 0, 0, 1, 0, 0, 0], [0, 0, 0.1, 1, 0, 0, 0]]])}
        with self.assertRaisesRegex(ValueError, "robot base"):
            episode_features(scene, {"body_names": ["base", "gripper"]}, arrays)

    def test_invalid_transform_is_rejected(self):
        bad = np.eye(4)
        bad[0, 0] = -1
        with self.assertRaisesRegex(ValueError, "reflection"):
            validate_pose(bad)
        with self.assertRaises(ValueError):
            augmented_scene(self.scene, self.cameras, height_delta_m=np.nan, seed=0)

    def test_legacy_camera_snapshot_allows_line_endings_but_not_changed_pose(self):
        with TemporaryDirectory() as temporary:
            run, render = Path(temporary) / "run", Path(temporary) / "render"
            run.mkdir()
            render.mkdir()
            source, snapshot = run / "camera_views.json", render / "camera_config.json"
            source.write_bytes(json.dumps(self.cameras, indent=2).replace("\n", "\r\n").encode())
            snapshot.write_text(json.dumps(self.cameras, indent=2))
            report = {"camera_config_sha256": sha256(source)}
            self.assertNotEqual(sha256(source), sha256(snapshot))
            self.assertEqual(rendered_cameras(run, render, report), self.cameras["cameras"])
            changed = copy.deepcopy(self.cameras)
            changed["cameras"]["side"]["T_parent_cv"][0][3] += 0.01
            snapshot.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "differs"):
                rendered_cameras(run, render, report)

    def test_new_camera_snapshot_hash_rejects_tampering(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            path = folder / "camera_config.json"
            path.write_text(json.dumps(self.cameras))
            report = {"camera_snapshot_sha256": sha256(path)}
            self.assertEqual(rendered_cameras(folder, folder, report), self.cameras["cameras"])
            path.write_text(json.dumps({"cameras": {}}))
            with self.assertRaisesRegex(ValueError, "changed"):
                rendered_cameras(folder, folder, report)

    def test_singleton_parquet_storage_preserves_height_values(self):
        values = spatial.stored_feature_rows([0.02, 0.0, -0.02], (1,))
        np.testing.assert_array_equal(values, np.array([[0.02], [0.0], [-0.02]], dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "shape"):
            spatial.stored_feature_rows([0.02, 0.0, -0.02], (7,))


if __name__ == "__main__":
    unittest.main()
