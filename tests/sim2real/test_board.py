"""Measured-scale regressions requiring no camera, motor or MuJoCo runtime."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from lerobot.sim2real.board import chessboard_points
from lerobot.sim2real.cli import main


class BoardTests(unittest.TestCase):
    def test_actual_print_dimensions_and_corner_order(self):
        points = chessboard_points({"inner_corners": [9, 6], "square_m": 0.017})
        self.assertEqual(points.shape, (54, 3))
        np.testing.assert_allclose(
            points[[0, 8, 45, 53]], [[0, 0, 0], [0.136, 0, 0], [0, 0.085, 0], [0.136, 0.085, 0]]
        )

    def test_scale_correction_changes_translation_not_pixels(self):
        old = chessboard_points({"inner_corners": [9, 6], "square_m": 0.018})
        new = chessboard_points({"inner_corners": [9, 6], "square_m": 0.017})
        k = np.array([[580.0, 0, 310], [0, 575, 240], [0, 0, 1]])
        r = np.array([0.3, -0.2, 0.1])
        t = np.array([-0.05, -0.02, 0.45])
        pixels = cv2.projectPoints(old, r, t, k, np.zeros(5))[0]
        corrected = cv2.projectPoints(new, r, t * 17 / 18, k, np.zeros(5))[0]
        np.testing.assert_allclose(pixels, corrected, atol=1e-9)

    def test_rejects_unknown_invalid_geometry(self):
        for size in (0, -1, np.nan, np.inf):
            with self.subTest(size=size), self.assertRaises(ValueError):
                chessboard_points({"inner_corners": [9, 6], "square_m": size})
        for corners in ([9.5, 6], [9, 2], [9], [9, np.nan]):
            with self.subTest(corners=corners), self.assertRaises(ValueError):
                chessboard_points({"inner_corners": corners, "square_m": 0.017})

    def test_world_capture_requires_scale_before_accessing_hardware(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "capture"
            argv = ["so101_r2s2r", "capture", "--board-transform", "unused.json", "--output", str(output)]
            with patch("sys.argv", argv), patch("lerobot.sim2real.cli.capture") as hardware:
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    main()
                self.assertEqual(error.exception.code, 2)
                hardware.assert_not_called()
                self.assertFalse(output.exists())

    def test_assemble_refuses_to_relabel_a_different_measured_scale(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            captures = root / "captures"
            captures.mkdir()
            row = {"board_detected": True, "split": "intrinsic", "world_points_m": None}
            (captures / "capture.json").write_text(
                json.dumps({"board": {"square_m": 0.018}, "cameras": {"front": row}}), encoding="utf-8"
            )
            joints = root / "joints.json"
            joints.write_text("[]", encoding="utf-8")
            output = root / "assembled"
            argv = [
                "so101_r2s2r",
                "assemble",
                "--square-mm",
                "17",
                "--captures",
                str(captures),
                "--joint-references",
                str(joints),
                "--output",
                str(output),
            ]
            with patch("sys.argv", argv), self.assertRaisesRegex(ValueError, "board size differs"):
                main()
            self.assertFalse((output / "observations.json").exists())


if __name__ == "__main__":
    unittest.main()
