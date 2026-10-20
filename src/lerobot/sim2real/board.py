"""Measured chessboard geometry, independent of cameras and robot hardware."""

import numpy as np


def chessboard_points(board):
    """Return row-major inner corners using the measured square size in meters."""
    corners = np.asarray(board["inner_corners"], dtype=float)
    if (
        corners.shape != (2,)
        or not np.isfinite(corners).all()
        or np.any(corners < 3)
        or np.any(corners != np.floor(corners))
    ):
        raise ValueError("Expected two integer inner-corner counts, each >=3")
    size = float(board["square_m"])
    if not np.isfinite(size) or size <= 0:
        raise ValueError("Measured chessboard square_m must be finite and positive")
    cols, rows = corners.astype(int)
    points = np.zeros((cols * rows, 3), dtype=np.float64)
    points[:, :2] = np.mgrid[:cols, :rows].T.reshape(-1, 2) * size
    return points
