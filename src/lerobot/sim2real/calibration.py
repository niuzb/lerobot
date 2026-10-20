"""Fit mappings and cameras from measured poses and held-out point correspondences."""

from copy import deepcopy
from typing import TYPE_CHECKING

import cv2
import numpy as np

from lerobot.utils.import_utils import _mujoco_available, require_package

from .board import chessboard_points
from .config import JOINTS, JointMap, fingerprint, validate_physics, write_json
from .gripper_contact import measured_gap_m

if TYPE_CHECKING or _mujoco_available:
    import mujoco


def fit_joint_map(pairs):
    """Known physical reference poses must be expressed in the asset's joint frame."""
    real = np.asarray([p["robot_state"] for p in pairs], dtype=float)
    reference = np.asarray([p["sim_q_rad"] for p in pairs], dtype=float)
    if real.ndim != 2 or real.shape[1] != 6 or real.shape != reference.shape or len(real) < 3:
        raise ValueError("Need >=3 measured six-joint reference poses")
    if not np.isfinite(real).all() or not np.isfinite(reference).all():
        raise ValueError("Non-finite reference poses")
    signs, offsets = [], []
    for i in range(5):
        x, y = np.deg2rad(real[:, i]), reference[:, i]
        if np.ptp(real[:, i]) < 10:
            raise ValueError(f"Reference poses must exercise {JOINTS[i]} by >=10 degrees")
        sign = 1 if np.sum((x - x.mean()) * (y - y.mean())) > 0 else -1
        offset = np.median(y - sign * x)
        if np.max(np.abs(y - (sign * x + offset))) > np.deg2rad(2):
            raise ValueError(f"Reference poses disagree for {JOINTS[i]}")
        signs.append(sign)
        offsets.append(float(offset))
    return signs, offsets


def fit_intrinsics(images, board):
    """Fit lens parameters from at least ten varied measured chessboard images."""
    points = chessboard_points(board).astype(np.float32)
    cols, rows = map(int, board["inner_corners"])
    object_points, image_points = [], []
    for path in images:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None or gray.shape != (480, 640):
            raise ValueError(f"Expected 640x480 calibration image: {path}")
        ok, corners = cv2.findChessboardCornersSB(gray, (cols, rows))
        if not ok:
            raise ValueError(f"Chessboard corners not detected: {path}")
        object_points.append(points.copy())
        image_points.append(corners)
    if len(images) < 10:
        raise ValueError("Need at least ten varied chessboard views per camera")
    rms, k, distortion, _, _ = cv2.calibrateCamera(object_points, image_points, (640, 480), None, None)
    if not np.isfinite(rms) or rms > 2:
        raise ValueError(f"Intrinsic fit RMS too large: {rms}")
    return k.tolist(), distortion.ravel().tolist(), float(rms)


def projection_errors(k, distortion, transform, points, pixels):
    """T_parent_cv maps OpenCV camera coordinates into the parent frame."""
    inverse = np.linalg.inv(transform)
    camera_points = np.asarray(points) @ inverse[:3, :3].T + inverse[:3, 3]
    if np.any(camera_points[:, 2] <= 0):
        raise ValueError("Calibration places observed points behind the camera")
    rvec = cv2.Rodrigues(inverse[:3, :3])[0]
    projected = cv2.projectPoints(np.asarray(points, dtype=float), rvec, inverse[:3, 3], k, distortion)[0]
    return np.linalg.norm(projected.reshape(-1, 2) - np.asarray(pixels), axis=1)


def calibrate(config, observations, out):
    """Fit on declared fit views and accept only using independent check views."""
    require_package("mujoco", extra="sim2real")
    candidate = deepcopy(config)
    fit_pairs = [p for p in observations.get("joint_reference_poses", []) if p["split"] == "fit"]
    if fit_pairs:
        signs, offsets = fit_joint_map(fit_pairs)
        candidate["joint_map"]["sign"] = signs
        candidate["joint_map"]["offset_rad"] = offsets
    mapping = JointMap(candidate["joint_map"])
    model = mujoco.MjModel.from_xml_path(candidate["asset"])
    data = mujoco.MjData(model)
    indices = [model.jnt_qposadr[model.joint(name).id] for name in JOINTS]
    report = {"passed": True, "cameras": {}, "joint_checks": [], "gripper_gap_checks_m": []}
    checks = [p for p in observations.get("joint_reference_poses", []) if p["split"] == "check"]
    if len(checks) < 3:
        raise ValueError("Provide >=3 held-out joint reference poses")
    for p in checks:
        error = np.rad2deg(np.abs(mapping.to_sim(p["robot_state"]) - p["sim_q_rad"]))
        report["joint_checks"].append(error.tolist())
        report["passed"] &= bool(np.max(error) <= 2)
        gap = p.get("gripper_gap_m")
        if gap is None or not np.isfinite(gap) or gap < 0:
            raise ValueError("Each check pose needs a measured gripper_gap_m between jaw tip contact points")
        data.qpos[indices] = mapping.to_sim(p["robot_state"])
        mujoco.mj_forward(model, data)
        predicted_gap = measured_gap_m(model, data)
        gap_error = float(abs(predicted_gap - gap))
        report["gripper_gap_checks_m"].append(gap_error)
        report["passed"] &= gap_error <= 0.003
    for name, camera in candidate["cameras"].items():
        inputs = observations["cameras"][name]
        if inputs.get("intrinsic_images"):
            camera["K"], camera["distortion"], rms = fit_intrinsics(
                inputs["intrinsic_images"], inputs["board"]
            )
        else:
            rms = None
        k = np.asarray(camera["K"], dtype=float)
        distortion = np.asarray(camera["distortion"], dtype=float)
        groups = {"fit": [], "check": []}
        view_ids = set()
        for row in inputs["views"]:
            if not row.get("view_id") or row["view_id"] in view_ids:
                raise ValueError("Every fit/check view needs a unique view_id")
            view_ids.add(row["view_id"])
            if row["split"] not in groups:
                raise ValueError("Each calibration view must declare fit or check")
            data.qpos[indices] = mapping.to_sim(row["robot_state"])
            mujoco.mj_forward(model, data)
            world = np.asarray(row["world_points_m"], dtype=float)
            if camera["parent"] == "world":
                points = world
            else:
                body = model.body(camera["parent"]).id
                points = (world - data.xpos[body]) @ data.xmat[body].reshape(3, 3)
            pixels = np.asarray(row["image_points_px"], dtype=float)
            if points.ndim != 2 or points.shape[1] != 3 or pixels.shape != (len(points), 2):
                raise ValueError("Expected matching world_points_m[N,3] and image_points_px[N,2]")
            if not np.isfinite(points).all() or not np.isfinite(pixels).all():
                raise ValueError("Non-finite point correspondences")
            groups[row["split"]].append((points, pixels))
        if min(len(groups["fit"]), len(groups["check"])) < 3:
            raise ValueError(f"{name} requires >=3 fit views and >=3 independent check views")
        fit_xyz = np.concatenate([x[0] for x in groups["fit"]])
        fit_uv = np.concatenate([x[1] for x in groups["fit"]])
        ok, rvec, tvec = cv2.solvePnP(fit_xyz, fit_uv, k, distortion, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            raise ValueError(f"Camera fit failed: {name}")
        inverse = np.eye(4)
        inverse[:3, :3] = cv2.Rodrigues(rvec)[0]
        inverse[:3, 3] = tvec.ravel()
        transform = np.linalg.inv(inverse)
        camera["T_parent_cv"] = transform.tolist()
        errors = np.concatenate(
            [projection_errors(k, distortion, transform, x, y) for x, y in groups["check"]]
        )
        metrics = {
            "median_px": float(np.median(errors)),
            "p95_px": float(np.quantile(errors, 0.95)),
            "intrinsic_rms_px": rms,
            "check_views": len(groups["check"]),
        }
        metrics["passed"] = metrics["median_px"] <= 8 and metrics["p95_px"] <= 20
        report["cameras"][name] = metrics
        report["passed"] &= metrics["passed"]
    validate_physics(candidate)
    report["config_sha256"] = fingerprint(candidate)
    report["observations_sha256"] = fingerprint(observations)
    write_json(out / "scene.calibrated.json", candidate)
    write_json(out / "calibration_report.json", report)
    return report
