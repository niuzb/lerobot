"""Run an explicitly unvalidated SO-101 scene using saved measurements, without hardware."""

import argparse
import json
import time
import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path

import av
import cv2
import mujoco
import numpy as np

from lerobot.sim2real.config import JOINTS, JointMap, digest, fingerprint, load_config, write_json
from lerobot.sim2real.gripper_contact import measured_gap_m
from lerobot.sim2real.simulation import Simulator

ROOT = Path(__file__).resolve().parent


def encode_browser_preview(source, destination):
    """Encode H.264 so the saved MP4 plays in a browser or the Codex preview."""
    with (
        av.open(str(source)) as reader,
        av.open(str(destination), "w", options={"movflags": "+faststart"}) as writer,
    ):
        stream = writer.add_stream("libx264", rate=15)
        stream.width, stream.height, stream.pix_fmt = 1280, 960, "yuv420p"
        stream.options = {"crf": "20", "preset": "veryfast"}
        for i, frame in enumerate(reader.decode(video=0)):
            frame.pts = i
            frame.time_base = Fraction(1, 15)
            for packet in stream.encode(frame):
                writer.mux(packet)
        for packet in stream.encode():
            writer.mux(packet)


def build_experiment(out):
    """Use existing fits as estimates and enumerate every unmeasured scene choice."""
    base = ROOT / "configs/so101_r2s2r_20261018_mesh_holder.json"
    pose_path = ROOT / "outputs/calibration/20261018/pose_mapping_diagnostic_002/report.json"
    grip_dir = ROOT / "outputs/calibration/20261018/gripper_contact_alignment_002"
    manifest_path = ROOT / "outputs/calibration/20261017/pose_capture_combined_review/reviewed_samples.json"
    config = load_config(base)
    pose = json.loads(pose_path.read_text(encoding="utf-8"))
    grip = json.loads((grip_dir / "contact_alignment.json").read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = [row for row in manifest["samples"] if row["split"] == "fit"]
    config["joint_map"] = {
        "sign": pose["sign"],
        "offset_rad": pose["offset_rad_gauge_fixed"],
        "gripper_knots": grip["gripper_knots"],
    }
    config["cameras"]["front"]["T_parent_cv"] = pose["T_camera_mount_cv_gauge_fixed"]
    # Reconstruct the fixed side-camera pose from previously saved board corners.
    # No new images, measurements, or acceptance tests are requested or performed.
    board = np.zeros((54, 3))
    board[:, :2] = np.mgrid[:9, :6].T.reshape(-1, 2) * 0.017
    side = config["cameras"]["side"]
    pixels = [
        json.loads(Path(row["source_capture"]).read_text(encoding="utf-8"))["cameras"]["side"][
            "image_points_px"
        ]
        for row in samples
    ]
    ok, rvec, tvec = cv2.solvePnP(
        np.tile(board, (len(samples), 1)),
        np.concatenate(pixels),
        np.array(side["K"]),
        np.array(side["distortion"]),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError("Cannot construct a camera from the saved corners")
    camera_board = np.eye(4)
    camera_board[:3, :3] = cv2.Rodrigues(rvec)[0]
    camera_board[:3, 3] = tvec.ravel()
    base_board = np.array(pose["T_base_board_gauge_fixed"])
    side["T_parent_cv"] = (base_board @ np.linalg.inv(camera_board)).tolist()
    centre = (base_board @ np.array([0.068, 0.0425, 0, 1]))[:3]
    config["tape"]["position_m"] = [*centre[:2], config["tape"]["height_m"] / 2]
    config["holder"]["position_m"] = [float(centre[0] + 0.06), float(centre[1] + 0.12), 0]
    config["measurements"]["notes"] = (
        "User waived further calibration on 2026-10-18; exploratory simulation only."
    )
    config["experiment"] = {
        "status": "unvalidated_experiment",
        "user_instruction": "就按现在给的数据来做模拟，跳过测试校准",
        "calibration_checks_skipped": True,
        "calibration_passed": False,
        "assumptions": {
            "holder_mesh_pattern": "Approximate diamond pitch/strand width and rigid structure; wall 1 mm is user measured",
            "holder_mass_distribution": "60% lattice, 20% rims, 20% bottom; total remains the user's estimated 5 g",
            "tape_position": "Assumed at saved board centre; not measured object placement",
            "holder_position": "Assumed offset (+60,+120) mm from tape in base XY",
            "base_and_wrist_zero": "Gauge-fixed saved estimates, not absolute registration",
            "contact_and_actuation": "Existing model friction/servo parameters; no real dynamics fit",
            "joint_limits": "Simulation copy extended only as needed for saved fit poses",
            "background": "Plain tabletop, no photoreal reconstruction",
        },
        "input_sha256": {
            str(p): digest(p) for p in (base, pose_path, grip_dir / "contact_alignment.json", manifest_path)
        },
    }
    # Put the new XML beside the vendor meshes so the existing scene loader and
    # fingerprint both resolve the original assets. Never overwrite the source XML.
    source_asset = Path(config["asset"])
    new_asset = source_asset.with_name(f"so101.{out.name}.experimental.xml")
    if new_asset.exists():
        raise FileExistsError(new_asset)
    tree = ET.parse(grip_dir / "so101.contact_candidate.xml")
    tree.getroot().find("compiler").set("meshdir", "assets")
    mapping = JointMap(config["joint_map"])
    angles = np.array([mapping.to_sim(row["state"]) for row in samples])
    limits = {}
    for i, name in enumerate(JOINTS[:5]):
        joint = tree.find(f".//joint[@name='{name}']")
        actuator = tree.find(f"./actuator/position[@name='{name}']")
        old_joint = np.fromstring(joint.get("range"), sep=" ")
        old_ctrl = np.fromstring(actuator.get("ctrlrange"), sep=" ")
        span = [
            min(old_joint[0], old_ctrl[0], angles[:, i].min()),
            max(old_joint[1], old_ctrl[1], angles[:, i].max()),
        ]
        for item, key in ((joint, "range"), (actuator, "ctrlrange")):
            item.set(key, " ".join(map(str, span)))
        limits[name] = {"source_joint_rad": old_joint.tolist(), "experiment_rad": span}
    tree.write(new_asset, encoding="utf-8", xml_declaration=True)
    config["asset"] = str(new_asset)
    config["experiment"]["joint_limit_changes"] = limits
    config["experiment"]["original_asset_sha256"] = digest(source_asset)
    write_json(out / "scene.experimental.json", config)
    write_json(out / "source_poses.json", samples)
    return config, samples


def panel(frame, title):
    """Label a rendered scientific preview without changing its source images."""
    result = frame.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 28), (18, 24, 32), -1)
    cv2.putText(result, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 240, 245), 1, cv2.LINE_AA)
    return result


def run_preview(config, samples, out):
    """Physical actuator playback of saved reference poses, then a gripper cycle."""
    sim = Simulator(config, out / "scene.xml")
    video = cv2.VideoWriter(
        str(out / "simulation_preview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 15, (1280, 960)
    )
    if not video.isOpened():
        raise RuntimeError("Cannot open preview video writer")
    overview = mujoco.MjvCamera()
    overview.lookat[:] = [0.16, -0.03, 0.12]
    overview.distance = 0.82
    overview.azimuth = 135
    overview.elevation = -25
    events, timeline = [], []
    initial = np.array(samples[0]["state"], dtype=float)
    initial[5] = 80
    start_time = time.monotonic()
    try:
        sim.reset(initial)
        # Saved calibration poses are reference targets, not a recorded grasp demo.
        previous = initial.copy()
        for row in samples:
            goal = np.array(row["state"], dtype=float)
            goal[5] = 80
            steps = max(24, int(np.ceil(np.max(np.abs(goal[:5] - previous[:5])) / 30 * 30)))
            for tick in range(steps):
                u = (tick + 1) / steps
                u = u * u * (3 - 2 * u)
                timeline.append((previous + (goal - previous) * u, "saved pose", row["id"]))
            previous = goal
        for opening in (0, 90, 0, 80):
            goal = previous.copy()
            goal[5] = opening
            for tick in range(60):
                u = (tick + 1) / 60
                u = u * u * (3 - 2 * u)
                timeline.append((previous + (goal - previous) * u, "jaw cycle", "scripted opening target"))
            previous = goal
        for index, (action, stage, source) in enumerate(timeline):
            outcome = sim.step(action)
            state = sim.state()
            gap = measured_gap_m(sim.model, sim.data) * 1000
            jaw_contacts = 0
            for contact in sim.data.contact:
                names = [sim.model.geom(g).name or "" for g in (contact.geom1, contact.geom2)]
                jaw_contacts += int(
                    any(n.startswith("fixed_jaw") for n in names)
                    and any(n.startswith("moving_jaw") for n in names)
                )
            events.append(
                {
                    "step": index,
                    "time_s": (index + 1) / 30,
                    "stage": stage,
                    "source": source,
                    "requested_state": action.tolist(),
                    "observed_state": state.tolist(),
                    "landmark_gap_mm": gap,
                    "jaw_self_contacts": jaw_contacts,
                    **outcome,
                }
            )
            if index % 2 == 0:
                images = sim.images()
                sim.renderer.update_scene(sim.data, camera=overview)
                wide = sim.renderer.render().copy()
                info = np.full((480, 640, 3), (18, 24, 32), np.uint8)
                lines = [
                    "SO-101 / EXISTING-DATA SIMULATION",
                    "UNVALIDATED EXPERIMENT",
                    "",
                    f"Time {index / 30:.1f}s | {stage}",
                    "Control 30 Hz / physics 300 Hz",
                    f"Gripper {state[5]:.1f}% | landmark gap {gap:.1f} mm",
                    f"Jaw self-contact count: {jaw_contacts}",
                    f"Tape contacts: fixed={outcome['fixed_contact']} moving={outcome['moving_contact']}",
                    "",
                    "Tape: OD46 / ID36 / H19 mm, 3 g",
                    "Holder: ID77 / H95 / bottom4 mm, 5 g",
                    f"Wall {config['holder']['wall_m'] * 1000:g} mm | {config['holder'].get('wall_style', 'solid')}",
                    "Saved arm poses + scripted jaw cycle",
                    "No policy inference; no real robot motion",
                ]
                for line, text in enumerate(lines):
                    cv2.putText(
                        info,
                        text,
                        (18, 32 + line * 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        (230, 235, 240),
                        1,
                        cv2.LINE_AA,
                    )
                frame = np.vstack(
                    [
                        np.hstack([panel(wide, "Overview"), panel(images["side"], "External camera / side")]),
                        np.hstack([panel(images["front"], "Wrist camera / front"), info]),
                    ]
                )
                video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if index == 0:
                    cv2.imwrite(str(out / "preview.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    closeup = mujoco.MjvCamera()
                    closeup.lookat[:] = sim.data.xpos[sim.model.body("holder").id] + [
                        0,
                        0,
                        config["holder"]["height_m"] / 2,
                    ]
                    closeup.distance, closeup.azimuth, closeup.elevation = 0.24, 150, -18
                    sim.renderer.update_scene(sim.data, camera=closeup)
                    cv2.imwrite(
                        str(out / "holder_closeup.png"),
                        cv2.cvtColor(sim.renderer.render(), cv2.COLOR_RGB2BGR),
                    )
            if index % 90 == 0:
                print(json.dumps({"step": index, "total": len(timeline), "stage": stage}), flush=True)
        write_json(out / "trajectory.json", events)
        summary = {
            "completed": True,
            "calibration_passed": False,
            "mode": "experimental_actuator_preview",
            "source_kind": "11 saved calibration poses, not a grasp demonstration or policy rollout",
            "steps": len(events),
            "simulation_seconds": len(events) / 30,
            "wall_seconds": time.monotonic() - start_time,
            "config_sha256": fingerprint(config),
            "config": str(out / "scene.experimental.json"),
            "video": str(out / "simulation_preview.mp4"),
            "hardware_commands_sent": 0,
            "jaw_contact_frames": sum(r["jaw_self_contacts"] > 0 for r in events),
            "tape_contact_frames": sum(r["fixed_contact"] or r["moving_contact"] for r in events),
            "tape_lifted": any(r["lifted"] for r in events),
            "final_containment": bool(events[-1]["success"]),
            "holder_wall_m": config["holder"]["wall_m"],
            "holder_wall_style": config["holder"].get("wall_style", "solid"),
            "holder_lattice_geoms": sum(
                (sim.model.geom(i).name or "").startswith("holder_wall_strand_")
                for i in range(sim.model.ngeom)
            ),
        }
        write_json(out / "report.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
        return summary
    finally:
        video.release()
        sim.close()


def main():
    """Create a separate experiment; do not mark or overwrite calibration reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    config, samples = build_experiment(out)
    report = run_preview(config, samples, out)
    encode_browser_preview(out / "simulation_preview.mp4", out / "simulation_preview_h264.mp4")
    report["browser_video"] = str(out / "simulation_preview_h264.mp4")
    write_json(out / "report.json", report)


if __name__ == "__main__":
    main()
