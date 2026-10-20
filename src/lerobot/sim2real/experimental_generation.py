"""Explicitly unvalidated scene experiments; physical success checks remain mandatory."""

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from lerobot.utils.import_utils import _mujoco_available, _scipy_available, require_package

from .config import TASK, TRAIN, digest, fingerprint, load_config, write_json
from .data import SourceData, new_dataset
from .simulation import Simulator

if TYPE_CHECKING or _mujoco_available:
    import mujoco
if TYPE_CHECKING or _scipy_available:
    from scipy.optimize import least_squares


class RingWallPlanner:
    """Use real starting posture and replan a pinch across the tape's annular wall."""

    def __init__(self, sim, initial):
        """Keep the source posture and resolve the fixed fingertip geometry."""
        require_package("scipy", extra="sim2real")
        expected = sim.config.get("simulation_parameters", {}).get("mujoco_version")
        if expected is None or mujoco.__version__ != expected:
            raise ValueError("Use the pinned MuJoCo version for this physical controller")
        self.sim = sim
        self.initial = np.asarray(initial, dtype=float).copy()
        self.initial[5] = sim.config["generation"]["open_percent"]
        self.body = sim.model.body("gripper").id
        self.tip = sim.model.geom("fixed_jaw_sph_tip1").id
        self.scratch = mujoco.MjData(sim.model)

    def ik(self, target, initial, grip):
        """Solve on scratch state, allowing pitch tilt near the arm's reach limit."""
        sim, scratch = self.sim, self.scratch
        q = sim.mapping.to_sim(initial)
        scratch.qpos[:] = sim.data.qpos
        scratch.qpos[sim.qidx[5]] = sim.mapping.to_sim(np.r_[initial[:5], grip])[5]

        def residual(x):
            scratch.qpos[sim.qidx[:5]] = x
            mujoco.mj_forward(sim.model, scratch)
            return np.r_[
                (scratch.geom_xpos[self.tip] - target) * 100,
                (scratch.xmat[self.body].reshape(3, 3)[:, 2] - [0, 0, 1]) * 2,
                x[4] - np.deg2rad(self.initial[4]),
            ]

        fit = least_squares(
            residual,
            np.clip(q[:5], sim.bounds[:5, 0] + 1e-7, sim.bounds[:5, 1] - 1e-7),
            bounds=(sim.bounds[:5, 0], sim.bounds[:5, 1]),
            max_nfev=120,
        )
        residual(fit.x)
        if np.linalg.norm(scratch.geom_xpos[self.tip] - target) > 0.002:
            raise RuntimeError("IK position error exceeds 2 mm")
        return sim.mapping.to_robot(scratch.qpos[sim.qidx])

    def attempt(self, seed):
        """All motion after reset uses actuators; retain pre-action states for video export."""
        sim = self.sim
        sim.reset(self.initial, seed, randomize=True)
        tape = sim.data.xpos[sim.object_id].copy()
        direction = tape[:2] / np.linalg.norm(tape[:2])
        params = sim.config["generation"]
        fixed = np.r_[tape[:2] + direction * params["inner_tip_radius_m"], params["tip_height_m"]]
        approach = fixed.copy()
        approach[2] = 0.080
        lift = fixed.copy()
        lift[2] = 0.145
        holder = sim.data.xpos[sim.model.body("holder").id].copy()
        transport = np.r_[holder[:2] + [params["inner_tip_radius_m"], 0], 0.145]
        opened, closed = params["open_percent"], params["closed_percent"]
        stages = [
            ("approach", approach, opened, 60),
            ("descend", fixed, opened, 90),
            ("close", fixed, closed, 60),
            ("lift", lift, closed, 120),
            ("transport", transport, closed, 150),
            ("open", transport, opened, 60),
            ("settle", transport, opened, 150),
        ]
        rows, qpos, qvel = [], [], []
        reason = "incomplete"
        try:
            for stage, target, grip, steps in stages:
                # Soft-limit overshoot in feedback is retained in observations. Only
                # the interpolation's command start is projected into legal bounds.
                start = sim.mapping.to_robot(
                    np.clip(sim.mapping.to_sim(sim.state()), sim.bounds[:, 0], sim.bounds[:, 1])
                )
                goal = self.ik(target, start, grip)
                for tick in range(steps):
                    u = (tick + 1) / steps
                    action = start + u * u * (3 - 2 * u) * (goal - start)
                    pre = sim.state()
                    pre_qpos, pre_qvel = sim.data.qpos.copy(), sim.data.qvel.copy()
                    outcome = sim.step(action)
                    qpos.append(pre_qpos)
                    qvel.append(pre_qvel)
                    rows.append(
                        {"state": pre.tolist(), "sent_action": action.tolist(), "stage": stage, **outcome}
                    )
                if stage in ("lift", "transport") and (
                    outcome["object_position_m"][2] < 0.08
                    or not (outcome["fixed_contact"] or outcome["moving_contact"])
                ):
                    raise RuntimeError(f"Grasp lost at {stage}")
            reason = "verified_containment" if rows[-1]["success"] else "no_verified_containment"
        except (ValueError, RuntimeError) as exc:
            reason = str(exc)
        success = bool(reason == "verified_containment")
        return {
            "rows": rows,
            "qpos": np.asarray(qpos),
            "qvel": np.asarray(qvel),
            "summary": {
                "seed": seed,
                "success": success,
                "reason": reason,
                "frames": len(rows),
                "grasped": bool(sim.tracker.grasped),
                "lifted": bool(sim.tracker.lifted),
                "carried": bool(sim.tracker.carried),
                "stable_control_ticks": sim.tracker.stable,
                "randomization": sim.randomization,
            },
        }


def render_attempt(sim, trajectory, video_path, dataset=None):
    """Render stored physical states. These assignments never determine success labels."""
    randomization = trajectory["summary"]["randomization"]
    sim.model.light_diffuse[:] = randomization["light_diffuse"]
    stride = 10 if dataset is None else 2
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30 / stride, (1280, 480))
    if not writer.isOpened():
        raise RuntimeError("Cannot open evidence video")
    try:
        for i, (row, qpos, qvel) in enumerate(
            zip(trajectory["rows"], trajectory["qpos"], trajectory["qvel"], strict=True)
        ):
            if dataset is None and i % stride:
                continue
            sim.data.qpos[:], sim.data.qvel[:] = qpos, qvel
            mujoco.mj_forward(sim.model, sim.data)
            if not np.allclose(sim.state(), row["state"], atol=1e-5):
                raise ValueError("Rendered state differs from pre-action observation")
            frames = sim.images()
            if dataset is not None:
                dataset.add_frame(
                    {
                        "observation.state": np.asarray(row["state"], dtype=np.float32),
                        "action": np.asarray(row["sent_action"], dtype=np.float32),
                        "observation.images.front": frames["front"],
                        "observation.images.side": frames["side"],
                        "task": TASK,
                    }
                )
            if i % stride == 0:
                frame = np.concatenate([frames["front"], frames["side"]], axis=1)
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if dataset is not None:
            dataset.save_episode()
    finally:
        writer.release()


def generate_experimental(config, out, count, seed, physics_only=False, pilot=None):
    """User-waived calibration is explicit; success, provenance and data checks are enforced."""
    if not config.get("experiment", {}).get("calibration_checks_skipped"):
        raise ValueError("Use the calibrated pipeline unless the user explicitly waived calibration")
    if count not in (20, 200) or seed < 1000 or seed + count * 10 >= 100000:
        raise ValueError("Require 20/200 successes and generation seeds in [1000, 100000)")
    scene_hash = fingerprint(config)
    prior_seeds = set()
    if count == 200:
        if pilot is None:
            raise ValueError("200-success generation requires a matching 20-success pilot")
        previous = json.loads(Path(pilot).read_text())
        if not previous["passed"] or previous["accepted"] != 20 or previous["config_sha256"] != scene_hash:
            raise ValueError("Pilot failed or belongs to another scene")
        prior_seeds = {a["seed"] for a in previous["attempts"]}
    if prior_seeds.intersection(range(seed, seed + count * 10)):
        raise ValueError("Pilot and production seed ranges overlap")
    source = SourceData(config)
    episode = config["generation"]["template_episode"]
    if episode not in TRAIN:
        raise ValueError("Validation episodes cannot supply starting postures")
    states, _ = source.episode(episode)
    initial = states[config["generation"]["start_frame"]]
    write_json(out / "scene.experimental.json", config)
    sim = Simulator(config, out / "scene.xml")
    planner = RingWallPlanner(sim, initial)
    report = {
        "kind": "generation",
        "scene_status": "unvalidated_experiment",
        "calibration_passed": False,
        "config_sha256": scene_hash,
        "source": source.manifest(),
        "template_episode": episode,
        "template_start_frame": config["generation"]["start_frame"],
        "controller_source": "IK ring-wall pinch informed by training episode 1; not a replayed demonstration",
        "requested": count,
        "accepted": 0,
        "attempts": [],
        "passed": False,
        "physics_only": physics_only,
        "exported": 0,
    }
    dataset = None
    try:
        for attempt in range(count * 10):
            begin = time.monotonic()
            trajectory = planner.attempt(seed + attempt)
            prefix = out / f"attempt_{attempt:04d}"
            write_json(prefix.with_suffix(".json"), {k: trajectory[k] for k in ("summary", "rows")})
            np.savez_compressed(prefix.with_suffix(".npz"), qpos=trajectory["qpos"], qvel=trajectory["qvel"])
            summary = {
                **trajectory["summary"],
                "attempt": attempt,
                "json_sha256": digest(prefix.with_suffix(".json")),
                "npz_sha256": digest(prefix.with_suffix(".npz")),
            }
            report["attempts"].append(summary)
            if summary["success"]:
                report["accepted"] += 1
            print(
                json.dumps(
                    {
                        "accepted": report["accepted"],
                        "attempt": attempt,
                        "reason": summary["reason"],
                        "seconds": time.monotonic() - begin,
                    }
                ),
                flush=True,
            )
            write_json(out / "generation_manifest.json", report)
            if not physics_only:
                if summary["success"] and dataset is None:
                    dataset = new_dataset(out / "dataset", "local/so101_r2s2r_sim")
                render_attempt(
                    sim, trajectory, prefix.with_suffix(".mp4"), dataset if summary["success"] else None
                )
                if summary["success"]:
                    report["exported"] += 1
                    write_json(out / "generation_manifest.json", report)
            if report["accepted"] == count:
                break
    finally:
        if dataset is not None:
            dataset.finalize()
        sim.close()
    report["passed"] = report["accepted"] == count and (physics_only or report["exported"] == count)
    if dataset is not None:
        report["dataset_sha256"] = {
            str(p.relative_to(out / "dataset")): digest(p)
            for p in sorted((out / "dataset").rglob("*"))
            if p.is_file()
        }
    write_json(out / "generation_manifest.json", report)
    return report


def export_physics_run(config, source, out):
    """Export verified physical recordings without executing or relabeling motion."""
    report = json.loads((source / "generation_manifest.json").read_text())
    if not report["passed"] or not report["physics_only"] or report["config_sha256"] != fingerprint(config):
        raise ValueError("Need a complete physical run for the exact scene")
    accepted = [row for row in report["attempts"] if row["success"]]
    if len(accepted) != report["requested"] or any(row["stable_control_ticks"] < 30 for row in accepted):
        raise ValueError("Incomplete success evidence")
    report.update(passed=False, physics_only=False, exported=0, physics_source=str(source.resolve()))
    write_json(out / "scene.experimental.json", config)
    sim = Simulator(config, out / "scene.xml")
    dataset = new_dataset(out / "dataset", "local/so101_r2s2r_sim")
    try:
        # Successful data first. Full failed states remain in the immutable physics
        # directory; their lower-frame-rate evidence previews are rendered afterward.
        ordered = sorted(report["attempts"], key=lambda row: not row["success"])
        for summary in ordered:
            prefix = source / f"attempt_{summary['attempt']:04d}"
            for extension in ("json", "npz"):
                if digest(prefix.with_suffix(f".{extension}")) != summary[f"{extension}_sha256"]:
                    raise ValueError(f"Physical recording changed: {prefix}")
                shutil.copy2(
                    prefix.with_suffix(f".{extension}"), out / prefix.with_suffix(f".{extension}").name
                )
            trajectory = json.loads(prefix.with_suffix(".json").read_text())
            with np.load(prefix.with_suffix(".npz")) as arrays:
                trajectory.update(qpos=arrays["qpos"], qvel=arrays["qvel"])
            if summary["success"] != bool(trajectory["summary"]["success"]):
                raise ValueError("Physical success labels disagree")
            render_attempt(
                sim,
                trajectory,
                out / prefix.with_suffix(".mp4").name,
                dataset if summary["success"] else None,
            )
            if summary["success"]:
                report["exported"] += 1
            write_json(out / "generation_manifest.json", report)
            print(json.dumps({"exported": report["exported"], "attempt": summary["attempt"]}), flush=True)
    finally:
        dataset.finalize()
        sim.close()
    report["dataset_sha256"] = {
        str(p.relative_to(out / "dataset")): digest(p)
        for p in sorted((out / "dataset").rglob("*"))
        if p.is_file()
    }
    report["passed"] = report["exported"] == report["requested"]
    write_json(out / "generation_manifest.json", report)
    return report


def main():
    """Run the explicitly requested experimental stage into a fresh directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, choices=[20, 200], default=20)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--physics-only", action="store_true")
    parser.add_argument("--export-from", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.export_from:
        report = export_physics_run(load_config(args.config), args.export_from, args.output)
    else:
        report = generate_experimental(
            load_config(args.config), args.output, args.count, args.seed, args.physics_only, args.pilot
        )
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
