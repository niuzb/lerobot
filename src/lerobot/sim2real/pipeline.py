"""Explicit stage gates for replay, physical demonstration generation and evaluation."""

import json
from pathlib import Path

import numpy as np

from .config import TRAIN, digest, fingerprint, require_gate, write_json
from .data import SourceData, export_episode, new_dataset, save_attempt
from .simulation import Simulator


def episode_setup(config, episode):
    """Use the measured source layout rather than assuming identical resets."""
    from copy import deepcopy

    setup = config.get("episode_setups", {}).get(str(episode))
    if setup is None:
        raise ValueError(f"Annotate measured source layout in episode_setups[{episode}] before replay")
    result = deepcopy(config)
    for obj in ("tape", "holder"):
        result[obj]["position_m"] = setup[f"{obj}_position_m"]
    return result


def replay(config, out, episodes, mode):
    """Report kinematic playback and physical action replay as different modes."""
    source = SourceData(config)
    report = {
        "kind": "replay",
        "mode": mode,
        "config_sha256": fingerprint(config),
        "source": source.manifest(),
        "episodes": [],
        "passed": False,
    }
    for episode in episodes:
        states, actions = source.episode(episode)
        scene = episode_setup(config, episode)
        sim = Simulator(scene, out / f"scene_{episode:03d}.xml")
        rows, frames, errors = [], [], []
        try:
            sim.reset(states[0])
            for step, (state, action) in enumerate(zip(states, actions, strict=True)):
                if mode == "state":
                    sim.set_pose(state)
                observed = sim.state()
                frames.append(sim.images())
                row = {
                    "step": step,
                    "state": observed.tolist(),
                    "source_state": state.tolist(),
                    "requested_action": action.tolist(),
                    "sent_action": action.tolist(),
                }
                if mode == "action":
                    row.update(sim.step(action))
                    if step + 1 < len(states):
                        errors.append(sim.state() - states[step + 1])
                rows.append(row)
            metrics = {"episode": episode, "frames": len(rows), "mode": mode}
            if errors:
                errors = np.asarray(errors)
                metrics.update(
                    body_rmse_deg=float(np.sqrt(np.mean(errors[:, :5] ** 2))),
                    gripper_rmse_percent=float(np.sqrt(np.mean(errors[:, 5] ** 2))),
                    lifted=any(r["lifted"] for r in rows),
                    placed=bool(rows[-1]["success"]),
                )
                metrics["passed"] = (
                    metrics["body_rmse_deg"] <= 5
                    and metrics["gripper_rmse_percent"] <= 10
                    and metrics["placed"]
                )
            else:
                metrics["passed"] = False  # Setting qpos does not validate physics.
            save_attempt(out, episode, {"rows": rows, "frames": frames, "summary": metrics})
            report["episodes"].append(metrics)
            write_json(out / "replay_report.json", report)
        finally:
            sim.close()
    report["passed"] = (
        mode == "action"
        and len(episodes) >= 3
        and set(episodes) <= set(TRAIN)
        and all(r["passed"] for r in report["episodes"])
    )
    write_json(out / "replay_report.json", report)
    return report


def generation_gates(config, calibration, replay_report, pilot=None):
    """Require camera, joint and dynamic replay acceptance before collecting labels."""
    require_gate(config, calibration)
    report = json.loads(Path(replay_report).read_text())
    if (
        report.get("kind") != "replay"
        or report.get("mode") != "action"
        or not report.get("passed")
        or report.get("config_sha256") != fingerprint(config)
    ):
        raise ValueError("Need a passed, scene-matched dynamic replay report covering >=3 training episodes")
    if pilot:
        pilot_report = json.loads(Path(pilot).read_text())
        if (
            pilot_report.get("kind") != "generation"
            or not pilot_report.get("passed")
            or pilot_report.get("accepted") != 20
            or pilot_report.get("config_sha256") != fingerprint(config)
        ):
            raise ValueError("Production generation requires a passed 20-episode pilot for this exact scene")


def template_pose(config, source):
    """Keep orientation and object-relative grasp offset from a training demonstration."""
    episode = config["generation"]["template_episode"]
    if episode not in TRAIN:
        raise ValueError("Cannot use validation episodes as generation templates")
    states, _ = source.episode(episode)
    frame = config["generation"]["grasp_frame"]
    if frame is None or not 0 <= frame < len(states):
        raise ValueError("Annotate generation.grasp_frame from the real successful grasp")
    scene = episode_setup(config, episode)
    return scene, states, frame


def generated_attempt(sim, states, grasp_frame, seed):
    """Replan an object-relative grasp and execute only through physical actuators."""
    # Capture the reference grasp in object coordinates before randomizing the scene.
    sim.set_pose(states[grasp_frame])
    reference = sim.data.site_xpos[sim.site].copy()
    rotation = sim.data.site_xmat[sim.site].reshape(3, 3).copy()
    offset = reference - np.asarray(sim.config["tape"]["position_m"])
    sim.reset(states[0], seed, randomize=True)
    tape = sim.data.xpos[sim.object_id].copy()
    yaw_quat = sim.data.qpos[sim.free_q + 3 : sim.free_q + 7]
    yaw = 2 * np.arctan2(yaw_quat[3], yaw_quat[0])
    rotate_z = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    grasp, orientation = tape + rotate_z @ offset, rotate_z @ rotation
    holder = sim.data.xpos[sim.model.body("holder").id].copy()
    clearance = max(sim.config["holder"]["height_m"] + sim.config["tape"]["height_m"] + 0.04, 0.12)
    carry = grasp.copy()
    carry[2] = clearance
    above_holder = holder + np.array([0, 0, clearance]) + rotate_z @ np.r_[offset[:2], 0]
    release = above_holder.copy()
    release[2] = holder[2] + sim.config["holder"]["height_m"] + sim.config["tape"]["height_m"] / 2 + 0.015
    opened = sim.config["generation"]["open_percent"]
    closed = sim.config["generation"]["closed_percent"]
    stages = [
        ("approach", grasp + [0, 0, 0.04], opened),
        ("descend", grasp, opened),
        ("close", grasp, closed),
        ("lift", carry, closed),
        ("transport", above_holder, closed),
        ("release_height", release, closed),
        ("open", release, opened),
        ("withdraw", above_holder, opened),
    ]
    rows, frames = [], []
    success = False
    try:
        for stage, target, grip in stages:
            start = sim.state()
            goal = sim.ik(target, orientation, grip, start)
            # At least one second per stage, and bound body target velocity.
            duration = max(1.0, 1.5 * float(np.max(np.abs(goal[:5] - start[:5]))) / 20)
            steps = int(np.ceil(duration * 30))
            if len(rows) + steps > 900:
                raise RuntimeError("Trajectory exceeds 30 second budget")
            for i in range(steps):
                u = (i + 1) / steps
                u = u * u * (3 - 2 * u)
                action = start + (goal - start) * u
                pre_state = sim.state()
                frames.append(sim.images())
                outcome = sim.step(action)
                rows.append(
                    {
                        "state": pre_state.tolist(),
                        "sent_action": action.tolist(),
                        "post_state": sim.state().tolist(),
                        "stage": stage,
                        **outcome,
                    }
                )
                success = outcome["success"]
            if stage == "lift" and not sim.tracker.lifted:
                raise RuntimeError("No physical grasp/lift; refusing transport labels")
        for _ in range(45):
            pre_state = sim.state()
            frames.append(sim.images())
            outcome = sim.step(goal)
            rows.append(
                {
                    "state": pre_state.tolist(),
                    "sent_action": goal.tolist(),
                    "post_state": sim.state().tolist(),
                    "stage": "settle",
                    **outcome,
                }
            )
            success = outcome["success"]
        reason = "verified_containment" if success else "no_verified_containment"
    except (ValueError, RuntimeError) as exc:
        reason = str(exc)
        success = False
    return {
        "rows": rows,
        "frames": frames,
        "summary": {
            "seed": seed,
            "success": bool(success),
            "reason": reason,
            "randomization": sim.randomization,
        },
    }


def generate(config, out, count, calibration, replay_report, pilot, seed):
    """Export verified successes while retaining all failed attempts separately."""
    if count not in (20, 200):
        raise ValueError("Use a 20-success pilot, followed by a 200-success production dataset")
    if seed < 0 or seed + count * 10 > 100000:
        raise ValueError("Keep generation seeds in 0..99999; evaluation reserves seeds >=100000")
    if count == 200 and pilot is None:
        raise ValueError("Specify --pilot manifest from a completed 20-episode pilot")
    generation_gates(config, calibration, replay_report, pilot)
    source = SourceData(config)
    scene, states, grasp_frame = template_pose(config, source)
    sim = Simulator(scene, out / "scene.xml")
    dataset = None
    report = {
        "kind": "generation",
        "config_sha256": fingerprint(config),
        "source": source.manifest(),
        "template_episode": config["generation"]["template_episode"],
        "requested": count,
        "accepted": 0,
        "attempts": [],
        "passed": False,
        "seed": seed,
    }
    try:
        for attempt in range(count * 10):
            attempt_seed = seed + attempt
            if pilot:
                previous = json.loads(Path(pilot).read_text())
                if attempt_seed in {a["seed"] for a in previous["attempts"]}:
                    raise ValueError("Production seeds must be disjoint from pilot seeds")
            trajectory = generated_attempt(sim, states, grasp_frame, attempt_seed)
            save_attempt(out, attempt, trajectory)
            report["attempts"].append(trajectory["summary"])
            if trajectory["summary"]["success"]:
                if dataset is None:
                    dataset = new_dataset(out / "dataset", "local/so101_r2s2r_sim")
                export_episode(dataset, trajectory)
                report["accepted"] += 1
            write_json(out / "generation_manifest.json", report)
            print(
                json.dumps({"attempt": attempt, "accepted": report["accepted"], **trajectory["summary"]}),
                flush=True,
            )
            if report["accepted"] == count:
                break
    finally:
        if dataset is not None:
            dataset.finalize()
        sim.close()
    report["passed"] = report["accepted"] == count
    if dataset is not None:
        report["dataset_sha256"] = {
            str(p.relative_to(out / "dataset")): digest(p)
            for p in sorted((out / "dataset").rglob("*"))
            if p.is_file()
        }
    write_json(out / "generation_manifest.json", report)
    return report


def trial_plan(config, baseline, mixed):
    """The plan is fixed before testing; it never commands physical hardware."""
    rng = np.random.default_rng(20261016)
    xy_range = np.asarray(config["generation"]["xy_range_m"])
    trials = []
    for i in range(30):
        pos = np.asarray(config["tape"]["position_m"]).copy()
        pos[:2] += rng.uniform(-xy_range, xy_range)
        trials.append(
            {
                "layout_id": i,
                "tape_position_m": pos.tolist(),
                "holder_position_m": config["holder"]["position_m"],
                "tape_yaw_rad": float(rng.uniform(-0.17, 0.17)),
                "order": ["baseline", "mixed"] if i % 2 == 0 else ["mixed", "baseline"],
            }
        )
    return {
        "schema_version": 1,
        "kind": "physical_paired_trial_plan",
        "config_sha256": fingerprint(config),
        "checkpoints": {"baseline": baseline, "mixed": mixed},
        "trials": trials,
    }


def evaluate_policy(config, out, checkpoint, device, seed, count, calibration, replay_report):
    """Run SmolVLA on disjoint simulation layouts, never physical hardware."""
    import torch

    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    generation_gates(config, calibration, replay_report)
    if seed < 100000 or not 1 <= count <= 30:
        raise ValueError("Use disjoint evaluation seeds >=100000 and 1..30 episodes")
    source = SourceData(config)
    _, states, _ = template_pose(config, source)
    policy_config = SmolVLAConfig.from_pretrained(str(checkpoint))
    policy_config.device = device
    policy_config.n_action_steps = 30
    policy = SmolVLAPolicy.from_pretrained(str(checkpoint), config=policy_config).to(device).eval()
    pre, post = make_pre_post_processors(
        policy_cfg=policy_config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    sim = Simulator(config, out / "scene.xml")
    report = {
        "kind": "simulation_policy_evaluation",
        "checkpoint": str(checkpoint),
        "config_sha256": fingerprint(config),
        "seed": seed,
        "episodes": [],
        "interpretation": "Simulation result; verify physical success independently",
    }
    try:
        for episode in range(count):
            torch.manual_seed(seed + episode)
            sim.reset(states[0], seed + episode, randomize=True)
            policy.reset()
            pre.reset()
            post.reset()
            rows, frames, fault = [], [], None
            try:
                for step in range(900):
                    images, state = sim.images(), sim.state()
                    observation = {
                        "observation.state": torch.from_numpy(state),
                        "task": "Put the tape into the pen holder",
                    }
                    for name, frame in images.items():
                        observation[f"observation.images.{name}"] = (
                            torch.from_numpy(frame).permute(2, 0, 1).float() / 255
                        )
                    with torch.inference_mode():
                        action = (
                            post(policy.select_action(pre(observation))).detach().cpu().numpy().reshape(-1)
                        )
                    frames.append(images)
                    outcome = sim.step(action)
                    rows.append(
                        {"step": step, "state": state.tolist(), "sent_action": action.tolist(), **outcome}
                    )
                    if outcome["success"]:
                        break
            except (ValueError, RuntimeError) as exc:
                fault = str(exc)
                frames = frames[: len(rows)]
            summary = {
                "seed": seed + episode,
                "success": bool(rows and rows[-1]["success"]),
                "grasped": bool(sim.tracker.grasped),
                "fault": fault,
                "steps": len(rows),
                "randomization": sim.randomization,
            }
            save_attempt(out, episode, {"rows": rows, "frames": frames, "summary": summary})
            report["episodes"].append(summary)
            write_json(out / "simulation_evaluation.json", report)
    finally:
        sim.close()
    report["successes"] = sum(row["success"] for row in report["episodes"])
    report["faults"] = sum(row["fault"] is not None for row in report["episodes"])
    write_json(out / "simulation_evaluation.json", report)
    return report


def compare_trials(plan, results):
    """Compare a complete set of human-verified, matched physical trials."""
    entries = results["results"]
    by_key = {}
    for row in entries:
        key = (row["candidate"], row["layout_id"])
        if key in by_key or row["candidate"] not in ("baseline", "mixed"):
            raise ValueError("Duplicate or unknown candidate/layout")
        for field in ("grasped", "placed", "fault"):
            if type(row.get(field)) is not bool:
                raise ValueError(f"Every completed trial needs a human-verified boolean: {field}")
        if not row.get("evidence") or not np.isfinite(row.get("seconds", float("nan"))) or row["seconds"] < 0:
            raise ValueError("Every trial needs evidence and finite elapsed seconds")
        if row["placed"] and not row["grasped"]:
            raise ValueError("Placement cannot succeed without a grasp")
        by_key[key] = row
    if results.get("plan_sha256") != fingerprint(plan):
        raise ValueError("Results do not belong to this fixed trial plan")
    expected = {(candidate, t["layout_id"]) for candidate in ("baseline", "mixed") for t in plan["trials"]}
    if len(plan["trials"]) != 30 or len(expected) != 60 or set(by_key) != expected:
        raise ValueError("Need exactly 30 paired layouts and 60 completed trials")
    summary = {}
    for candidate in ("baseline", "mixed"):
        rows = [row for (name, _), row in by_key.items() if name == candidate]
        summary[candidate] = {
            "grasp_successes": sum(r["grasped"] for r in rows),
            "placement_successes": sum(r["placed"] for r in rows),
            "faults": sum(r["fault"] for r in rows),
            "mean_seconds": float(np.mean([r["seconds"] for r in rows])),
        }
    delta = summary["mixed"]["placement_successes"] - summary["baseline"]["placement_successes"]
    return {
        "summary": summary,
        "extra_successes": delta,
        "passed": delta >= 6 and summary["mixed"]["faults"] <= summary["baseline"]["faults"],
        "interpretation": "Preliminary 30-layout paired evaluation, not a statistical significance claim",
    }
