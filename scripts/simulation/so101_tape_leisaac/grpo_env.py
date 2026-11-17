"""Live PhysX rollouts: only policy actions control the robot, never replay poses."""

import argparse
import hashlib
import json
import os
import socket
import time
from pathlib import Path

import numpy as np
from bootstrap import launch
from common import JOINTS, JointMap, Kinematics, SuccessTracker, transform, write_json
from grpo_curriculum import CurriculumSchedule, tracker_from_prefix, validate_anchor
from grpo_wire import request
from spatial import frame_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--leisaac-root", type=Path, required=True)
    parser.add_argument("--leisaac-assets", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--curriculum", type=Path)
    parser.add_argument("--epoch-offset", type=int, default=0)
    parser.add_argument("--restore-curriculum-state", type=Path)
    args = parser.parse_args()
    config = json.loads((args.run / "scene.json").read_text())
    curriculum = json.loads(args.curriculum.read_text()) if args.curriculum else None
    source_physics = None
    if curriculum:
        source_path = Path(curriculum["source_physics"])
        if hashlib.sha256(source_path.read_bytes()).hexdigest() != curriculum["source_physics_sha256"]:
            raise ValueError("Curriculum source physics changed")
        source_physics = np.load(source_path, allow_pickle=False)
        for name, anchor in curriculum["anchors"].items():
            validate_anchor(config, source_physics, curriculum["initial_tape_position"], name, anchor["tick"])
    os.environ["LEISAAC_ASSETS_ROOT"] = str(args.leisaac_assets)
    app = launch(args.run, args.leisaac_root, cameras=True)
    env = None
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.settimeout(1800)
    try:
        import cv2
        import gymnasium as gym
        import omni.replicator.core as rep
        import torch
        from collect import force
        from task import make_config
        from texture_randomization import GROUPS, FrameTextures

        cfg, _ = make_config(config, args.run / "assets", cameras=True)
        # Match the tested renderer's initialization order; importing managers
        # before the task/env modules causes an Isaac Lab circular import.
        from isaaclab.managers import TerminationTermCfg

        cfg.sim.use_fabric = True
        for name in dir(cfg.terminations):
            if not name.startswith("_") and isinstance(getattr(cfg.terminations, name), TerminationTermCfg):
                setattr(cfg.terminations, name, None)
        task_id = "LeIsaac-SO101-TapeGRPO-v0"
        if task_id not in gym.registry:
            gym.register(task_id, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True)
        env = gym.make(task_id, cfg=cfg).unwrapped
        joint_ids, names = env.scene["robot"].find_joints(JOINTS, preserve_order=True)
        assert names == JOINTS
        robot, tape, holder = env.scene["robot"], env.scene["tape"], env.scene["holder"]
        mapping, fk = JointMap(config["joint_map"]), Kinematics(config["asset"])
        base_i, gripper_i = robot.body_names.index("base"), robot.body_names.index("gripper")
        initial = np.array(config["initial_robot_state"])
        initial[5] = config["generation"]["open_percent"]
        initial_q = mapping.to_sim(initial)
        rep.orchestrator.set_capture_on_play(False)

        def status(stage, **values):
            write_json(
                args.run / "status.json",
                {"stage": stage, "native_pid": os.getpid(), "at_unix": time.time(), **values},
            )
            print(json.dumps({"stage": stage, **values}), flush=True)

        def step(q):
            if not env.sim.is_playing():
                env.sim.play()
            _, _, terminated, truncated, _ = env.step(torch.as_tensor(q, dtype=torch.float32).unsqueeze(0))
            if bool(terminated.any() | truncated.any()):
                raise RuntimeError("Unexpected native autoreset would invalidate group identity")

        def reset(seed, reference=None, anchor=None):
            env.tape_success = False
            env.reset(seed=seed)
            robot.write_joint_state_to_sim(
                torch.as_tensor(initial_q, dtype=torch.float32).unsqueeze(0),
                torch.zeros((1, 6), dtype=torch.float32),
                joint_ids=joint_ids,
            )
            holder.write_root_state_to_sim(holder.data.default_root_state.clone())
            rng = np.random.default_rng(seed)
            state = tape.data.default_root_state.clone()
            state[0, :2] += torch.as_tensor(
                rng.uniform(
                    -np.array(config["generation"]["xy_range_m"]), config["generation"]["xy_range_m"]
                ),
                dtype=state.dtype,
            )
            yaw = rng.uniform(-config["generation"]["yaw_range_rad"], config["generation"]["yaw_range_rad"])
            state[0, 3:7] = torch.tensor([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
            tape.write_root_state_to_sim(state)
            materials = tape.root_physx_view.get_material_properties().clone()
            materials[..., :2] = float(rng.uniform(0.8, 1.2))
            tape.root_physx_view.set_material_properties(materials, torch.tensor([0], dtype=torch.int64))
            for _ in range(15):
                step(initial_q)
            if anchor is not None:
                tick = anchor["tick"]
                anchor_q = mapping.to_sim(source_physics["state"][tick])
                robot.write_joint_state_to_sim(
                    torch.as_tensor(anchor_q, dtype=torch.float32).unsqueeze(0),
                    torch.zeros((1, 6), dtype=torch.float32),
                    joint_ids=joint_ids,
                )
                for body, key in [(tape, "tape"), (holder, "holder")]:
                    body_state = source_physics[key][tick].copy()
                    body_state[7:] = 0
                    body.write_root_state_to_sim(
                        torch.as_tensor(body_state, dtype=torch.float32).unsqueeze(0)
                    )
                materials[..., :2] = curriculum["source_friction"]
                tape.root_physx_view.set_material_properties(materials, torch.tensor([0], dtype=torch.int64))
                env.sim.forward()
                for _ in range(10):
                    step(anchor_q)
                displacement = float(
                    np.linalg.norm(tape.data.root_pos_w[0].cpu().numpy() - source_physics["tape"][tick, :3])
                )
                contacts = [force(env.scene[name]) > 1e-6 for name in ["fixed_contact", "moving_contact"]]
                if displacement > 0.01 or (anchor["require_held"] and not all(contacts)):
                    raise RuntimeError("Curriculum initial state did not retain the required physical grasp")
                if anchor.get("require_opening_action"):
                    gripper = mapping.to_robot(robot.data.joint_pos[0, joint_ids].cpu().numpy())[5]
                    probe_tracker = tracker_from_prefix(
                        config, source_physics, curriculum["initial_tape_position"], tick
                    )
                    probe_tracker.update(
                        tape.data.root_pos_w[0].cpu().numpy(),
                        tape.data.root_quat_w[0].cpu().numpy(),
                        tape.data.root_vel_w[0].cpu().numpy(),
                        config["generation"]["open_percent"],
                        *contacts,
                        force(env.scene["support_contact"]) > 1e-6,
                        holder.data.root_pos_w[0].cpu().numpy(),
                        holder.data.root_quat_w[0].cpu().numpy(),
                    )
                    if gripper >= config["generation"]["open_percent"] - 5 or probe_tracker.stable != 1:
                        raise RuntimeError(
                            "Open-hold reset must require opening and retain physical containment/support"
                        )
            # Seeded settle can leave small solver-dependent residual velocities.
            # Restore a matched, quiescent snapshot only at the episode boundary.
            if reference is None:
                q = robot.data.joint_pos[0, joint_ids].cpu().numpy().copy()
                ts = tape.data.root_state_w[0].cpu().numpy().copy()
                hs = holder.data.root_state_w[0].cpu().numpy().copy()
                ts[7:] = hs[7:] = 0
            else:
                q, ts, hs = reference[:6], reference[6:19], reference[19:32]
            robot.write_joint_state_to_sim(
                torch.as_tensor(q, dtype=torch.float32).unsqueeze(0),
                torch.zeros((1, 6), dtype=torch.float32),
                joint_ids=joint_ids,
            )
            tape.write_root_state_to_sim(torch.as_tensor(ts, dtype=torch.float32).unsqueeze(0))
            holder.write_root_state_to_sim(torch.as_tensor(hs, dtype=torch.float32).unsqueeze(0))
            env.sim.forward()
            return np.r_[
                robot.data.joint_pos[0, joint_ids].cpu().numpy(),
                tape.data.root_state_w[0].cpu().numpy(),
                holder.data.root_state_w[0].cpu().numpy(),
            ]

        def images():
            rep.orchestrator.step(rt_subframes=4, delta_time=0.0, pause_timeline=False)
            result = {}
            for name in ["front", "side"]:
                camera = env.scene[name]
                camera.update(1 / 30, force_recompute=True)
                frame = camera.data.output["rgb"][0].cpu().numpy().copy()[..., :3]
                if frame.shape != (480, 640, 3) or frame.std() < 1:
                    raise RuntimeError("Missing/non-informative live camera: " + name)
                result[name] = frame
            return result

        def observation(rgb):
            links = robot.data.body_link_state_w[0].cpu().numpy()
            base, grip = links[base_i], links[gripper_i]
            auxiliary, _ = frame_features(
                config, transform(base[:3], base[3:7]), transform(grip[:3], grip[3:7])
            )
            raw_state = mapping.to_robot(robot.data.joint_pos[0, joint_ids].cpu().numpy())
            return {"state": np.r_[raw_state, auxiliary].astype(np.float32), **rgb}

        status("camera_motion_probe")
        reset(14000000)
        for _ in range(3):
            images()
        before = images()
        probe_q_before = robot.data.joint_pos[0, joint_ids].cpu().numpy().copy()
        target = initial_q.copy()
        target[0] += np.deg2rad(3)
        for _ in range(20):
            step(target)
        after = images()
        if abs(float(robot.data.joint_pos[0, joint_ids[0]]) - probe_q_before[0]) < np.deg2rad(1):
            raise RuntimeError("Physical camera probe did not actually move the robot")
        probe = {name: float(np.abs(before[name].astype(float) - after[name]).mean()) for name in before}
        write_json(args.run / "camera_motion_probe.json", probe)
        if any(value < 0.01 for value in probe.values()):
            raise RuntimeError("Camera observations stayed frozen after physical joint motion")
        # Execute a previously validated action sequence against live physics.
        # Only the INITIAL state is restored; no replayed poses are written later.
        # This is an infrastructure control, never a policy/GRPO rollout.
        reference = np.load(
            "/root/data/leisaac/runs/tape-train100-20261109-01/samples/candidate_0000/physics/attempt_0000.npz"
        )
        control_success = False
        for control_attempt in range(1):
            reset(1000000)
            robot.write_joint_state_to_sim(
                torch.as_tensor(reference["qpos"][0]).unsqueeze(0),
                torch.as_tensor(reference["qvel"][0]).unsqueeze(0),
                joint_ids=joint_ids,
            )
            tape.write_root_state_to_sim(
                torch.as_tensor(reference["tape"][0], dtype=torch.float32).unsqueeze(0)
            )
            holder.write_root_state_to_sim(
                torch.as_tensor(reference["holder"][0], dtype=torch.float32).unsqueeze(0)
            )
            env.sim.forward()
            tracker = SuccessTracker(config, reference["tape"][0, :3])
            for target in reference["target"]:
                step(target)
                forces = [force(env.scene[n]) for n in ["fixed_contact", "moving_contact", "support_contact"]]
                gripper = mapping.to_robot(robot.data.joint_pos[0, joint_ids].cpu().numpy())[5]
                control_success = tracker.update(
                    tape.data.root_pos_w[0].cpu().numpy(),
                    tape.data.root_quat_w[0].cpu().numpy(),
                    tape.data.root_vel_w[0].cpu().numpy(),
                    gripper,
                    *(f > 1e-6 for f in forces),
                    holder.data.root_pos_w[0].cpu().numpy(),
                    holder.data.root_quat_w[0].cpu().numpy(),
                )
                if control_success:
                    break
            write_json(
                args.run / f"positive_control_{control_attempt}.json",
                {
                    "success": bool(control_success),
                    "training_rollout": False,
                    "actions_from": "frozen successful source trace",
                    "stable_ticks": tracker.stable,
                },
            )
            if control_success:
                break
        write_json(
            args.run / "positive_control_probe.json",
            {"success": bool(control_success), "training_rollout": False, "attempts": control_attempt + 1},
        )
        if not control_success:
            print(
                "DIAGNOSTIC_ACTION_REPLAY_DID_NOT_SUCCEED: preserve failure and measure policy outcomes; reward gate unchanged",
                flush=True,
            )
        views = [robot.root_physx_view, tape.root_physx_view, holder.root_physx_view]
        material_properties = [view.get_material_properties().clone() for view in views]
        textures = FrameTextures(env.sim.stage, config, args.run / "texture_bank", 81000, 16)
        for texture_id in range(16):
            textures.apply(dict.fromkeys(GROUPS, texture_id))
            images()
        for view, properties in zip(views, material_properties, strict=True):
            view.set_material_properties(properties, torch.tensor([0], dtype=torch.int64))
        write_json(
            args.run / "material_preservation.json",
            {
                "restored_runtime_materials": True,
                "physics_authored_fingerprint_unchanged": textures.before_physics,
            },
        )
        connection.connect(str(args.socket))
        episode_counter = 0

        if curriculum:
            probe_stage = curriculum["schedule"].get("stages", ["release"])[0]
            status("curriculum_reset_probe", task_stage=probe_stage)
            reset(18000000, anchor=curriculum["anchors"][probe_stage])
            write_json(
                args.run / "curriculum_reset_probe.json",
                {"passed": True, "training_rollout": False, "task_stage": probe_stage},
            )

        def rollout(epoch, index, group_seed, mode, reference=None, sample_seed=None, task_stage="full"):
            nonlocal episode_counter
            episode_counter += 1
            out = args.run / "rollouts" / f"epoch_{epoch:03d}" / f"episode_{index:02d}"
            out.mkdir(parents=True, exist_ok=False)
            anchor = curriculum["anchors"].get(task_stage) if curriculum else None
            initial_state = reset(group_seed, reference, anchor)
            if reference is not None and not np.allclose(initial_state, reference, atol=1e-4, rtol=0):
                raise RuntimeError("Reset-matched GRPO group identity failed")
            rng = np.random.default_rng(group_seed + 41)
            texture_ids = dict.fromkeys(GROUPS, 0)
            request(
                connection,
                {
                    "operation": "begin",
                    "episode": episode_counter,
                    "epoch": epoch,
                    "group_seed": group_seed,
                    "sample_seed": sample_seed if sample_seed is not None else 23000000 + epoch * 100 + index,
                    "mode": mode,
                    "task_stage": task_stage,
                },
            )
            tracker = SuccessTracker(config, tape.data.root_pos_w[0].cpu().numpy().copy())
            if anchor is not None:
                tracker = tracker_from_prefix(
                    config, source_physics, curriculum["initial_tape_position"], anchor["tick"]
                )
            records = []
            action_chunk = None
            writers = {}
            if epoch == 0 and index == 0:
                writers = {
                    name: cv2.VideoWriter(
                        str(out / (name + ".mp4")), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 480)
                    )
                    for name in ["front", "side"]
                }
            try:
                horizon = anchor["max_steps"] if anchor else args.max_steps
                for tick in range(horizon):
                    for name in GROUPS:
                        texture_ids[name] = (texture_ids[name] + int(rng.integers(1, 16))) % 16
                    textures.apply(texture_ids)
                    if tick % 30 == 0:
                        rgb = images()
                        _, packet = request(connection, {"operation": "act"}, **observation(rgb))
                        action_chunk = packet["action"]
                    raw = action_chunk[tick % 30].copy()
                    q = np.clip(mapping.to_sim(raw), fk.bounds[:, 0], fk.bounds[:, 1])
                    step(q)
                    forces = [
                        force(env.scene[name])
                        for name in ["fixed_contact", "moving_contact", "support_contact"]
                    ]
                    robot_state = mapping.to_robot(robot.data.joint_pos[0, joint_ids].cpu().numpy())
                    success = tracker.update(
                        tape.data.root_pos_w[0].cpu().numpy(),
                        tape.data.root_quat_w[0].cpu().numpy(),
                        tape.data.root_vel_w[0].cpu().numpy(),
                        robot_state[5],
                        *(value > 1e-6 for value in forces),
                        holder.data.root_pos_w[0].cpu().numpy(),
                        holder.data.root_quat_w[0].cpu().numpy(),
                    )
                    records.append(
                        {
                            "state": robot_state.copy(),
                            "raw_action": raw,
                            "target_q": q.copy(),
                            "tape": tape.data.root_state_w[0].cpu().numpy().copy(),
                            "holder": holder.data.root_state_w[0].cpu().numpy().copy(),
                            "forces": forces,
                            "success": success,
                            "stable": tracker.stable,
                        }
                    )
                    if writers:
                        for name, frame in images().items():
                            writers[name].write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    if success or tape.data.root_pos_w[0, 2] < -0.1:
                        break
            finally:
                for writer in writers.values():
                    writer.release()
            success = bool(records and records[-1]["success"])
            np.savez_compressed(
                out / "physics.npz", **{key: np.asarray([r[key] for r in records]) for key in records[0]}
            )
            summary = {
                "episode": episode_counter,
                "epoch": epoch,
                "index": index,
                "group_seed": group_seed,
                "mode": mode,
                "success": success,
                "reward": int(success),
                "control_steps": len(records),
                "grasped": tracker.grasped,
                "lifted": tracker.lifted,
                "carried": tracker.carried,
                "stable": tracker.stable,
                "initial_state": initial_state.tolist(),
                "physics_npz_sha256": hashlib.sha256((out / "physics.npz").read_bytes()).hexdigest(),
                "policy_controls_only": True,
                "task_stage": task_stage,
                "full_task": task_stage == "full",
                "curriculum_source_tick": anchor["tick"] if anchor else None,
                "historical_prerequisites_from_recorded_prefix": anchor is not None,
                "source_prefix_counted_as_rl": False,
            }
            write_json(out / "summary.json", summary)
            request(
                connection,
                {
                    "operation": "end",
                    "episode": episode_counter,
                    "success": success,
                    "control_steps": len(records),
                },
            )
            print(json.dumps(summary), flush=True)
            return summary, initial_state

        baseline = []
        for index in range(2):
            status("baseline_ode", completed=index, total=2)
            result, _ = rollout(0, index, 15000000 + index, "ode")
            baseline.append(result)
        write_json(
            args.run / "baseline.json",
            {
                "episodes": baseline,
                "n": len(baseline),
                "successes": sum(x["success"] for x in baseline),
                "exploratory_only": True,
            },
        )
        history = []
        schedule = CurriculumSchedule(**curriculum["schedule"]) if curriculum else None
        if args.restore_curriculum_state:
            restored = json.loads(args.restore_curriculum_state.read_text())
            schedule = CurriculumSchedule.from_state_dict(restored["curriculum_state"])
            if list(schedule.stages) != curriculum["schedule"].get("stages", list(schedule.stages)):
                raise ValueError("Restored curriculum does not match the frozen plan")
        stalled = False
        target_epoch = args.epoch_offset + args.epochs
        for epoch in range(args.epoch_offset + 1, target_epoch + 1):
            reference = None
            task_stage = schedule.stage if schedule else "full"
            rewards = []
            for index in range(8):
                status(
                    "collecting",
                    epoch=epoch,
                    target_epochs=target_epoch,
                    rollout=index,
                    group_size=8,
                    task_stage=task_stage,
                )
                result, initial_state = rollout(
                    epoch,
                    index,
                    18000000 + epoch if schedule else 16000000 + epoch,
                    "sde",
                    reference,
                    task_stage=task_stage,
                )
                rewards.append(result["reward"])
                if reference is None:
                    reference = initial_state
            status("grpo_update", epoch=epoch, target_epochs=target_epoch)
            curriculum_result = schedule.observe(rewards) if schedule else None
            result, _ = request(
                connection,
                {
                    "operation": "update",
                    "epoch": epoch,
                    "curriculum_state": schedule.state_dict() if schedule else None,
                },
            )
            if schedule:
                result["curriculum"] = curriculum_result
                write_json(
                    args.run / "curriculum_status.json",
                    {"epoch": epoch, **result["curriculum"], "completed_stages": schedule.completed_stages},
                )
            history.append(result)
            write_json(args.run / "training_history.json", history)
            status(
                "epoch_finished",
                epoch=epoch,
                target_epochs=target_epoch,
                optimizer_steps=result["optimizer_steps"],
                rewards=result["rewards"],
                skipped=result["skipped"],
                task_stage=task_stage,
            )
            if schedule and result["curriculum"]["decision"] == "stalled":
                stalled = True
                break
        if stalled:
            final, _ = request(connection, {"operation": "finish", "epoch": epoch})
            status("curriculum_stalled", completed_epochs=epoch, task_stage=task_stage, **final)
            return
        final_evaluation = []
        for index in range(2):
            status("final_ode_evaluation", completed=index, total=2)
            result, _ = rollout(
                target_epoch + 1, index, 15000000 + index, "ode", sample_seed=23000000 + index
            )
            final_evaluation.append(result)
        write_json(
            args.run / "simulation_comparison.json",
            {
                "episodes": 2,
                "baseline_successes": sum(x["success"] for x in baseline),
                "grpo_successes": sum(x["success"] for x in final_evaluation),
                "same_reset_and_sampling_seeds": True,
                "exploratory_only": True,
            },
        )
        final, _ = request(connection, {"operation": "finish", "epoch": target_epoch})
        status(
            "complete" if final["optimizer_steps"] > 0 else "no_learning_signal",
            completed_epochs=target_epoch,
            **final,
        )
    except BaseException as error:
        write_json(args.run / "environment_failure.json", {"error": repr(error), "pid": os.getpid()})
        raise
    finally:
        connection.close()
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
