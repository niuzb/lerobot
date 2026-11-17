"""Run the LeIsaac tape task, retaining physical evidence before image export."""

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
from bootstrap import launch
from common import SuccessTracker, sha256, write_json
from spatial import base_position, table_height


def force(sensor):
    values = sensor.data.force_matrix_w
    if values is None:
        raise RuntimeError("Filtered contact force evidence is unavailable")
    return float(values.norm(dim=-1).sum().cpu())


def snapshot(env, joint_ids):
    robot = env.scene["robot"]
    return {
        "qpos": robot.data.joint_pos[0, joint_ids].cpu().numpy().copy(),
        "qvel": robot.data.joint_vel[0, joint_ids].cpu().numpy().copy(),
        "robot_links": robot.data.body_link_state_w[0, :, :7].cpu().numpy().copy(),
        "tape": env.scene["tape"].data.root_state_w[0].cpu().numpy().copy(),
        "holder": env.scene["holder"].data.root_state_w[0].cpu().numpy().copy(),
    }


def attempt(env, sm, config, seed, out):
    import torch

    rng = np.random.default_rng(seed)
    env.tape_success = False
    env.reset(seed=seed)
    tape = env.scene["tape"]
    state = tape.data.default_root_state.clone()
    delta = rng.uniform(
        -np.asarray(config["generation"]["xy_range_m"]), np.asarray(config["generation"]["xy_range_m"])
    )
    yaw = rng.uniform(-config["generation"]["yaw_range_rad"], config["generation"]["yaw_range_rad"])
    state[0, :2] += torch.as_tensor(delta, dtype=state.dtype, device=env.device)
    state[0, 3:7] = torch.tensor([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)], device=env.device)
    tape.write_root_state_to_sim(state)
    friction = float(rng.uniform(0.8, 1.2))
    materials = tape.root_physx_view.get_material_properties().clone()
    materials[..., :2] = friction
    tape.root_physx_view.set_material_properties(materials, torch.tensor([0], dtype=torch.int64))
    initial = sm.mapping.to_sim(sm.initial)
    command = torch.as_tensor(initial, dtype=torch.float32, device=env.device).unsqueeze(0)
    for _ in range(15):
        env.step(command)
    sm.begin(env)
    initial_position = tape.data.root_pos_w[0].cpu().numpy().copy()
    tracker = SuccessTracker(config, initial_position)
    records = []
    reason = "no_verified_containment"
    try:
        while not sm.is_episode_done:
            current = snapshot(env, sm.joint_ids)
            stage = sm.stage
            last_in_stage = sm.tick + 1 == sm.stages[sm.stage_index][3]
            action = sm.get_action(env)
            env.step(action)
            forces = [
                force(env.scene[name]) for name in ("fixed_contact", "moving_contact", "support_contact")
            ]
            fixed, moving, support = (value > 1e-6 for value in forces)
            obj, holder = tape.data, env.scene["holder"].data
            position = obj.root_pos_w[0].cpu().numpy().copy()
            state_robot = sm.mapping.to_robot(
                env.scene["robot"].data.joint_pos[0, sm.joint_ids].cpu().numpy()
            )
            success = tracker.update(
                position,
                obj.root_quat_w[0].cpu().numpy(),
                obj.root_vel_w[0].cpu().numpy(),
                state_robot[5],
                fixed,
                moving,
                support,
                holder.root_pos_w[0].cpu().numpy(),
                holder.root_quat_w[0].cpu().numpy(),
            )
            records.append(
                {
                    **current,
                    "target": action[0].cpu().numpy().copy(),
                    "stage": sm.stage_index,
                    "forces": forces,
                    "success": success,
                    "object_after": position,
                    "stable": tracker.stable,
                }
            )
            if success:
                # Complete the episode when the physical success termination
                # fires, after 30 consecutive stable control ticks.
                env.tape_success = True
                break
            if last_in_stage:
                print(
                    json.dumps(
                        {
                            "seed": seed,
                            "stage": stage,
                            "object_m": position.tolist(),
                            "forces_n": forces,
                            "support_net_force_n": env.scene["support_contact"].data.net_forces_w.tolist(),
                            "object_velocity": obj.root_vel_w[0].tolist(),
                            "stable": tracker.stable,
                        }
                    ),
                    flush=True,
                )
                if stage in ("lift", "transport") and (position[2] < 0.08 or not (fixed or moving)):
                    raise RuntimeError(f"Grasp lost at {stage}")
            sm.advance()
        sm.accepted = bool(records and records[-1]["success"])
        reason = "verified_containment" if sm.accepted else reason
    except (ValueError, RuntimeError) as exc:
        reason = str(exc)
    arrays = {
        name: np.asarray([record[name] for record in records])
        for name in (
            "qpos",
            "qvel",
            "robot_links",
            "tape",
            "holder",
            "target",
            "stage",
            "forces",
            "success",
            "object_after",
            "stable",
        )
    }
    np.savez_compressed(out.with_suffix(".npz"), **arrays)
    summary = {
        "seed": seed,
        "success": sm.accepted,
        "reason": reason,
        "frames": len(records),
        "grasped": tracker.grasped,
        "lifted": tracker.lifted,
        "carried": tracker.carried,
        "stable_control_ticks": tracker.stable,
        "fk_error_m": sm.fk_error_m,
        "randomization": {"tape_xy_delta_m": delta.tolist(), "yaw_rad": yaw, "tape_friction": friction},
        "npz_sha256": sha256(out.with_suffix(".npz")),
        "calibration_passed": False,
    }
    write_json(out.with_suffix(".json"), summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--leisaac-root", type=Path, required=True)
    parser.add_argument("--leisaac-assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=25)
    parser.add_argument("--seed", type=int, default=60000)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads((args.run / "scene.json").read_text())
    os.environ["LEISAAC_ASSETS_ROOT"] = str(args.leisaac_assets)
    app = launch(args.run, args.leisaac_root)
    env = None
    try:
        print("LOADING_LEISAAC_TASK", flush=True)
        import gymnasium as gym
        from planner import TapeStateMachine
        from task import make_config

        cfg, links = make_config(config, args.run / "assets")
        print("TASK_CONFIG_CREATED", flush=True)
        task_id = "LeIsaac-SO101-TapePenHolder-v0"
        if task_id not in gym.registry:
            gym.register(task_id, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True)
        env = gym.make(task_id, cfg=cfg).unwrapped
        env.reset()
        stage = env.sim.stage
        write_json(
            args.output / "physics_attributes.json",
            {
                str(prim.GetPath()): {
                    attr.GetName(): str(attr.Get())
                    for attr in prim.GetAttributes()
                    if attr.GetName().startswith(("physxRigidBody:", "physxContactReport:"))
                }
                for prim in stage.Traverse()
                if str(prim.GetPath()) in ("/World/envs/env_0/Tape", "/World/envs/env_0/Holder")
            },
        )
        sm = TapeStateMachine(config)
        sm.setup(env)
        write_json(
            args.output / "environment.json",
            {
                "task": task_id,
                "reused_foundation": "LeIsaac PickOrange / SingleArmTaskEnvCfg / StateMachineBase",
                "physical_engine": "PhysX",
                "joints": env.scene["robot"].data.joint_names,
                "body_names": env.scene["robot"].body_names,
                "body_links": links,
                "fk_error_m": sm.fk_error_m,
                "fps": 30,
                "physics_hz": 300,
                "table_top_height_world_m": table_height(config),
                "robot_base_position_world_m": base_position(config).tolist(),
                "scene_config_sha256": sha256(args.run / "scene.json"),
                "calibration_passed": False,
            },
        )
        if args.inspect_only:
            print("LEISAAC_TAPE_ENVIRONMENT_READY", flush=True)
            return
        report = {
            "accepted": 0,
            "requested": args.count,
            "passed": False,
            "calibration_passed": False,
            "attempts": [],
            "source": config["migration"],
        }
        for index in range(args.max_attempts):
            begin = time.monotonic()
            summary = attempt(env, sm, config, args.seed + index, args.output / f"attempt_{index:04d}")
            report["attempts"].append(summary)
            report["accepted"] += int(summary["success"])
            print(
                json.dumps(
                    {
                        "attempt": index,
                        "accepted": report["accepted"],
                        "reason": summary["reason"],
                        "seconds": time.monotonic() - begin,
                    }
                ),
                flush=True,
            )
            report["passed"] = report["accepted"] >= args.count
            write_json(args.output / "generation_manifest.json", report)
            if report["passed"]:
                break
        if not report["passed"]:
            raise RuntimeError("Requested physical successes not reached; no training data was exported")
    except BaseException as exc:
        traceback.print_exc()
        write_json(args.output / "failure.json", {"error": repr(exc), "calibration_passed": False})
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
