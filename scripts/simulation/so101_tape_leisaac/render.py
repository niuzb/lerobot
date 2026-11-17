"""Render recorded PhysX observations through LeIsaac's two camera sensors."""

import argparse
import faulthandler
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
from bootstrap import launch
from common import JOINTS, Kinematics, sha256, transform, write_json


def main():
    faulthandler.enable()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--physics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--leisaac-root", type=Path, required=True)
    parser.add_argument("--leisaac-assets", type=Path, required=True)
    parser.add_argument("--camera-config", type=Path, help="Camera-only overrides for view revisions")
    parser.add_argument("--attempt", type=int)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--framewise-textures", action="store_true", help="Change each texture group at every recorded frame"
    )
    parser.add_argument("--texture-seed", type=int, default=81000)
    parser.add_argument(
        "--texture-schedule-seed", type=int, help="Independent frame schedule; defaults to the bank seed"
    )
    parser.add_argument("--texture-count", type=int, default=16)
    parser.add_argument("--texture-subframes", type=int, default=4)
    parser.add_argument(
        "--max-frames", type=int, help="Short diagnostic capture; incomplete episodes cannot be exported"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads((args.run / "scene.json").read_text())
    if args.camera_config is not None:
        camera_revision = json.loads(args.camera_config.read_text())
        config["cameras"] = camera_revision["cameras"]
        write_json(args.output / "camera_config.json", camera_revision)
    os.environ["LEISAAC_ASSETS_ROOT"] = str(args.leisaac_assets)
    if args.stride < 1:
        raise ValueError("Stride must be positive")
    if args.framewise_textures and args.texture_subframes < 2:
        raise ValueError("Texture changes require at least two renderer subframes")
    if args.framewise_textures and (args.texture_count < 2 or args.texture_seed < 0):
        raise ValueError("Require at least two textures per group and a nonnegative seed")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("max-frames must be positive")
    if args.attempt is None:
        manifest = json.loads((args.physics / "generation_manifest.json").read_text())
        selected = [i for i, item in enumerate(manifest["attempts"]) if item["success"]]
    else:
        selected = [args.attempt]
    if not selected:
        raise ValueError("No physical trajectories selected")

    app = launch(args.run, args.leisaac_root, cameras=True)
    env = None
    writers = []
    try:
        import cv2
        import gymnasium as gym
        import omni.replicator.core as rep
        import torch
        from pxr import Gf, UsdGeom
        from task import make_config

        cfg, links = make_config(config, args.run / "assets", cameras=True)
        # Replay uses recorded world poses. USD updates avoid the CPU PhysX /
        # Fabric transform cache retaining the first frame while paused.
        cfg.sim.use_fabric = False
        task_id = "LeIsaac-SO101-TapePenHolder-v0"
        if task_id not in gym.registry:
            gym.register(task_id, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True)
        env = gym.make(task_id, cfg=cfg).unwrapped
        print("RENDER_ENV_CREATED", flush=True)
        env.reset()
        print("RENDER_ENV_RESET", flush=True)
        env.sim.pause()
        robot = env.scene["robot"]
        physical_environment = json.loads((args.physics / "environment.json").read_text())
        if "body_names" in physical_environment and physical_environment["body_names"] != robot.body_names:
            raise ValueError("Replay body order differs from the physical recording")
        joint_ids, names = robot.find_joints(JOINTS, preserve_order=True)
        if names != JOINTS:
            raise ValueError("Joint order changed")
        rep.orchestrator.set_capture_on_play(False)
        report = {
            "passed": False,
            "episodes": [],
            "fps": 30,
            "stride": args.stride,
            "camera_roles": {"front": "wrist", "side": "external"},
            "calibration_passed": False,
            "camera_config_sha256": sha256(args.camera_config) if args.camera_config else None,
            "camera_snapshot_sha256": sha256(args.output / "camera_config.json")
            if args.camera_config
            else None,
            "scene_config_sha256": sha256(args.run / "scene.json"),
            "spatial": config.get("spatial", {}),
        }
        distortion_maps = {}
        replay_ops = {}
        paths = {name: "/World/envs/env_0/Robot" + link for name, link in links.items()}
        paths.update(tape="/World/envs/env_0/Tape", holder="/World/envs/env_0/Holder")
        for name, path in paths.items():
            xform = UsdGeom.Xformable(env.sim.stage.GetPrimAtPath(path))
            replay_ops[name] = xform.MakeMatrixXform()
            xform.SetResetXformStack(True)
        fk = Kinematics(config["asset"])
        for name in ("front", "side"):
            intrinsics = np.asarray(config["cameras"][name]["K"])
            distortion = np.asarray(config["cameras"][name]["distortion"])
            grid = np.stack(np.meshgrid(np.arange(640), np.arange(480)), -1).astype(np.float32)
            ideal = cv2.undistortPoints(grid.reshape(-1, 1, 2), intrinsics, distortion, P=intrinsics).reshape(
                480, 640, 2
            )
            distortion_maps[name] = ideal[..., 0], ideal[..., 1]
        textures = None
        texture_probe = None
        if args.framewise_textures:
            from texture_randomization import FrameTextures

            textures = FrameTextures(
                env.sim.stage, config, args.output / "texture_bank", args.texture_seed, args.texture_count
            )
            report["appearance"] = textures.describe()
            report["appearance"]["renderer_subframes"] = args.texture_subframes
            report["appearance"]["schedule_seed"] = (
                args.texture_schedule_seed if args.texture_schedule_seed is not None else args.texture_seed
            )

        def restore(arrays, index):
            robot.write_joint_state_to_sim(
                torch.as_tensor(arrays["qpos"][index], dtype=torch.float32).unsqueeze(0),
                torch.as_tensor(arrays["qvel"][index], dtype=torch.float32).unsqueeze(0),
                joint_ids=joint_ids,
            )
            if "robot_links" in arrays:
                poses = {
                    name: transform(value[:3], value[3:7])
                    for name, value in zip(robot.body_names, arrays["robot_links"][index], strict=True)
                }
            else:
                # Older failed attempts can still be viewed for diagnosis.
                if summary["success"]:
                    raise ValueError("Successful data requires recorded physical link poses")
                poses = fk.frames(arrays["qpos"][index])
            for name in ("tape", "holder"):
                value = arrays[name][index]
                poses[name] = transform(value[:3], value[3:7])
            maximum_error = 0.0
            for name, op in replay_ops.items():
                matrix = poses[name]
                op.Set(Gf.Matrix4d(matrix.T.tolist()))
                actual = np.asarray(
                    UsdGeom.Xformable(op.GetAttr().GetPrim()).ComputeLocalToWorldTransform(0)
                ).T
                maximum_error = max(maximum_error, float(np.max(np.abs(actual - matrix))))
            if maximum_error > 1e-6:
                raise RuntimeError("USD scene does not match recorded physical world poses")
            return maximum_error

        def images():
            # Capture explicitly; an ordinary world.step alone does not trigger
            # the Replicator graph reliably on the tested standalone runtime.
            rep.orchestrator.step(
                rt_subframes=args.texture_subframes if textures else 1, delta_time=0.0, pause_timeline=True
            )
            result = {}
            for name in ("front", "side"):
                camera = env.scene[name]
                camera.update(1 / 30, force_recompute=True)
                frame = camera.data.output["rgb"][0].cpu().numpy().copy()[..., :3]
                if frame.shape != (480, 640, 3) or frame.std() < 1:
                    raise RuntimeError(f"Missing/non-informative {name} camera image: {frame.shape}")
                result[name] = cv2.remap(frame, *distortion_maps[name], interpolation=cv2.INTER_LINEAR)
            return result

        for attempt_index in selected:
            prefix = args.physics / f"attempt_{attempt_index:04d}"
            summary = json.loads(prefix.with_suffix(".json").read_text())
            if sha256(prefix.with_suffix(".npz")) != summary["npz_sha256"]:
                raise ValueError("Physical recording changed")
            arrays = np.load(prefix.with_suffix(".npz"))
            out = args.output / prefix.name
            out.mkdir()
            restore(arrays, 0)
            print("RENDER_STATE_RESTORED", flush=True)
            for _ in range(3):
                rep.orchestrator.step(rt_subframes=4, delta_time=0.0, pause_timeline=True)
            if textures is not None and texture_probe is None:
                from texture_randomization import GROUPS

                # Preload every bank entry while the physical pose is held fixed.
                for texture_id in range(args.texture_count):
                    textures.apply(dict.fromkeys(GROUPS, texture_id))
                    images()
                fixed_ids = dict.fromkeys(GROUPS, 0)
                textures.apply(fixed_ids)
                images()
                baseline = images()
                visibility = {}
                for group in GROUPS:
                    textures.apply({**fixed_ids, group: 1})
                    images()
                    changed = images()
                    measurements = {}
                    for name in ("front", "side"):
                        difference = np.abs(changed[name].astype(float) - baseline[name].astype(float))
                        measurements[name] = {
                            "mean_absolute_change": float(difference.mean()),
                            "changed_pixels": int((difference.max(axis=2) > 5).sum()),
                        }
                    if not any(
                        item["mean_absolute_change"] > 0.05 and item["changed_pixels"] >= 50
                        for item in measurements.values()
                    ):
                        raise RuntimeError(
                            f"Texture group has no verified visible effect: {group}: {measurements}"
                        )
                    visibility[group] = measurements
                # A separate fixed-texture motion probe prevents appearance
                # changes from concealing a stale geometry/camera stream.
                textures.apply(fixed_ids)
                restore(arrays, 0)
                images()
                motion_start = images()
                transport = np.flatnonzero(arrays["stage"] == 4)
                motion_index = (
                    int(transport[len(transport) // 2]) if len(transport) else len(arrays["qpos"]) // 2
                )
                restore(arrays, motion_index)
                images()
                motion_end = images()
                motion = {
                    name: float(
                        np.abs(motion_end[name].astype(float) - motion_start[name].astype(float)).mean()
                    )
                    for name in ("front", "side")
                }
                if any(value < 1 for value in motion.values()):
                    raise RuntimeError(f"Fixed-texture motion probe failed: {motion}")
                restore(arrays, 0)
                textures.assert_physics_unchanged()
                texture_probe = {
                    "passed": True,
                    "fixed_pose_visibility": visibility,
                    "fixed_texture_motion_change": motion,
                    "motion_probe_frame": motion_index,
                    "physics_properties_unchanged": True,
                }
                write_json(args.output / "texture_probe.json", texture_probe)
                report["appearance"]["probe"] = texture_probe
            print("RENDER_CAMERAS_WARM", flush=True)
            writers = [
                cv2.VideoWriter(
                    str(out / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30 / args.stride, (640, 480)
                )
                for name in ("front", "side")
            ]
            if not all(writer.isOpened() for writer in writers):
                raise RuntimeError("Cannot open camera video writers")
            preview = cv2.VideoWriter(
                str(out / "preview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30 / args.stride, (1280, 480)
            )
            writers.append(preview)
            begin = time.monotonic()
            captured = 0
            maximum_state_error = 0.0
            maximum_pose_error = 0.0
            first_images = None
            temporal_change = dict.fromkeys(("front", "side"), 0.0)
            last_stage = None
            appearance_rows = []
            schedule = None
            if textures is not None:
                from texture_randomization import TextureSchedule

                schedule = TextureSchedule(
                    args.texture_schedule_seed
                    if args.texture_schedule_seed is not None
                    else args.texture_seed,
                    attempt_index,
                    args.texture_count,
                )
            stop = min(len(arrays["qpos"]), args.max_frames) if args.max_frames else len(arrays["qpos"])
            for index in range(0, stop, args.stride):
                maximum_pose_error = max(maximum_pose_error, restore(arrays, index))
                if schedule is not None:
                    appearance = schedule.next(index)
                    textures.apply(appearance["texture_ids"])
                frames = images()
                if schedule is not None:
                    appearance["capture_rgb_sha256"] = {
                        name: hashlib.sha256(frame.tobytes()).hexdigest() for name, frame in frames.items()
                    }
                    appearance_rows.append(appearance)
                if first_images is None:
                    first_images = frames
                for name, frame in frames.items():
                    temporal_change[name] = max(
                        temporal_change[name],
                        float(np.abs(frame.astype(float) - first_images[name]).mean()),
                    )
                error = float(
                    np.max(np.abs(robot.data.joint_pos[0, joint_ids].cpu().numpy() - arrays["qpos"][index]))
                )
                maximum_state_error = max(maximum_state_error, error)
                if error > 1e-5:
                    raise RuntimeError("Rendering replay differs from recorded observation")
                for writer, name in zip(writers[:2], ("front", "side"), strict=True):
                    writer.write(cv2.cvtColor(frames[name], cv2.COLOR_RGB2BGR))
                combined = np.concatenate([frames["front"], frames["side"]], axis=1)
                preview.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
                if textures is not None and captured < 3:
                    cv2.imwrite(
                        str(out / f"texture_frame_{captured:03d}.png"),
                        cv2.cvtColor(combined, cv2.COLOR_RGB2BGR),
                    )
                stage = int(arrays["stage"][index])
                if stage != last_stage:
                    cv2.imwrite(str(out / f"stage_{stage}.png"), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
                    last_stage = stage
                captured += 1
                if captured % 60 == 0:
                    print(
                        json.dumps(
                            {
                                "attempt": attempt_index,
                                "frames_rendered": captured,
                                "seconds": time.monotonic() - begin,
                            }
                        ),
                        flush=True,
                    )
            for writer in writers:
                writer.release()
            writers = []
            if any(value < 1 for value in temporal_change.values()):
                raise RuntimeError(f"A camera remained stale during robot motion: {temporal_change}")
            if textures is not None:
                textures.assert_physics_unchanged()
                (out / "appearance.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in appearance_rows)
                )
                if args.stride == 1:
                    from texture_randomization import validate_schedule

                    validate_schedule(appearance_rows, captured, textures.bank)
            if sha256(prefix.with_suffix(".npz")) != summary["npz_sha256"]:
                raise RuntimeError("Appearance rendering changed the physical source")
            episode = {
                "attempt": attempt_index,
                "seed": summary["seed"],
                "frames": captured,
                "physical_frames": len(arrays["qpos"]),
                "physics_npz_sha256": summary["npz_sha256"],
                "physical_success": summary["success"],
                "training_eligible": bool(
                    summary["success"] and args.stride == 1 and captured == len(arrays["qpos"])
                ),
                "appearance_frames": len(appearance_rows),
                "max_replay_joint_error_rad": maximum_state_error,
                "max_replay_world_pose_error": maximum_pose_error,
                "camera_temporal_mean_absolute_change": temporal_change,
                "seconds": time.monotonic() - begin,
                "files_sha256": {p.name: sha256(p) for p in out.iterdir() if p.is_file()},
            }
            report["episodes"].append(episode)
            write_json(args.output / "render_manifest.json", report)
        report["passed"] = bool(report["episodes"]) and all(
            e["training_eligible"] for e in report["episodes"]
        )
        write_json(args.output / "render_manifest.json", report)
    except BaseException as exc:
        traceback.print_exc()
        write_json(args.output / "failure.json", {"error": repr(exc)})
        raise
    finally:
        for writer in writers:
            writer.release()
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
