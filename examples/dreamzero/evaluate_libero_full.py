"""Deterministic closed-loop LIBERO selection and final evaluation for DreamZero."""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from .libero_full_utils import file_hash, sim_action, write_json

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
MAX_STEPS = {suite: 700 if suite == "libero_10" else 400 for suite in SUITES}
WAIT_STEPS = 30
RESOLUTION = 256


def _seed(seed):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _axis_angle(quaternion):
    quat = np.asarray(quaternion, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(max(0.0, 1.0 - quat[3] * quat[3]))
    if math.isclose(denominator, 0.0):
        return np.zeros(3)
    return quat[:3] * 2 * math.acos(quat[3]) / denominator


def _observation(obs, task, camera_keys):
    if len(camera_keys) != 2:
        raise ValueError(f"LIBERO evaluation requires two camera features, got {camera_keys}")
    cameras = (obs["agentview_image"], obs["robot0_eye_in_hand_image"])
    state = np.concatenate(
        (obs["robot0_eef_pos"], _axis_angle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
    ).astype(np.float32)
    if state.shape != (8,) or not np.isfinite(state).all():
        raise FloatingPointError("LIBERO produced invalid 8D robot state")
    return {
        **{key: torch.from_numpy(np.ascontiguousarray(image[::-1, ::-1])).permute(2, 0, 1)
           for key, image in zip(camera_keys, cameras, strict=True)},
        "observation.state": torch.from_numpy(state),
        "task": task,
    }


def _action(raw):
    action = sim_action(raw)
    if not np.isfinite(action).all():
        raise FloatingPointError("Policy emitted a nonfinite LIBERO action")
    return action


def _task(suite_name, task_id):
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    states_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    states = torch.load(states_path, map_location="cpu", weights_only=False)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    return task, states, bddl


def _run_episode(env, initial_state, task, policy, preprocessor, postprocessor, suite, task_id, trial):
    seed = 42 + SUITES.index(suite) * 100000 + task_id * 1000 + trial
    _seed(seed)
    env.seed(42)
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    env.reset()
    obs = env.set_init_state(initial_state)
    for _ in range(WAIT_STEPS):
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
    success, steps = False, 0
    started = time.monotonic()
    while steps < MAX_STEPS[suite]:
        batch = preprocessor(_observation(obs, task.language, policy.config.camera_keys))
        with torch.inference_mode():
            normalized = policy.select_action(batch)
            raw = postprocessor(normalized).detach().cpu().float().numpy()[0]
        obs, _, done, _ = env.step(_action(raw).tolist())
        steps += 1
        success = bool(env.check_success())
        if success:
            break
        if done:
            raise RuntimeError(f"{suite} task {task_id} terminated without success")
    return {
        "suite": suite,
        "task_id": task_id,
        "trial": trial,
        "success": success,
        "steps": steps,
        "max_steps": MAX_STEPS[suite],
        "episode_policy_seed": seed,
        "env_seed": 42,
        "initial_state_index": trial,
        "elapsed_s": time.monotonic() - started,
    }


def _aggregate(records, complete, update):
    suites = {}
    for suite in SUITES:
        rows = [row for row in records if row["suite"] == suite]
        successes = sum(row["success"] for row in rows)
        suites[suite] = {
            "successes": successes,
            "episodes": len(rows),
            "success_rate": successes / len(rows) if rows else 0.0,
        }
    successes = sum(row["success"] for row in records)
    return {
        "complete": complete,
        "update": update,
        "episodes": len(records),
        "successes": successes,
        "success_rate": successes / len(records) if records else 0.0,
        "suites": suites,
        "records": records,
    }


def evaluate_selection(policy, preprocessor, postprocessor, root, update, protocol_manifest=None):
    """Run task ids 0,1 × initial states 0..4 in each suite (40 episodes)."""
    records = []
    for suite in SUITES:
        for task_id in (0, 1):
            task, states, bddl = _task(suite, task_id)
            if len(states) < 5:
                raise ValueError(f"Only {len(states)} initial states for {suite} task {task_id}")
            env = OffScreenRenderEnv(
                bddl_file_name=str(bddl), camera_heights=RESOLUTION, camera_widths=RESOLUTION,
                control_freq=20, hard_reset=True,
            )
            try:
                for trial in range(5):
                    records.append(_run_episode(env, states[trial], task, policy, preprocessor,
                                                postprocessor, suite, task_id, trial))
            finally:
                env.close()
    for row in records:
        row["used_for_candidate_selection"] = True
    result = _aggregate(records, complete=True, update=update)
    if result["episodes"] != 40 or len(result["suites"]) != 4:
        raise RuntimeError("LIBERO selection evaluation is incomplete")
    path = Path(root) / "closed_loop" / f"selection_update_{update:06d}.json"
    result["protocol_fingerprint"] = protocol_fingerprint(protocol_manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, result)
    return result


def evaluate_full(policy, preprocessor, postprocessor, root, update, selection_result=None, protocol_manifest=None):
    """Run remaining LIBERO episodes and report selection and holdout separately."""
    selected = list((selection_result or {}).get("records", []))
    remainder = []
    selected_ids = {(s, task, trial) for s in SUITES for task in (0, 1) for trial in range(5)}
    for suite in SUITES:
        for task_id in range(10):
            task, states, bddl = _task(suite, task_id)
            if len(states) < 50:
                raise ValueError(f"Only {len(states)} initial states for {suite} task {task_id}")
            env = OffScreenRenderEnv(
                bddl_file_name=str(bddl), camera_heights=RESOLUTION, camera_widths=RESOLUTION,
                control_freq=20, hard_reset=True,
            )
            try:
                for trial in range(50):
                    if (suite, task_id, trial) in selected_ids:
                        continue
                    row = _run_episode(env, states[trial], task, policy, preprocessor,
                                       postprocessor, suite, task_id, trial)
                    row["used_for_candidate_selection"] = False
                    remainder.append(row)
            finally:
                env.close()
    result = {
        "complete": len(selected) == 40 and len(remainder) == 1960,
        "update": update,
        "episodes": len(selected) + len(remainder),
        "selected_40": _aggregate(selected, len(selected) == 40, update),
        "remaining_1960": _aggregate(remainder, len(remainder) == 1960, update),
        "all_2000": _aggregate(selected + remainder, len(selected) == 40 and len(remainder) == 1960, update),
        "protocol_fingerprint": protocol_fingerprint(protocol_manifest),
    }
    if not result["complete"]:
        raise RuntimeError(f"Full LIBERO evaluation incomplete: {len(selected) + len(remainder)} / 2000")
    path = Path(root) / "closed_loop" / f"full_update_{update:06d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, result)
    return result


def protocol_fingerprint(manifest_path):
    """Bind each score report to a fixed, previously audited LIBERO asset manifest."""
    if manifest_path is None:
        raise ValueError("Pass the audited --protocol-manifest used by the previous LIBERO evaluation")
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text())
    return {"manifest": str(path), "sha256": file_hash(path), "protocol": manifest.get("protocol", manifest)}
