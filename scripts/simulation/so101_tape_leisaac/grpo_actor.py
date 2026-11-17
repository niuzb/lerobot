"""One FP32 SmolVLA actor/learner for live LeIsaac GRPO rollouts."""

import argparse
import json
import os
import random
import shutil
import socket
from contextlib import suppress
from pathlib import Path

import numpy as np
import torch
from common import TASK
from eval_real_loss import save, sha256
from flowvla_rl.sde import gaussian_log_prob, marginal_preserving_transition
from flowvla_rl.smolvla_sde import prepare_policy_prefix, sample_sde_chunk
from grpo_objective import advantages, clipped_loss
from grpo_wire import receive, send


class Actor:
    def __init__(self, args):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        self.args = args
        args.epochs = getattr(args, "epochs", 50)
        args.optimizer_state = getattr(args, "optimizer_state", None)
        args.checkpoint_groups = getattr(args, "checkpoint_groups", False)
        args.curriculum_sha256 = getattr(args, "curriculum_sha256", None)
        cfg = PreTrainedConfig.from_pretrained(args.checkpoint, local_files_only=True)
        cfg.device, cfg.load_vlm_weights = "cuda", False
        self.policy = (
            SmolVLAPolicy.from_pretrained(args.checkpoint, config=cfg, local_files_only=True, strict=True)
            .float()
            .eval()
        )
        self.policy.model.vlm_with_expert.vlm.requires_grad_(False)
        self.policy.model.state_proj.requires_grad_(False)
        self.parameters = [p for p in self.policy.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(self.parameters, lr=5e-6, weight_decay=0)
        self.pre, self.post = make_pre_post_processors(
            cfg, args.checkpoint, preprocessor_overrides={"device_processor": {"device": "cuda"}}
        )
        self.episodes = []
        self.current = None
        self.optimizer_steps = 0
        self.valid_groups = self.skipped_groups = 0
        self.parent_optimizer_steps = 0
        self.ancestor_sft_hash = None
        self.inherited_course_steps = 0
        self.curriculum_state = None
        self.last_checkpoint = args.optimizer_state.parent if args.optimizer_state else None
        self.last_checkpoint_steps = 0
        if args.optimizer_state is not None:
            parent = json.loads(args.parent_training_state.read_text())
            self.optimizer.load_state_dict(
                torch.load(args.optimizer_state, map_location="cuda", weights_only=True)
            )
            self.parent_optimizer_steps = parent.get("total_optimizer_steps", parent["optimizer_steps"])
            self.ancestor_sft_hash = parent["source_sft_sha256"]
            self.inherited_course_steps = parent.get(
                "course_optimizer_steps", max(0, self.parent_optimizer_steps - 2)
            )
            steps = [float(state["step"]) for state in self.optimizer.state.values() if "step" in state]
            if not steps or max(steps) != self.parent_optimizer_steps:
                raise ValueError("Restored optimizer steps do not match parent checkpoint")
        self.rng = torch.Generator(device="cuda")
        self.original_hash = sha256(args.checkpoint / "model.safetensors")
        self.version = 0
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        save(
            args.root / "actor_ready.json",
            {
                "pid": os.getpid(),
                "model_sha256": self.original_hash,
                "trainable_parameters": sum(p.numel() for p in self.parameters),
                "precision": "FP32 for both collection and rescoring",
                "kl_beta": 0,
                "lr": 5e-6,
                "eta": args.eta,
                "density_shape": [50, 32],
                "optimizer_state_restored": args.optimizer_state is not None,
                "parent_optimizer_steps": self.parent_optimizer_steps,
                "new_optimizer_steps": 0,
            },
        )

    def prepare(self, observation):
        frame = {"observation.state": torch.from_numpy(observation["state"]).float(), "task": TASK}
        for name in ["front", "side"]:
            frame["observation.images." + name] = (
                torch.from_numpy(observation[name]).permute(2, 0, 1).float() / 255
            )
        return self.pre(frame)

    def begin(self, metadata):
        if self.current is not None:
            raise RuntimeError("Previous rollout has not ended")
        self.rng.manual_seed(metadata["sample_seed"])
        self.policy.reset()
        self.current = {
            "episode": metadata["episode"],
            "epoch": metadata["epoch"],
            "group_seed": metadata["group_seed"],
            "mode": metadata["mode"],
            "task_stage": metadata.get("task_stage", "full"),
            "version": self.version,
            "chunks": [],
        }
        return {"begun": True, "policy_version": self.version}

    def act(self, observation):
        if self.current is None:
            raise RuntimeError("No active live rollout")
        prepared = self.prepare(observation)
        noise = torch.randn((1, 50, 32), generator=self.rng, device="cuda")
        with torch.no_grad():
            if self.current["mode"] == "ode":
                normalized = self.policy.predict_action_chunk(prepared, noise=noise)
                trajectory = None
            else:
                masks, cache = prepare_policy_prefix(self.policy, prepared)
                # The full latent path is a joint policy action. Ignored actuator
                # dimensions/positions still influence later denoising states.
                trajectory = sample_sde_chunk(
                    self.policy.model,
                    masks,
                    cache,
                    action_dim=32,
                    eta=self.args.eta,
                    noise=noise,
                    generator=self.rng,
                )
                normalized = trajectory.actions[..., :6]
            action = self.post(normalized).cpu().numpy()[0].astype(np.float32)
        if action.shape != (50, 6) or not np.isfinite(action).all():
            raise ValueError("Non-finite live policy action")
        if trajectory is not None:
            self.current["chunks"].append({"observation": observation, "trajectory": trajectory.to("cpu")})
        return action

    def end(self, metadata):
        episode = self.current
        if episode is None or episode["episode"] != metadata["episode"]:
            raise RuntimeError("Rollout identity mismatch")
        episode["reward"] = int(metadata["success"])
        episode["control_steps"] = metadata["control_steps"]
        if episode["mode"] == "sde":
            folder = (
                self.args.root
                / "traces"
                / f"epoch_{episode['epoch']:03d}"
                / f"episode_{episode['episode']:04d}"
            )
            folder.mkdir(parents=True, exist_ok=False)
            for index, chunk in enumerate(episode["chunks"]):
                trace = chunk["trajectory"]
                np.savez_compressed(
                    folder / f"chunk_{index:03d}.npz",
                    **chunk["observation"],
                    states=trace.states.numpy(),
                    element_log_probs=trace.element_log_probs.numpy(),
                )
            save(folder / "outcome.json", {k: v for k, v in episode.items() if k != "chunks"})
            self.episodes.append(episode)
        self.current = None
        return {
            "reward": episode["reward"],
            "group_episodes": len(self.episodes),
            "optimizer_steps": self.optimizer_steps,
        }

    def update(self, epoch, curriculum_state=None):
        if (
            len(self.episodes) != 8
            or len({e["group_seed"] for e in self.episodes}) != 1
            or {e["version"] for e in self.episodes} != {self.version}
            or len({e["task_stage"] for e in self.episodes}) != 1
        ):
            raise ValueError("GRPO requires eight reset-matched behavior-policy episodes")
        rewards = [e["reward"] for e in self.episodes]
        task_stage = self.episodes[0]["task_stage"]
        self.curriculum_state = curriculum_state
        advantage = advantages(rewards)
        if advantage is None:
            self.skipped_groups += 1
            self.episodes.clear()
            result = {
                "epoch": epoch,
                "rewards": rewards,
                "skipped": True,
                "reason": "uniform_binary_rewards",
                "optimizer_steps": self.optimizer_steps,
                "task_stage": task_stage,
            }
            save(self.args.root / "learner_status.json", result)
            if self.args.checkpoint_groups:
                self.checkpoint(epoch)
            return result
        self.valid_groups += 1
        jobs = [
            (e, chunk, float(advantage[i]), 1 / (8 * len(e["chunks"])))
            for i, e in enumerate(self.episodes)
            for chunk in e["chunks"]
        ]
        if not jobs:
            raise ValueError("Mixed rewards without any live action traces")
        first_ratio_drift = None
        metrics = []
        for ppo_pass in range(2):
            self.optimizer.zero_grad(set_to_none=True)
            order = list(range(len(jobs)))
            random.Random(20261110 + epoch * 17 + ppo_pass).shuffle(order)
            for completed, index in enumerate(order, 1):
                episode, chunk, adv, weight = jobs[index]
                with torch.no_grad():
                    prepared = self.prepare(chunk["observation"])
                    masks, cache = prepare_policy_prefix(self.policy, prepared)
                trace = chunk["trajectory"].to("cuda")
                # Uniformly sample a denoising transition: unbiased sampling of
                # the mean transition objective while retaining ten-step rollout.
                k = random.Random(epoch * 100003 + index * 13 + ppo_pass).randrange(10)
                current_time = 1 - k / 10
                velocity = self.policy.model.denoise_step(
                    prefix_pad_masks=masks,
                    past_key_values=cache,
                    x_t=trace.states[:, k],
                    timestep=torch.tensor([current_time], device="cuda"),
                )
                transition = marginal_preserving_transition(
                    trace.states[:, k], velocity, current_time, 0.1, self.args.eta
                )
                logp = gaussian_log_prob(trace.states[:, k + 1], transition.mean, transition.std)
                old_logp = trace.log_probs[:, k]
                loss, detail = clipped_loss(logp, old_logp, adv)
                if ppo_pass == 0 and completed == 1:
                    first_ratio_drift = detail["ratio_max_drift"]
                    if first_ratio_drift > 0.05:
                        raise RuntimeError("Unchanged-policy rescore ratio is not one")
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite GRPO objective")
                (loss * weight).backward()
                if completed % 10 == 0:
                    save(
                        self.args.root / "learner_status.json",
                        {
                            "stage": "backward_accumulation",
                            "epoch": epoch,
                            "ppo_pass": ppo_pass,
                            "chunks_done": completed,
                            "chunks_total": len(jobs),
                            "optimizer_steps": self.optimizer_steps,
                            "rewards": rewards,
                        },
                    )
            norm = torch.nn.utils.clip_grad_norm_(self.parameters, 1.0)
            if not torch.isfinite(norm):
                raise ValueError("No finite nonzero RL gradient")
            if float(norm) <= 0:
                if ppo_pass == 0:
                    raise ValueError("No nonzero gradient in the initial on-policy pass")
                metrics.append({"later_pass_skipped": "clipped_surrogate_zero_gradient"})
                break
            active = [p for p in self.parameters if p.grad is not None][:3]
            before = [p.detach().clone() for p in active]
            self.optimizer.step()
            delta = (
                sum(float((p.detach() - b).square().sum()) for p, b in zip(active, before, strict=True))
                ** 0.5
            )
            if delta <= 0:
                raise ValueError("Optimizer did not change actor parameters")
            self.optimizer_steps += 1
            metrics.append({"gradient_norm_before_clip": float(norm), "sampled_parameter_delta_l2": delta})
        self.version += 1
        self.episodes.clear()
        result = {
            "stage": "optimizer_updated",
            "epoch": epoch,
            "rewards": rewards,
            "advantages": advantage.tolist(),
            "skipped": False,
            "optimizer_steps": self.optimizer_steps,
            "first_ratio_max_drift": first_ratio_drift,
            "gradient_updates": metrics,
            "policy_version": self.version,
            "kl_beta": 0,
            "task_stage": task_stage,
            "parent_optimizer_steps": self.parent_optimizer_steps,
            "total_optimizer_steps": self.parent_optimizer_steps + self.optimizer_steps,
        }
        save(self.args.root / "learner_status.json", result)
        if (
            self.args.checkpoint_groups
            or self.valid_groups == 1
            or epoch % 5 == 0
            or epoch == self.args.epochs
        ):
            self.checkpoint(epoch)
        print(json.dumps(result), flush=True)
        return result

    def checkpoint(self, epoch, final=False):
        target = self.args.root / "checkpoints" / ("final" if final else f"epoch_{epoch:03d}")
        target.mkdir(parents=True, exist_ok=False)
        if self.last_checkpoint is not None and self.optimizer_steps == self.last_checkpoint_steps:
            # Immutable files can be shared across skipped groups without copying
            # gigabytes or ever rewriting a parent checkpoint.
            shutil.copytree(
                self.last_checkpoint / "pretrained_model", target / "pretrained_model", copy_function=os.link
            )
            os.link(self.last_checkpoint / "optimizer.pt", target / "optimizer.pt")
        else:
            self.policy.save_pretrained(target / "pretrained_model")
            self.pre.save_pretrained(target / "pretrained_model")
            self.post.save_pretrained(target / "pretrained_model")
            torch.save(self.optimizer.state_dict(), target / "optimizer.pt")
        save(
            target / "training_state.json",
            {
                "epoch": epoch,
                "optimizer_steps": self.optimizer_steps,
                "valid_groups": self.valid_groups,
                "skipped_groups": self.skipped_groups,
                "policy_version": self.version,
                "source_sft_sha256": self.ancestor_sft_hash or self.original_hash,
                "source_checkpoint_sha256": self.original_hash,
                "parent_optimizer_steps": self.parent_optimizer_steps,
                "total_optimizer_steps": self.parent_optimizer_steps + self.optimizer_steps,
                "course_optimizer_steps": self.inherited_course_steps + self.optimizer_steps,
                "curriculum_state": self.curriculum_state,
                "curriculum_sha256": self.args.curriculum_sha256,
            },
        )
        self.last_checkpoint = target
        self.last_checkpoint_steps = self.optimizer_steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--eta", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--optimizer-state", type=Path)
    parser.add_argument("--parent-training-state", type=Path)
    parser.add_argument("--checkpoint-groups", action="store_true")
    parser.add_argument("--curriculum-sha256")
    args = parser.parse_args()
    if (args.optimizer_state is None) != (args.parent_training_state is None):
        parser.error("Optimizer and parent training state must be supplied together")
    torch.set_num_threads(4)
    actor = Actor(args)
    if args.socket.exists():
        raise FileExistsError("Existing actor socket must be inspected before reuse")
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(args.socket))
    os.chmod(args.socket, 0o600)
    server.listen(1)
    connection, _ = server.accept()
    try:
        while True:
            metadata, observation = receive(connection)
            op = metadata["operation"]
            if op == "begin":
                send(connection, actor.begin(metadata))
            elif op == "act":
                send(connection, {"policy_version": actor.version}, action=actor.act(observation))
            elif op == "end":
                send(connection, actor.end(metadata))
            elif op == "update":
                send(connection, actor.update(metadata["epoch"], metadata.get("curriculum_state")))
            elif op == "finish":
                assert sha256(args.checkpoint / "model.safetensors") == actor.original_hash
                if actor.optimizer_steps or actor.parent_optimizer_steps:
                    actor.checkpoint(metadata.get("epoch", args.epochs), final=True)
                save(
                    args.root / "actor_finished.json",
                    {
                        "optimizer_steps": actor.optimizer_steps,
                        "valid_groups": actor.valid_groups,
                        "skipped_groups": actor.skipped_groups,
                        "parent_optimizer_steps": actor.parent_optimizer_steps,
                        "total_optimizer_steps": actor.parent_optimizer_steps + actor.optimizer_steps,
                    },
                )
                send(
                    connection,
                    {
                        "optimizer_steps": actor.optimizer_steps,
                        "valid_groups": actor.valid_groups,
                        "skipped_groups": actor.skipped_groups,
                        "parent_optimizer_steps": actor.parent_optimizer_steps,
                        "total_optimizer_steps": actor.parent_optimizer_steps + actor.optimizer_steps,
                    },
                )
                break
            else:
                raise ValueError("Unknown actor operation")
    except BaseException as error:
        save(
            args.root / "actor_failure.json", {"error": repr(error), "optimizer_steps": actor.optimizer_steps}
        )
        with suppress(OSError):
            send(connection, {"error": str(error)})
        raise
    finally:
        connection.close()
        server.close()
        args.socket.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
