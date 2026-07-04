#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Train a policy.

Requires: pip install 'lerobot[training]'  (includes dataset + accelerate + wandb extras)
"""

import dataclasses
import logging
import sys
import time
from contextlib import nullcontext
from pprint import pformat
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from accelerate import Accelerator

import torch
from termcolor import colored
from torch.optim import Optimizer
from tqdm import tqdm

from lerobot.common.train_utils import (
    gather_fsdp_state_dicts,
    get_step_checkpoint_dir,
    get_step_identifier,
    load_fsdp_optimizer_state,
    load_training_batch_size,
    load_training_num_processes,
    load_training_state,
    push_checkpoint_to_hub,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.wandb_utils import WandBLogger
from lerobot.configs import JobConfig, parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets import EpisodeAwareSampler, compute_sampler_state
from lerobot.datasets.factory import make_train_eval_datasets
from lerobot.envs import close_envs, make_env, make_env_pre_post_processors
from lerobot.jobs import submit_to_hf
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies import PreTrainedPolicy, make_policy, make_pre_post_processors
from lerobot.rewards import make_reward_pre_post_processors
from lerobot.utils.collate import lerobot_collate_fn
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import (
    cycle,
    format_big_number,
    has_method,
    init_logging,
    inside_slurm,
)

from .lerobot_eval import eval_policy_all


# =============================================================================
# 中文导读
# =============================================================================
# 这个文件是 `lerobot-train` 命令的主入口，负责把“训练一套 policy/reward model”
# 所需的各个子系统串起来。它本身不实现某个具体神经网络结构，而是训练流程编排器。
#
# 一次典型的离线训练会经过：
# 1. parser.wrap() 解析 CLI/config，构造 TrainPipelineConfig；
# 2. 创建 Accelerator，统一 CPU/GPU、多卡、混合精度、DDP/FSDP 等训练差异；
# 3. 创建 train/eval dataset；
# 4. 创建 policy 或 reward model；
# 5. 创建 preprocessor/postprocessor；
# 6. 创建 optimizer/scheduler；
# 7. 创建 DataLoader，并交给 accelerator.prepare() 包装；
# 8. 在主训练循环里执行 batch -> preprocessor -> forward/backward -> optimizer.step；
# 9. 按频率记录日志、跑 held-out eval loss、保存 checkpoint、跑 env rollout eval；
# 10. 训练结束后可选 push_to_hub，并清理分布式进程组。
#
# 阅读路线建议：
# - 先看 train() 的大段流程；
# - 再看 update_policy() 的单步参数更新；
# - 最后看 main() 和 _remote_target_in_argv() 如何处理 CLI/远程 job。


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: "Accelerator",
    lr_scheduler=None,
    lock=None,
    sample_weighter=None,
) -> tuple[MetricsTracker, dict | None]:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler. Accelerator handles mixed-precision training automatically.

    Args:
        train_metrics: A MetricsTracker instance to record training statistics.
        policy: The policy model to be trained.
        batch: A batch of training data.
        optimizer: The optimizer used to update the policy's parameters.
        grad_clip_norm: The maximum norm for gradient clipping.
        accelerator: The Accelerator instance for distributed training and mixed precision.
        lr_scheduler: An optional learning rate scheduler.
        lock: An optional lock for thread-safe optimizer updates.
        sample_weighter: Optional SampleWeighter instance for per-sample loss weighting.

    Returns:
        A tuple containing:
        - The updated MetricsTracker with new statistics for this step.
        - A dictionary of outputs from the policy's forward pass, for logging purposes.
    """
    # 这里定义的是“一次参数更新”的最小闭环：
    # 前向计算 loss -> 反向传播 -> 梯度裁剪 -> optimizer.step -> scheduler.step。
    start_time = time.perf_counter()

    # 进入训练模式，启用 dropout、训练态 normalization 等模块行为。
    policy.train()

    if torch.cuda.is_available():
        # 每一步开始前重置 CUDA 峰值显存统计，后面可以记录这一 step 的峰值显存。
        torch.cuda.reset_peak_memory_stats()

    # Compute sample weights if a weighter is provided
    # sample_weighter 是一个通用的“样本加权器”接口。
    # 例如 RA-BC 会根据样本的重要性/阶段进度给不同样本不同 loss 权重。
    sample_weights = None
    weight_stats = None
    if sample_weighter is not None:
        # 返回值一般包含：
        # - sample_weights：当前 batch 中每个样本的权重；
        # - weight_stats：用于日志记录的权重分布统计。
        sample_weights, weight_stats = sample_weighter.compute_batch_weights(batch)

    # Let accelerator handle mixed precision
    # accelerator.autocast() 会按照 Accelerator 的 mixed_precision 设置自动进入
    # fp16/bf16/no autocast 上下文，避免手写 torch.cuda.amp.autocast。
    with accelerator.autocast():
        if sample_weights is not None:
            # Use per-sample loss for weighted training
            # Note: Policies supporting sample weighting must implement forward(batch, reduction="none")
            # 样本加权要求 policy 返回“每个样本自己的 loss”，所以 reduction 必须是 none。
            # 如果某个 policy 不支持这个约定，开启 sample weighting 时会在这里暴露问题。
            per_sample_loss, output_dict = policy.forward(batch, reduction="none")

            # Weighted loss: each sample's contribution is scaled by its weight.
            # We divide by weight sum (not batch size) so that if some weights are zero,
            # the remaining samples contribute proportionally more, preserving gradient scale.
            # Weights are pre-normalized to sum to batch_size for stable training dynamics.
            # 加权 loss 的分母用 weight sum，而不是固定 batch size。
            # 这样当部分样本权重为 0 时，其余样本不会被整体稀释。
            epsilon = 1e-6
            loss = (per_sample_loss * sample_weights).sum() / (sample_weights.sum() + epsilon)

            # Log weighting statistics
            # output_dict 允许 policy 或 sample_weighter 把额外指标带到日志系统。
            if output_dict is None:
                output_dict = {}
            for key, value in weight_stats.items():
                output_dict[f"sample_weight_{key}"] = value
        else:
            # 普通训练路径：policy.forward 直接返回标量 loss 和可选日志字典。
            loss, output_dict = policy.forward(batch)

        # TODO(rcadene): policy.unnormalize_outputs(out_dict)

    # Use accelerator's backward method
    # 不直接 loss.backward()，因为 accelerator.backward 会处理混合精度、梯度缩放、
    # DDP/FSDP 等场景下需要的同步细节。
    accelerator.backward(loss)

    # Clip gradients if specified
    # grad_clip_norm > 0 时执行真正的梯度裁剪；否则用 inf 只计算 grad norm。
    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float("inf"), error_if_nonfinite=False
        )

    # Optimizer step
    # lock 用于兼容可能存在多线程/服务式更新的场景；普通离线训练一般是 nullcontext。
    with lock if lock is not None else nullcontext():
        optimizer.step()

    # PyTorch 默认梯度会累积，所以每次更新后必须清空梯度。
    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    # 这里的学习率调度按“训练 step”推进，而不是按 epoch 推进。
    if lr_scheduler is not None:
        lr_scheduler.step()

    # Update internal buffers if policy has update method
    # 某些 policy 有额外内部状态，例如 EMA target network 或 action queue 缓冲；
    # 如果实现了 update()，就在 optimizer.step() 之后同步它们。
    if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    # 把本 step 的核心指标写回 MetricsTracker，后面统一 reduce/log。
    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    if torch.cuda.is_available():
        # 记录本 step 的峰值显存，单位 GB。
        train_metrics.gpu_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
    return train_metrics, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig, accelerator: "Accelerator | None" = None):
    """
    Main function to train a policy.

    This function orchestrates the entire training pipeline, including:
    - Setting up logging, seeding, and device configuration.
    - Creating the dataset, evaluation environment (if applicable), policy, and optimizer.
    - Handling resumption from a checkpoint.
    - Running the main training loop, which involves fetching data batches and calling `update_policy`.
    - Periodically logging metrics, saving model checkpoints, and evaluating the policy.
    - Pushing the final trained model to the Hugging Face Hub if configured.

    Args:
        cfg: A `TrainPipelineConfig` object containing all training configurations.
        accelerator: Optional Accelerator instance. If None, one will be created automatically.
    """
    # 如果用户通过 --job.target 指定远程 HF Jobs，这里直接提交远程任务，
    # 当前本地进程不再继续创建 dataset/model/optimizer。
    if cfg.job.is_remote:
        return submit_to_hf(cfg)

    # accelerate 是训练 extra 依赖；这里运行时检查，错误信息会提示安装 lerobot[training]。
    from lerobot.utils.import_utils import require_package

    require_package("accelerate", extra="training")
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs, DistributedType

    # 配置自检：检查 dataset/policy/reward model/env/训练步数等组合是否合理。
    cfg.validate()

    # Create Accelerator if not provided
    # It will automatically detect if running in distributed mode or single-process mode
    # We set step_scheduler_with_optimizer=False to prevent accelerate from adjusting the lr_scheduler steps based on the num_processes
    # We set find_unused_parameters=True to handle models with conditional computation
    if accelerator is None:
        # DDP 下有些模型存在条件分支，不是每次 forward 都会用到所有参数。
        # find_unused_parameters=True 可以避免这类模型在 DDP 同步梯度时报错。
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        # Accelerate auto-detects the device based on the available hardware and ignores the policy.device setting.
        # Force the device to be CPU when the active config's device is set to CPU (works for both policy and reward model training).
        # trainable_config 是“当前真正要训练的对象”的配置：
        # 普通 policy 训练时是 cfg.policy，reward model 训练时是 cfg.reward_model。
        force_cpu = cfg.trainable_config.device == "cpu"
        # Drive Accelerate's autocast from policy.dtype (bf16/fp16 activate it; float32/absent -> launcher default).
        # dtype 映射到 accelerate 的 mixed_precision：
        # bfloat16 -> bf16, float16 -> fp16, float32 -> no。
        policy_dtype = getattr(cfg.trainable_config, "dtype", None)
        mixed_precision = {"bfloat16": "bf16", "float16": "fp16", "float32": "no"}.get(policy_dtype)
        accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            mixed_precision=mixed_precision,
            kwargs_handlers=[ddp_kwargs],
            cpu=force_cpu,
        )

    # 初始化日志系统；传入 accelerator 后，多进程训练时日志输出会更干净。
    init_logging(accelerator=accelerator)

    # Determine if this is the main process (for logging and checkpointing)
    # When using accelerate, only the main process should log to avoid duplicate outputs
    # 多卡/多进程训练时，只有主进程负责打印配置、保存 checkpoint、上传、env eval 等副作用。
    is_main_process = accelerator.is_main_process

    # Only log on main process
    # 打印最终配置，便于复现实验。
    if is_main_process:
        logging.info(pformat(cfg.to_dict()))

    # Initialize wandb only on main process
    # W&B 只在主进程初始化，避免每个 rank 都创建一个 run。
    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        # 固定随机种子；accelerator 参数用于让分布式环境的随机状态也按约定设置。
        set_seed(cfg.seed, accelerator=accelerator)

    # Use accelerator's device
    # 后续以 accelerator.device 为准，而不是直接读 cfg.policy.device。
    device = accelerator.device
    if cfg.cudnn_deterministic:
        # deterministic=True 更利于复现，但通常会牺牲一些性能。
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # benchmark=True 会为固定 shape 的卷积选择更快算法。
        torch.backends.cudnn.benchmark = True
    # 允许 NVIDIA Ampere+ GPU 使用 TF32 矩阵乘法，通常能提速且精度影响较小。
    torch.backends.cuda.matmul.allow_tf32 = True

    # Dataset loading synchronization: the global main process downloads once to the shared
    # dataset root, then a barrier lets every other rank read the already-populated copy.
    # LeRobotDataset skips its snapshot_download when try_load() succeeds, so no rank re-downloads.
    if is_main_process:
        logging.info("Creating dataset")
        # make_train_eval_datasets 同时返回训练集和可选 held-out eval dataset。
        # 如果数据集需要从 Hub 下载，只让主进程先下载，避免多进程抢同一缓存目录。
        dataset, eval_dataset = make_train_eval_datasets(cfg)

    # 等待主进程完成数据集准备，其它 rank 再继续。
    accelerator.wait_for_everyone()

    # Other ranks read from the shared copy populated by the main process.
    if not is_main_process:
        # 非主进程通常会命中本地缓存/共享目录，不再重复下载。
        dataset, eval_dataset = make_train_eval_datasets(cfg)

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.env_eval_freq > 0 and cfg.env is not None and is_main_process:
        logging.info("Creating env")
        # env_eval_freq 控制“在线 rollout 评估”的频率。
        # 这和下面的 eval_dataloader loss 评估不同：这里是真的把 policy 放进环境里跑。
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    if cfg.is_reward_model_training:
        # 当前训练目标是 reward model，而不是 robot policy。
        if is_main_process:
            logging.info("Creating reward model")
        from lerobot.rewards import make_reward_model

        # reward model 需要 dataset stats/meta 来确定输入 feature、归一化和任务相关信息。
        policy = make_reward_model(
            cfg=cfg.reward_model,
            dataset_stats=dataset.meta.stats,
            dataset_meta=dataset.meta,
        )
        if not policy.is_trainable:
            raise ValueError(
                f"Reward model '{policy.name}' is zero-shot and cannot be trained via lerobot-train. "
                "Use it directly for inference via compute_reward() (e.g. offline precompute)."
            )
    else:
        # 当前训练目标是普通策略 policy，例如 ACT / Diffusion / SmolVLA 等。
        if is_main_process:
            logging.info("Creating policy")
        policy = make_policy(
            cfg=cfg.policy,
            ds_meta=dataset.meta,
            rename_map=cfg.rename_map,
        )

    if cfg.peft is not None:
        if cfg.is_reward_model_training:
            raise ValueError("PEFT is only supported for policy training. ")
        from peft import PeftModel

        if isinstance(policy, PeftModel):
            # 如果 checkpoint 自己已经带了 PEFT adapter，就不要重复包一层。
            logging.info("PEFT adapter already loaded from checkpoint, skipping wrap_with_peft.")
        else:
            logging.info("Using PEFT! Wrapping model.")
            # dataclass -> dict，交给具体 policy 去创建 LoRA/adapter 等 PEFT 结构。
            peft_cli_overrides = dataclasses.asdict(cfg.peft)
            policy = policy.wrap_with_peft(peft_cli_overrides=peft_cli_overrides)

    # Wait for all processes to finish model creation before continuing
    # 等所有 rank 都创建完模型，再继续创建 processor/optimizer。
    accelerator.wait_for_everyone()

    # active_cfg 指向当前训练对象的配置：
    # - policy 训练：cfg.policy；
    # - reward model 训练：cfg.reward_model。
    active_cfg = cfg.trainable_config
    processor_pretrained_path = active_cfg.pretrained_path

    # processor_kwargs 用来给 pre/post processor 注入 dataset stats/meta 或覆盖项。
    processor_kwargs = {}
    if (processor_pretrained_path and not cfg.resume) or not processor_pretrained_path:
        # 非 resume 情况下，用当前数据集统计量创建/覆盖 normalizer。
        # resume 时通常优先使用 checkpoint 中保存的 processor 状态。
        processor_kwargs["dataset_stats"] = dataset.meta.stats

    if cfg.is_reward_model_training:
        # reward model processor 需要完整 dataset_meta 来理解 reward 输入结构。
        processor_kwargs["dataset_meta"] = dataset.meta

    if not cfg.is_reward_model_training and processor_pretrained_path is not None:
        # 从预训练 policy 微调时，模型权重来自 pretrained_path，
        # 但归一化统计、设备、feature rename 应该匹配当前训练数据集。
        preprocessor_overrides = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        }
        postprocessor_overrides = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }
        if getattr(active_cfg, "use_relative_actions", False):
            # 某些 policy 训练时使用相对动作，推理/回放时需要再转换回绝对动作。
            preprocessor_overrides["relative_actions_processor"] = {
                "enabled": True,
                "exclude_joints": getattr(active_cfg, "relative_exclude_joints", []),
                "action_names": getattr(active_cfg, "action_feature_names", None),
            }
            postprocessor_overrides["absolute_actions_processor"] = {"enabled": True}
        processor_kwargs["preprocessor_overrides"] = preprocessor_overrides
        processor_kwargs["postprocessor_overrides"] = postprocessor_overrides

    if cfg.is_reward_model_training:
        # reward model 有自己的 processor 构造逻辑。
        preprocessor, postprocessor = make_reward_pre_post_processors(
            cfg.reward_model,
            **processor_kwargs,
        )
    else:
        # policy 的 processor 会根据 policy 类型选择不同流水线。
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=processor_pretrained_path,
            pretrained_revision=getattr(cfg.policy, "pretrained_revision", None),
            **processor_kwargs,
        )

    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    # 根据 cfg.optimizer/cfg.scheduler 创建优化器和学习率调度器。
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    # Create sample weighter if configured (e.g., for RA-BC training)
    sample_weighter = None
    if cfg.sample_weighting is not None:
        # sample weighting 是对每个样本的 loss 乘权重，常见用途包括 RA-BC。
        from lerobot.utils.sample_weighting import make_sample_weighter

        if is_main_process:
            logging.info(f"Creating sample weighter: {cfg.sample_weighting.type}")
        # sample_weighter 需要知道 policy、device 和 dataset 位置，以便加载外部权重/progress 文件。
        sample_weighter = make_sample_weighter(
            cfg.sample_weighting,
            policy,
            device,
            dataset_root=cfg.dataset.root,
            dataset_repo_id=cfg.dataset.repo_id,
        )

    # step 表示已经完成的参数更新次数，不是 epoch，也不是 batch 在 dataloader 里的索引。
    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        # Under FSDP the optimizer state is sharded and must be loaded after `accelerator.prepare()`
        # (see load_fsdp_optimizer_state below), so skip the optimizer here and load it then.
        # FSDP 下 optimizer state 是分片的，模型/优化器必须先经过 accelerator.prepare()
        # 包装成 FSDP 形态后才能正确加载；普通 DDP/单卡可以在这里直接加载。
        is_fsdp = accelerator.distributed_type == DistributedType.FSDP
        step, optimizer, lr_scheduler = load_training_state(
            cfg.checkpoint_path, optimizer, lr_scheduler, load_optimizer=not is_fsdp
        )

    # 参数量统计：
    # - learnable params：真正会被优化器更新的参数；
    # - total params：模型总参数量。
    # PEFT/LoRA 场景下二者差距通常很大。
    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
            logging.info("Creating environment processors")
            # env processor 用于仿真 rollout 评估时，把 env observation/action 和 policy 格式互相转换。
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(
                env_cfg=cfg.env, policy_cfg=cfg.policy
            )
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        num_processes = accelerator.num_processes
        effective_bs = cfg.batch_size * num_processes
        logging.info(f"Effective batch size: {cfg.batch_size} x {num_processes} = {effective_bs}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    if not cfg.dataset.streaming:
        # All non-streaming (map-style) datasets use EpisodeAwareSampler.
        # The order is a pure function of (seed, epoch), so every rank independently produces the
        # same permutation. accelerate then shards it disjointly across ranks via BatchSamplerShard
        # without needing a `generator` attribute to synchronize an RNG, and resume is sample-exact.
        # 非 streaming 数据集是 map-style dataset，可以精确按 index 采样。
        # EpisodeAwareSampler 知道 episode 边界，能避免采到 episode 尾部未来帧不够的样本。
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=getattr(active_cfg, "drop_n_last_frames", 0),
            shuffle=True,
            seed=cfg.seed if cfg.seed is not None else 0,
            absolute_to_relative_idx=dataset.absolute_to_relative_idx,
        )
        if cfg.resume and step > 0:
            # The resume offset depends on the (num_processes, batch_size) that produced `step`, so
            # use the values recorded in the checkpoint (falling back to the current ones for older
            # ckpts that did not store them).
            # 恢复训练时，不仅要恢复模型/优化器，还要尽量恢复“数据采样进度”。
            # 这个进度依赖当时保存 checkpoint 时的 world size 和 batch size。
            saved_num_processes = load_training_num_processes(cfg.checkpoint_path)
            saved_batch_size = load_training_batch_size(cfg.checkpoint_path)
            ckpt_num_processes = saved_num_processes or accelerator.num_processes
            ckpt_batch_size = saved_batch_size or cfg.batch_size
            if is_main_process and saved_num_processes not in (None, accelerator.num_processes):
                logging.warning(
                    f"Resuming with num_processes={accelerator.num_processes} but the checkpoint was "
                    f"written with num_processes={saved_num_processes}. The data order resumes at the "
                    "right epoch/offset, but per-rank sample-exactness requires the same world size."
                )
            if is_main_process and saved_batch_size not in (None, cfg.batch_size):
                logging.warning(
                    f"Resuming with batch_size={cfg.batch_size} but the checkpoint was written with "
                    f"batch_size={saved_batch_size}. The data order resumes at the right epoch/offset, "
                    "but per-rank sample-exactness requires the same batch size."
                )
            sampler_state = compute_sampler_state(step, len(sampler), ckpt_batch_size, ckpt_num_processes)
            sampler.load_state_dict(sampler_state)
            if is_main_process:
                logging.info(
                    f"Resuming data order at epoch {sampler_state['epoch']}, "
                    f"sample {sampler_state['start_index']}"
                )
    else:
        # streaming dataset 不支持随机 index 访问，交给 iterable/streaming 自己产生样本。
        shuffle = True
        sampler = None

    # Only swap in the language-aware collate when the dataset actually
    # declares language columns; otherwise stay on PyTorch's default
    # collate so non-language training runs are unaffected.
    # 如果数据里有 language/token 字段，需要自定义 collate；否则使用 PyTorch 默认 collate。
    collate_fn = lerobot_collate_fn if dataset.meta.has_language_columns else None
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=collate_fn,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
    )

    # Build eval dataloader if a held-out split exists
    # 这里的 eval_dataloader 是“离线 held-out eval loss”，和 env rollout eval 不是一回事。
    eval_dataloader = None
    if eval_dataset is not None:
        eval_ds = eval_dataset
        if cfg.max_eval_samples > 0 and hasattr(eval_dataset, "hf_dataset"):
            # 为了让 eval 更快，可以限制最大评估样本数。
            # 这里按 task_index 尽量均匀抽样，避免只评估某一个 task。
            task_arr = eval_dataset.hf_dataset.data.column("task_index").to_numpy()
            unique_tasks = sorted(set(task_arr.tolist()))
            per_task = max(1, cfg.max_eval_samples // len(unique_tasks))
            selected: list[int] = []
            for t in unique_tasks:
                frames = (task_arr == t).nonzero()[0][:per_task]
                selected.extend(frames.tolist())
            eval_ds = torch.utils.data.Subset(eval_dataset, selected)

        # eval collate 需要和 train collate 保持一致，尤其是语言列。
        eval_collate_fn = lerobot_collate_fn if dataset.meta.has_language_columns else None
        eval_dataloader = torch.utils.data.DataLoader(
            eval_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
            collate_fn=eval_collate_fn,
            prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
            persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
        )

    # Prepare everything with accelerator
    # accelerator.prepare 是分布式/设备包装的关键步骤：
    # policy 可能被 DDP/FSDP 包装，dataloader 可能被按 rank 切分。
    accelerator.wait_for_everyone()
    if eval_dataloader is not None:
        policy, optimizer, dataloader, lr_scheduler, eval_dataloader = accelerator.prepare(
            policy, optimizer, dataloader, lr_scheduler, eval_dataloader
        )
    else:
        policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            policy, optimizer, dataloader, lr_scheduler
        )

    # FSDP optimizer state is sharded across ranks, so it can only be loaded once the optimizer and
    # model are FSDP-wrapped (i.e. after `prepare`). Collective: every rank must participate.
    if cfg.resume and accelerator.distributed_type == DistributedType.FSDP:
        # FSDP 的 optimizer state 加载是 collective 操作，所有 rank 必须一起参与。
        load_fsdp_optimizer_state(policy, optimizer, cfg.checkpoint_path)

    # cycle(dataloader) 会无限循环 dataloader。
    # 本脚本按 cfg.steps 控制训练长度，而不是按 epoch 数控制。
    dl_iter = cycle(dataloader)

    # 再次确保模型处于训练模式。
    policy.train()

    train_metrics = {
        # Per-rank loss reflects only one shard of the global batch; mean recovers the loss DDP
        # is actually optimizing. grad_norm and lr are already identical on every rank (post
        # gradient sync / deterministic scheduler) so reducing them would be a no-op collective.
        # loss 在每个 rank 上只看到全局 batch 的一部分，所以日志时需要跨 rank 求平均。
        "loss": AverageMeter("loss", ":.3f", reduction="mean"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        # Report the slowest rank for bottleneck-style timings so multi-GPU runs surface the
        # true straggler instead of rank 0's view.
        # 多卡训练速度由最慢的 rank 决定，所以耗时指标用 max reduction 更符合真实瓶颈。
        "update_s": AverageMeter("updt_s", ":.3f", reduction="max"),
        "dataloading_s": AverageMeter("data_s", ":.3f", reduction="max"),
        # Derived from the post-reduce max step time; set once per log window on the main rank.
        # samples_per_s 后面由 effective batch size / 最慢 step time 计算。
        "samples_per_s": AverageMeter("smp/s", ":.0f"),
    }
    if torch.cuda.is_available():
        # max() because headroom is gated by the worst-case rank.
        # 显存也看最坏 rank，否则 rank0 显存低会掩盖其它 rank OOM 风险。
        train_metrics["gpu_mem_gb"] = AverageMeter("mem_gb", ":.2f", reduction="max")

    # Keep global batch size for logging; MetricsTracker handles world size internally.
    # effective_batch_size 用于日志展示吞吐；MetricsTracker 内部会处理 world size。
    effective_batch_size = cfg.batch_size * accelerator.num_processes
    train_tracker = MetricsTracker(
        cfg.batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    if is_main_process:
        progbar = tqdm(
            total=cfg.steps - step,
            desc="Training",
            unit="step",
            disable=inside_slurm(),
            position=0,
            leave=True,
        )
        logging.info(
            f"Start offline training on a fixed dataset, with effective batch size: {effective_batch_size}"
        )

    # 主训练循环：从当前 step 跑到 cfg.steps。
    # 如果 cfg.resume=True，step 已经从 checkpoint 恢复，循环会从恢复点继续。
    for _ in range(step, cfg.steps):
        # 1. 取一个 batch，并统计“取数 + 轻量预处理”的耗时。
        start_time = time.perf_counter()
        batch = next(dl_iter)
        for cam_key in dataset.meta.camera_keys:
            if cam_key in batch and batch[cam_key].dtype == torch.uint8:
                # 有些数据集图像以 uint8 [0,255] 读出；这里先转成 float32 [0,1]。
                # 后续 normalizer/image processor 再按 policy 需要继续处理。
                batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0
        # 2. policy/reward model 专属 preprocessor：
        # feature rename、归一化、tokenize、device move、动作格式转换等都可能在这里发生。
        batch = preprocessor(batch)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        # 3. 执行一次参数更新。
        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
            sample_weighter=sample_weighter,
        )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        # 这一次 update 已经完成，所以先 step += 1，再判断是否需要 log/eval/save。
        step += 1
        if is_main_process:
            progbar.update(1)
        # 更新 tracker 内部 step 计数，用于进度、吞吐和 epoch 估算。
        train_tracker.step()
        # 四类周期性动作：
        # - log：训练日志；
        # - saving：保存 checkpoint；
        # - env eval：在环境中 rollout；
        # - eval dataloader：在 held-out split 上算 eval_loss。
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_env_eval_step = cfg.env_eval_freq > 0 and step % cfg.env_eval_freq == 0
        is_eval_step = cfg.eval_steps > 0 and eval_dataloader is not None and step % cfg.eval_steps == 0

        if is_log_step:
            # Collective reduce must run on every rank, before the main-process gate below.
            # reduce_across_ranks 是 collective，所有 rank 都必须执行，不能只放在主进程里。
            train_tracker.reduce_across_ranks()
            if is_main_process:
                # Cluster-wide throughput, derived from the already-reduced (max) step time so it
                # reflects the slowest rank — which is what actually gates the next iteration.
                step_time = train_tracker.update_s.avg + train_tracker.dataloading_s.avg
                if step_time > 0:
                    # 用最慢 rank 的 step time 计算全局吞吐，更接近真实训练速度。
                    train_tracker.samples_per_s = effective_batch_size / step_time
                logging.info(train_tracker)
                if wandb_logger:
                    wandb_log_dict = train_tracker.to_dict()
                    if output_dict:
                        wandb_log_dict.update(output_dict)
                    # Log sample weighting statistics if enabled
                    if sample_weighter is not None:
                        # 记录样本加权器的全局统计，方便看权重是否塌缩或异常。
                        weighter_stats = sample_weighter.get_stats()
                        wandb_log_dict.update({f"sample_weighting/{k}": v for k, v in weighter_stats.items()})
                    wandb_logger.log_dict(wandb_log_dict, step)
            # 一个日志窗口结束后清空平均值，下一窗口重新累计。
            train_tracker.reset_averages()

        if is_eval_step:
            # held-out eval：不进入环境，只在 eval_dataset 上算 loss。
            # 这通常比 rollout eval 更快，也更适合真实机器人数据集。
            policy.eval()
            eval_loss_sum = 0.0
            n_eval_batches = 0
            with torch.no_grad(), accelerator.autocast():
                for eval_batch in eval_dataloader:
                    for cam_key in dataset.meta.camera_keys:
                        if cam_key in eval_batch and eval_batch[cam_key].dtype == torch.uint8:
                            # 和训练 batch 保持一致的图像缩放。
                            eval_batch[cam_key] = eval_batch[cam_key].to(dtype=torch.float32) / 255.0
                    eval_batch = preprocessor(eval_batch)
                    loss, _ = policy.forward(eval_batch)
                    eval_loss_sum += loss.item()
                    n_eval_batches += 1
            eval_loss = eval_loss_sum / max(n_eval_batches, 1)
            eval_loss = torch.tensor(eval_loss, device=device)
            # eval_loss 在每个 rank 上各自算一部分，这里跨 rank 求平均。
            eval_loss = accelerator.reduce(eval_loss, reduction="mean").item()
            policy.train()

            if is_main_process:
                logging.info(f"step {step}: eval_loss={eval_loss:.4f}")
                if wandb_logger:
                    wandb_logger.log_dict({"eval_loss": eval_loss}, step=step, mode="eval")

        if cfg.save_checkpoint and is_saving_step:
            # Under FSDP, gathering the full model + optimizer state dicts is a cross-rank collective,
            # so all ranks must participate; rank 0 then writes the materialized dicts. For DDP /
            # single-GPU the state dicts are saved the normal way inside save_checkpoint.
            # FSDP 保存 checkpoint 时需要先把分片权重/优化器状态 gather 成完整 state dict。
            # 这个 gather 是 collective，所有 rank 都要执行；最终只有主进程写磁盘。
            is_fsdp = accelerator.distributed_type == DistributedType.FSDP
            if is_fsdp:
                model_state_dict, optim_state_dict = gather_fsdp_state_dicts(policy, optimizer)
            else:
                model_state_dict, optim_state_dict = None, None
            if is_main_process:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                # 保存内容包括：模型、训练配置、optimizer/scheduler、processor、batch/world size。
                # processor 很关键，因为推理时必须知道如何预处理 observation 和反处理 action。
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    num_processes=accelerator.num_processes,
                    batch_size=cfg.batch_size,
                    model_state_dict=model_state_dict,
                    optim_state_dict=optim_state_dict,
                )
                # 更新 last checkpoint 指针，方便 resume 自动找到最近一次保存点。
                update_last_checkpoint(checkpoint_dir)
                if cfg.save_checkpoint_to_hub:
                    # 可选把 checkpoint 同步到 Hub，适合远程训练或长时间训练防丢。
                    push_checkpoint_to_hub(
                        checkpoint_dir,
                        cfg.policy.repo_id,
                        private=cfg.policy.private,
                    )
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            # 等主进程保存完成，其它 rank 再继续，避免训练状态错位。
            accelerator.wait_for_everyone()

        if cfg.env and is_env_eval_step:
            if is_main_process:
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")
                # env rollout eval：把当前 policy 放进仿真环境中真正执行动作，得到 reward/success/video。
                with torch.no_grad(), accelerator.autocast():
                    eval_info = eval_policy_all(
                        envs=eval_env,  # dict[suite][task_id] -> vec_env
                        policy=accelerator.unwrap_model(policy),
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        n_episodes=cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed,
                        max_parallel_tasks=cfg.env.max_parallel_tasks,
                    )
                # overall metrics (suite-agnostic)
                # overall 是跨 suite/task 聚合后的总指标。
                aggregated = eval_info["overall"]

                # optional: per-suite logging
                # 同时打印每个 suite 的结果，方便定位是哪个 benchmark/task 表现不好。
                for suite, suite_info in eval_info.items():
                    logging.info("Suite %s aggregated: %s", suite, suite_info)

                # meters/tracker
                # 这里复用 MetricsTracker 记录评估 reward、success rate 和评估耗时。
                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size,
                    dataset.num_frames,
                    dataset.num_episodes,
                    eval_metrics,
                    initial_step=step,
                    accelerator=accelerator,
                )
                eval_tracker.eval_s = aggregated.pop("eval_s")
                eval_tracker.avg_sum_reward = aggregated.pop("avg_sum_reward")
                eval_tracker.pc_success = aggregated.pop("pc_success")
                if wandb_logger:
                    # 把评估指标和一段 rollout 视频写入 W&B。
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(eval_info["overall"]["video_paths"][0], step, mode="eval")

            # rollout eval 完成后同步所有 rank。
            accelerator.wait_for_everyone()

    if is_main_process:
        progbar.close()

    if eval_env:
        # 关闭仿真环境，释放渲染器、物理引擎、子进程等资源。
        close_envs(eval_env)

    # FSDP 下最终 push_to_hub 也需要完整模型 state dict。
    is_fsdp = accelerator.distributed_type == DistributedType.FSDP
    model_state_dict = accelerator.get_state_dict(policy) if is_fsdp else None
    if is_main_process:
        logging.info("End of training")

        if getattr(active_cfg, "push_to_hub", False):
            # unwrap_model 去掉 DDP/FSDP/Accelerate 包装，拿到原始 policy/reward model。
            unwrapped_model = accelerator.unwrap_model(policy)
            # PEFT only applies when training a policy — reward models use the plain path.
            if not cfg.is_reward_model_training and cfg.policy.use_peft:
                # PEFT policy 通常上传 adapter/peft 权重，并附带 dataset_meta。
                unwrapped_model.push_model_to_hub(cfg, peft_model=unwrapped_model, dataset_meta=dataset.meta)
            else:
                # 普通 policy 或 reward model 上传完整权重；FSDP 时传入 gather 后的 state_dict。
                unwrapped_model.push_model_to_hub(cfg, state_dict=model_state_dict, dataset_meta=dataset.meta)
            # processor 也要上传，否则别人加载模型时不知道输入输出该如何处理。
            preprocessor.push_to_hub(active_cfg.repo_id)
            postprocessor.push_to_hub(active_cfg.repo_id)

    # Properly clean up the distributed process group
    # 所有 rank 最后同步，并让 accelerate 清理分布式进程组。
    accelerator.wait_for_everyone()
    accelerator.end_training()


def _remote_target_in_argv() -> bool:
    """True when the CLI requests a remote HF Jobs run (--job.target=<non-local>)."""
    # 这里在 parser.wrap() 真正解析配置之前，先手动扫一遍 sys.argv。
    # 目的：如果是远程 job，本地机器可能没有目标 GPU/环境，不应该提前因为 device warning 干扰用户。
    target = None
    args = sys.argv[1:]
    for i, tok in enumerate(args):
        if tok == "--job.target" and i + 1 < len(args):
            target = args[i + 1]
        elif tok.startswith("--job.target="):
            target = tok.split("=", 1)[1]
    return JobConfig.is_remote_target(target)


def main():
    # 允许第三方包注册自定义 policy/env/robot/processor 等组件。
    register_third_party_plugins()
    if _remote_target_in_argv():
        # The policy device is resolved on the remote pod, not here, so silence the
        # client-side "Device '...' is not available" warning PreTrainedConfig emits
        # while parsing the config (it fires before train() can dispatch remotely).
        # 远程 job 的设备在远程 pod 上解析，本地无需因为没有 cuda 等设备而刷 warning。
        logging.getLogger("lerobot.configs.policies").setLevel(logging.ERROR)
    # parser.wrap() 会把 CLI/config 转成 TrainPipelineConfig 后再调用 train(cfg)。
    train()


if __name__ == "__main__":
    main()
