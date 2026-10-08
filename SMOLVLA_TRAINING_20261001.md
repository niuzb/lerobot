# SmolVLA 微调运行记录（2026-10-01）

参考 ACT_TRAINING_COMPLETE_GUIDE.md，复用其中的胶带放入笔筒数据。

## 已确认状态

### 训练和上传已完成

北京时间 2026-10-01 12:15:10 完成 20000 步，最终日志 loss 0.045；12:23:40 上传成功，脚本退出码 0。最终 `checkpoints/020000/training_state/training_step.json` 确认 step=20000。

最终 checkpoint 含模型、配置、归一化处理器、tokenizer、优化器、调度器和随机数状态。独立查询 Hugging Face API，已核对上传文件齐全且大小与本地一致，模型权重 906712552 bytes。Hub revision：`6b4ed47e4b295c0f2c41419659bf772ea5639279`。

模型：https://huggingface.co/zhniu/smolvla_tape_into_pen_holder_20261001

训练监控已暂停；GPU worker 4374960 按约定保留。尚未进行实机成功率评估。

### 11:10 保存故障修复

原运行于北京时间 11:00:17 达到 5000/20000 steps，loss 0.144，但在保存时因 `draccus.encode(self, PreTrainedConfig)` 与云端 draccus 0.10.0 单参数接口不兼容退出（退出码 1）。原 005000 目录只有空 config.json，无法恢复原 5000 步。失败目录已改名为 `smolvla_tape_into_pen_holder_20261001_failed_save`，原日志为 `smolvla_tape_into_pen_holder_20261001.failed_save.log`。

本地和云端 `src/lerobot/configs/policies.py` 已修复为单参数 encode，并显式保留 type 字段；云端修改前备份为 `policies.py.before_smolvla_save_fix`。

11:10:40 完成一轮 1-step 保存验证，退出正常；`checkpoints/000001` 中已写入约 907 MB 模型、约 413 MB 优化器状态、处理器、配置和随机数状态。随后通过 `resume_smolvla_20261001.sh` 启动恢复到 20000 步，重新开启最终上传；外层脚本 PID 为 `81912`。该验证运行只有 1 步，其学习率调度按短运行自动缩放；恢复时按 20000 步重新构建调度。

### 原始启动记录（以下不是当前进度）

北京时间 2026-10-01 10:44:59 已完成 100/20000 steps，日志 loss 为 1.104；当前约 5.37 steps/s，GPU 利用率 94%，显存占用 3382 MiB。训练在后台继续，尚未完成，也尚未确认 Hub 上传。按当前速度估算剩余训练约 1 小时，保存和上传另计。

训练进程 PID 为 `851`，外层脚本 PID 为 `850`。首次相机字段预检失败日志保存在 `smolvla_tape_into_pen_holder_20261001.camera_precheck.log`，已修复后重启，当前日志为下文无 precheck 后缀的文件。

## 资源与配置

- Merlin worker：`4374960`，NVIDIA A10（23028 MiB），8 CPU、32 GiB RAM。
- SSH：`ssh 37944.niuzhibin.ws@merlin-ssh-proxy-cn.byted.org -p 22222`，再执行 `mlx worker login 4374960 -- bash`。连接前用 `mlx worker list` 核对 worker 仍存在。
- 代码：`/root/code/lerobot`，commit `e1cdd073e69ef891676ca2c6128127c707d31ba2`。保留了原有本地改动。
- 解释器：`/root/miniconda3/envs/lerobot/bin/python`。
- 已验证 CUDA 可用、SmolVLA 可导入、HF 身份为 `zhniu`。
- 已安装版本：PyTorch `2.7.0+cu118`、transformers `5.5.4`、accelerate `1.14.0`、huggingface-hub `1.19.0`。
- 数据：`zhniu/lerobot_zihao_dataset_a_20260927_202323`，固定 revision `01241e32b3b32f043ffb786a4d9afb264b4f981a`；40 episodes、12000 帧、30 FPS，状态/动作均为 6 维。
- 基础权重：`lerobot/smolvla_base`；启动前查询 revision 为 `d9f33c94a60fb382c90dea2164c96845bd955e28`。
- 相机：使用 `--rename_map` 将 `observation.images.front` 映射到 `observation.images.camera1`，`observation.images.side` 映射到 `observation.images.camera2`。SmolVLA 使用实际提供的两路图像；推理时需保持同样的映射。
- 微调：batch 8、20000 steps、每 5000 steps 保存、每 100 steps 记录日志；学习率采用 SmolVLA preset，衰减周期 20000 steps。
- 目标 Hub 仓库：`zhniu/smolvla_tape_into_pen_holder_20261001`，训练结束自动上传。

## 文件与监控

启动脚本为同目录 `train_smolvla_20261001.sh`，已复制到云端同名路径。脚本必须在 GPU worker 内执行。

```bash
cd /root/code/lerobot
pgrep -af lerobot-train
nvidia-smi
tr '\r' '\n' < smolvla_tape_into_pen_holder_20261001.log | tail -n 30
cat smolvla_tape_into_pen_holder_20261001.exitcode  # 仅在脚本退出后生成
```

输出目录：`/root/code/lerobot/outputs/train/smolvla_tape_into_pen_holder_20261001`。

最终 checkpoint 预期位置：`checkpoints/020000/pretrained_model`。这只是预期路径，不代表训练已经完成。需核对目标步数、退出码 0、最终 checkpoint 和 Hub 文件后确认完成。

不要在同一输出目录重复启动；中断时使用完整 checkpoint 的 `train_config.json` 和 `--resume=true` 恢复。

官方微调说明：https://huggingface.co/docs/lerobot/smolvla
