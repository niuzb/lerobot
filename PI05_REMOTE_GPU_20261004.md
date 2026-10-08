# Pi 0.5 清洗数据训练（2026-10-04）

## 完成确认（2026-10-06 检查）

- 已完成 80,000/80,000 小批次，即 20,000 次参数更新；正式退出码 `0`，日志记录 `End of training`。
- 完成时间：2026-10-05 20:51:37 北京时间（远端 UTC 12:51:37）。训练循环耗时约 35 小时 12 分 42 秒。
- 最后一组训练 loss `0.029`，grad_norm `0.481`；尚未实机评估。
- 最终模型：`/root/code/lerobot/outputs/train/pi05_tape_curated_20261004/checkpoints/080000/pretrained_model`。
- 已检查权重（约 8.8 GiB）、模型/训练配置、前后处理器与归一化参数、tokenizer，以及优化器/RNG/scheduler/training_step 状态文件均存在且非空。
- 模型已于 2026-10-06 上传 Hub 并校验通过，详见下方上传记录。worker `4380064` 仍在，本次未释放 GPU 或运行机械臂。
- 下方启动阶段状态为历史记录，以本节完成确认为准。

## Hugging Face 上传（2026-10-06）

- 模型：https://huggingface.co/zhniu/pi05_tape_curated_20261004
- 提交：`9d9688a57078df924724b1522e4964f7d75dc9d0`。
- 上传完整最终 pretrained_model（9 个文件），另附模型卡 README；未上传优化器/RNG 训练状态或数据集。
- 所有模型文件在该提交存在、大小一致、SHA-256 一致；LFS 文件比对 Hub SHA-256，普通文件下载后计算 SHA-256。
- 权重 `model.safetensors`：9,354,050,784 bytes，SHA-256 `73b63898971a98dca3ad8b10a9a95ede0f3c09e955dd7673728abf16a77a0bb6`。
- tokenizer 使用模型目录内的相对路径，前后处理器及归一化参数随模型保存。
- 上传脚本：`upload_pi05_curated_20261006.py`；使用 worker 上已有 `zhniu` 登录与代理。
- 校验清单：远端 `pi05_tape_curated_20261004.upload_verified.json`，本地 `outputs/pi05_launch_20261004/pi05_tape_curated_20261004.upload_verified.json`。

## 当前状态

已申请 A10 worker `4380064`（1 GPU，8 CPU，64 GiB 内存，显存 23,028 MiB）。
数据与 CUDA 预检通过。初次访问 tokenizer 返回 403；用户开通权限后重新验证通过。
基础权重与 tokenizer 已完整下载。四小批次训练预检通过，正式训练进程已启动。
正式训练日志为 `pi05_tape_curated_20261004.formal.log`，退出码为 `.formal.exitcode`。
最初的 `.exitcode` 是准备/预检调度进程的状态，不能用来判断正式训练结果。

2026-10-04 09:40 北京时间已确认正式训练到 40/80,000 小批次（10 次参数更新），
loss 0.193、grad_norm 0.851，数值有限。正式训练开始于 09:38:55 北京时间。
稳态约 1.55 秒/小批次，预计约 35 小时，实际耗时取决于数据加载、保存与资源稳定性。
GPU 利用率 100%，nvidia-smi 显存 14,624/23,028 MiB。
正式 shell PID `3814`，Python PID `3826`（仅对应本次 worker）。训练尚未完成。

公开基础权重预下载完成；状态查看 `pi05_base_20261004.download.log`、
`pi05_base_20261004.download.exitcode` 和成功后生成的 `pi05_base_20261004.download.json`。
不能以下载进程存在认定下载完成。

基础模型提交：`b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba`，权重 14,467,165,872 bytes。
Tokenizer 提交：`35e4f46485b4d07967e7e9935bc3786aad50687c`。
精确本地 snapshot 路径保存在 `pi05_tape_curated_20261004.assets.json`。

## 数据

与 SmolVLA 2026-10-02 训练完全相同的本机新采集清洗数据：

- ID：`zhniu/smolvla_tape_clean_20261002_142151_curated_v1`。
- 远端目录：`/root/code/lerobot/datasets/smolvla_tape_clean_20261002_142151_curated_v1`。
- 88 条 episode，27,632 帧，30 FPS，front（腕部）与 side（外部），640×480。
- 归档 SHA-256：`6ca6809526d514499a83c602949f96c52a7176df2663efc8d89c48ce20bf6ab5`，本次再次校验通过。
- 数据首、中、末样本图像和六维有限 state/action 检查通过；q01/q99 存在。
- 保持采集单位与绝对动作；保留全部 88 条用于训练，与 SmolVLA 一样未划验证集。

## 计划配置

从 `lerobot/pi05_base` 微调，不使用 SmolVLA 权重。
下载时记录基础模型与 tokenizer 的提交 SHA，并使用对应本地 snapshot 离线训练。
使用数据原始 front/side 特征名，增加一个 masked empty camera。

- 冻结视觉语言主干，训练动作专家：`freeze_vision_encoder=true`、`train_expert_only=true`。
- BF16，gradient checkpointing，关闭 compile。
- batch 4，梯度累积 4，有效 batch 16；4 个数据加载进程。
- 80,000 小批次步（CLI steps），累计 20,000 优化器更新，seed 1000。
- 该版本 scheduler 同样按小批次推进：decay 80,000、warmup 4,000，对应 20,000/1,000 次参数更新。
- chunk_size 50，n_action_steps 30，QUANTILES 归一化，绝对动作。
- 每 20,000 小批次（5,000 次优化器更新）保存，关闭自动 Hub 上传。
- 先独立四小批次训练/保存测试（包含一次优化器更新），通过后从基础权重重新启动正式训练。

当前环境：PyTorch `2.7.0+cu118`，transformers `5.5.4`，BF16 支持检查通过。
预检成功：四个 loss 为 0.368、0.251、0.099、0.100；第四步 grad_norm 0.809。
完整 optimizer_state.safetensors 约 2.1 GiB；峰值显存 12.87 GiB（PyTorch allocated）。
非首次小批次约 1.55–1.69 秒；这只是预检吞吐，正式耗时仍需按稳态测量。
预检目录为 `outputs/train/pi05_tape_curated_20261004_preflight_v3`。
初次加载权重成功后缺少处理器，现已在独立的 `outputs/pi05_tape_curated_20261004_base_assets`
生成本数据集的处理器并链接原始权重，官方缓存文件未改动。
旧预检目录及日志保留；v2 因修正梯度累积的步数含义而主动停止。

## 启动与检查

Merlin 入口：`37944.niuzhibin.ws@merlin-ssh-proxy-cn.byted.org:22222`。
Worker SSH：`root@fdbd:dc03:14:344::15`，端口 `10471`，仅这次 worker 有效。
Worker 由 `sleep infinity` 保持，训练脚本用 nohup 后台运行，断开本地连接不停止训练。

已完成预检后，正式训练使用 `start_pi05_formal_20261004.sh`，本次已后台启动。
下列命令是完整准备/预检入口，本次不要重复执行：

```bash
ssh -p 10471 root@fdbd:dc03:14:344::15 bash /root/code/lerobot/start_pi05_curated_20261004.sh
```

启动脚本归档之前失败的日志/PID/退出码；不会覆盖已有预检或正式训练输出。
启动前应核对 worker 仍在以及没有尚未结束的旧训练。

```powershell
ssh -p 22222 37944.niuzhibin.ws@merlin-ssh-proxy-cn.byted.org "tail -n 40 /root/code/lerobot/pi05_tape_curated_20261004.formal.log"
ssh -p 22222 37944.niuzhibin.ws@merlin-ssh-proxy-cn.byted.org "mlx worker list"
```

输出：`/root/code/lerobot/outputs/train/pi05_tape_curated_20261004/checkpoints`。
正式退出码：`/root/code/lerobot/pi05_tape_curated_20261004.formal.exitcode`。
初次准备失败或被停止的调度进程退出码不代表正式训练失败。

完成标准：80,000 小批次步（20,000 次优化器更新）、退出码 0、End of training、
最终 `checkpoints/080000` 的权重/配置/前后处理器及训练状态完整。
训练 loss 不等于实机成功率。本次没有运行机械臂。
