# SmolVLA 清洗数据训练（2026-10-02）

## 完成确认（2026-10-03 检查）

- 已成功完成 20,000/20,000 步，退出码为 `0`，日志记录 `End of training`。
- 完成时间：2026-10-03 06:58:13 北京时间（远端日志 UTC 2026-10-02 22:58:13）。训练循环耗时约 11 小时 59 分 43 秒。
- 最后一条训练 loss：`0.066`；这是训练集指标，尚未实机评估成功率。
- 最终模型路径：`/root/code/lerobot/outputs/train/smolvla_tape_curated_20261002/checkpoints/020000/pretrained_model`。
- 已核对最终权重 `model.safetensors`（906,712,552 bytes）、模型/训练配置、前后处理器、归一化参数及训练状态文件，5k/10k/15k/20k checkpoint 均存在。
- 训练配置关闭自动 Hub 上传；后续于 2026-10-03 手动上传到 `zhniu/smolvla_tape_curated_20261002`，提交 `f9c77976166d68f4416f6d2038f31574f33281ee`，文件大小与 LFS SHA-256 校验通过。
- Worker `4377679` 仍在，本次仅检查，没有释放资源或启动机械臂。

## 数据与传输

- 原始数据保持不变；使用 `zhniu/smolvla_tape_clean_20261002_142151_curated_v1`。
- 88 条 episode、27,632 帧、30 FPS，两路 640×480 视频。
- 本地数据：`C:/Users/Admin/.cache/huggingface/lerobot/zhniu/smolvla_tape_clean_20261002_142151_curated_v1`。
- 通过 SCP 上传 tar 到 Merlin，SHA-256 校验通过后解包。
- SHA-256：`6ca6809526d514499a83c602949f96c52a7176df2663efc8d89c48ce20bf6ab5`。
- 远端数据：`/root/code/lerobot/datasets/smolvla_tape_clean_20261002_142151_curated_v1`。
- 清洗范围和验证结果见 `DATASET_CLEANING_20261002.md` 以及数据集内 `cleaning_manifest.json`。

## GPU 与环境

- 入口：`37944.niuzhibin.ws@merlin-ssh-proxy-cn.byted.org:22222`。
- Worker：`4377679`，Tesla T4 15 GiB，8 CPU、32 GiB 内存。
- Worker SSH：`root@fdbd:dc03:6:622::69:11160`（从入口登录）。
- 代码：`/root/code/lerobot`；Python：`/root/miniconda3/envs/lerobot/bin/python`。
- 启动 shell PID：`240`（worker 内）。

## 训练配置

- 从 `lerobot/smolvla_base` 重新微调，不继承旧任务模型。
- 20,000 steps，batch size 8，4 个数据加载进程，seed 1000。
- `policy.scheduler_decay_steps=20000`，每 5,000 步保存，正式训练每 100 步打印。
- 相机映射：front → camera1；side → camera2。
- 视频后端 PyAV；保持采集时的单位与分辨率。
- 全部 88 条清洗示范参与训练；本次没有单独划分验证集。
- 先执行数据读取校验和独立的两步训练/保存测试，再从基础模型开始正式训练。
- `policy.push_to_hub=false`，当前仅保存 Merlin 上的 checkpoint。

## 文件与检查

- 本地/远端启动脚本：`start_smolvla_curated_20261002.sh`。
- 训练脚本：`train_smolvla_curated_20261002.sh`。
- 上传校验/解包：`prepare_smolvla_curated_20261002.py`。
- CUDA/数据读取校验：`preflight_smolvla_curated_20261002.py`。
- 日志：`/root/code/lerobot/smolvla_tape_curated_20261002.log`。
- 结束退出码：`/root/code/lerobot/smolvla_tape_curated_20261002.exitcode`。
- 正式 checkpoint：`/root/code/lerobot/outputs/train/smolvla_tape_curated_20261002/checkpoints`。
- 预检输出：`/root/code/lerobot/outputs/train/smolvla_tape_curated_20261002_preflight`。

启动时历史记录（2026-10-02 19:00 左右，北京时间）：正式训练已运行至 29/20,000 步，步数持续增加，约 2.18 秒/步。当时估算约需 12 小时。

- CUDA、数据集首/中/尾帧读取校验通过，PyTorch `2.7.0+cu118`。
- 两步预检训练正常结束，已核对权重（906,712,552 bytes）、配置、前/后处理器及归一化参数、tokenizer、优化器和训练状态文件。
- 正式训练从基础模型重新加载，20,000 步调度、1,000 步 warmup；未继承两步预检的调度或权重。
- 正式训练进程 PID `5344`；观察到 GPU 利用率 97%、显存 3,591/15,360 MiB。
- 尚未训练完成；正式训练第一条 loss 汇总在第 100 步打印。退出码文件尚未生成。

```powershell
ssh -p 22222 37944.niuzhibin.ws@merlin-ssh-proxy-cn.byted.org "tail -n 30 /root/code/lerobot/smolvla_tape_curated_20261002.log"
```

训练完成必须核对 20,000 steps、退出码 0、最终 checkpoint 的权重、配置和处理器文件；进程启动或目录存在不代表训练完成。训练 loss 不能替代实际抓取评估。
