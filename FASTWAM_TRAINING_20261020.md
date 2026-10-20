# SO-101 FastWAM 训练记录（2026-10-20）

用户要求当前 SmolVLA 完成后，在 Merlin 训练 FastWAM。本次核验两项 SmolVLA 均已完成，因此已提交后续训练任务。

## 当前状态

> **2026-10-20 09:43（北京时间）检查**：固定模型、VAE、T5 和 tokenizer 已全部下载并通过官方文件哈希校验（09:32:56 完成，约 24.44 GiB），`READY` 已生成，CPU 数据/权重覆盖/回归检查均通过。准备进程正常完成退出。worker 4421791 仍未分配地址，申请进程仍在等待；尚无 GPU 运行日志、正式训练清单或 checkpoint。保持自动等待与监控。

> **2026-10-20 09:26（北京时间）恢复更新**：用户明确要求清理后继续训练，已启动第二次资产准备与准备检查进程（PID 2250820、2250821），复用 worker 4421791，不重复申请 GPU。旧失败状态和日志保存在 `preparation_history/disk_quota_failure_before_resume_20261020T012637Z`，顶层失败标记已移入该历史目录。下载已恢复并观察到写入超过 1 GiB；GPU 尚未分配地址，当前公共资源列表暂未显示可用 A100。资源和固定权重齐备后，原脚本自动进行 GPU 预检、保存/加载检查，再从基础权重开始正式训练。定时检查 `fastwam` 已恢复为 ACTIVE，每 15 分钟检查；当前不能声称正式训练已经开始。

> **后续清理更新**：用户已授权清空 Merlin `train_data` 并删除特别大的旧模型文件。已清空 `/root/train_data` 实际指向的 `/mnt/rootfs`（保留目录及入口软链接），并删除 π0.5 两次实验、VLA-JEPA 的旧中间权重和预检权重/优化器文件，合计文件逻辑大小 **390,377,178,488 字节，约 363.57 GiB**。三项旧模型的最终检查点、两个 SmolVLA 候选、FastWAM 所需真实数据和仿真数据均保留。4 MiB 写入并 fsync 测试通过；虚拟文件系统 `df/statvfs` 的剩余容量未即时反映删除量，因此不以其报告存储配额额度。删除清单及检查结果位于本地和 Merlin 仓库的 `outputs/cleanup_20261020/`。本次仅执行清理，未重新下载或启动训练，原失败标记保留，监控仍暂停。

> **2026-10-20 00:43（北京时间）检查更新：资产准备失败，正式训练尚未开始。** `prepare.log` 在 00:37:03 报告下载写入 `Errno 5 Input/output error`，随后写失败状态时明确报告 `Errno 122 Disk quota exceeded`。因此 `preparation_status.json` 仍残留旧的 `download_base`，不可据此认为下载继续进行。00:37:21 已生成 `PREPARATION_FAILED`，资产准备及后续门禁进程均已退出。worker 4421791 仍在排队，无 `worker.log`、正式训练清单或 checkpoint。先解决工作区存储配额，再另行恢复下载；本次检查未重启、删除文件、申请或释放 GPU。下列提交说明保留作为历史。

- SmolVLA 基线和混合版：均为 20,000 步，退出码 0，完成清单、最终权重和处理器已核验。
- FastWAM：已申请 **worker 4421791，1 × A100-SXM-80GB，8 CPU、96 GiB 主存**；提交后首次检查尚未分配地址。
- 固定权重正在下载，CPU 数据预检已通过；目前还不能声称已开始 FastWAM 正式训练。
- 启动链：固定资产校验 → CPU 数据及权重结构检查 → GPU 两步训练 → 保存并重新加载推理 → 从基础权重重新开始正式 20,000 步 → 最终重新加载检查。任何阶段失败都会停止，不自动重启或重复申请 GPU。
- 原 T4 worker 4417488 保留；单卡 80 GB 配置尚需实际 GPU 预检验证显存及算子兼容性。
- 已完成 15 项 CPU 回归，2 项与本次 CPU 检查无关的编译性能/VAE 测试未运行；Ruff 检查和格式检查通过。实际基础权重的 1,645 个张量与 meta-device 架构逐项匹配，目标架构共 1,651 个张量，差额仅为预期的 6 个新接口张量。
- 本聊天定时检查 **`fastwam`** 每 15 分钟运行；00:43 准备失败时暂停，09:26 用户授权恢复后重新启用。正常下载、排队和训练进度不重复通知；首次正式开训、完成、失败、停滞或需用户操作时通知。

## 数据与模型

- 使用真实示范 0–78 共 79 条训练、79–87 共 9 条独立验证；另使用 200 条经过物理成功判定的仿真示范。原数据保持只读。
- 真实数据：`/root/code/lerobot/datasets/smolvla_tape_clean_20261002_142151_curated_v1`。
- 仿真数据：`/root/code/lerobot/outputs/r2s2r_generation_20261018/workflow/production_dataset/dataset`。200 条、138,000 帧，生成过程 457 次尝试；读取时核对生成清单内的文件哈希。
- 双相机：`front` 腕部、`side` 外部；单幅缩放至 224×224，再拼接为 224×448。原视频为 640×480、30 FPS。
- 每个训练窗口使用 t 到 t+32 的跨度，取 0、4、8、…、32 共 9 帧视频，搭配 32 步动作。不使用跨 episode 或补齐的未来窗口。
- 有效窗口：真实训练 **22,789**、独立验证 **2,027**、仿真 **131,600**。验证的 9 条不参与训练或归一化统计。
- 状态及动作顺序：`shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`。前五维为度，夹爪 0–100；不沿用 LIBERO 的夹爪翻转。真实 action 延续原始记录的处理后主臂目标语义，不将其冒称为已核验的驱动实际下发值。
- 每个优化器更新恰好 4 个真实样本、4 个仿真样本；单次前向 batch=1，累计 8 次梯度。20,000 个优化器更新，seed=1000，AdamW 学习率 1e-4，权重衰减 0.01，梯度裁剪 10，BF16，启用梯度检查点。
- 视频专家与动作专家一起训练。每 5,000 步保存模型、优化器/RNG 和全部 2,027 个独立验证窗口结果；验证噪声固定，不改变训练随机状态。
- 官方通用基础权重省略动作输入、动作输出及关节状态输入层。仅为 SO-101 初始化这 6 个权重/偏置张量，其余共享权重必须完整匹配；已有层名转换与上游 MoT 兼容。
- 该任务的语言指令恒定，冻结的 T5 提示向量只计算一次，然后把 T5 移到 CPU。VAE 仍用于编码未来视频，视频损失和动作损失权重均为 1。

场景仍为 **`calibration_passed=false` 的近似仿真实验**。训练 loss、验证 loss 或完成训练均不能证明真机抓取成功率提升。

## 固定资产与环境

| 资产 | 固定 revision |
|---|---|
| `lerobot/fastwam_base` | `385a4bd1df7a7895b24c6e530482092b7b08a783` |
| `Wan-AI/Wan2.2-TI2V-5B-Diffusers`，VAE + T5 | `b8fff7315c768468a5333511427288870b2e9635` |
| `google/umt5-xxl`，tokenizer | `66cb9e7e85526fe440a945569e42c72fb6cbc0ad` |

公开文件约 24.44 GiB。Merlin 访问 Hugging Face 官方下载端点超时，因此使用公共镜像下载，明确关闭隐式 token 且传入 `token=False`；下载后逐文件核对从官方 API 获取的大小及 SHA-256/Git blob SHA-1。锁文件为 `model_files.lock.json`，不能仅以下载结束认定完整。

工作目录内使用 `uv venv --system-site-packages` 创建隔离入口，继承已存在的 Python 3.12、PyTorch 2.7.0+cu118、Transformers 5.5.4、Diffusers 0.39、Accelerate 1.14。LeRobot 源码冻结到 `code/src`，经 `PYTHONPATH` 指向该副本。未修改 SmolVLA 运行环境。

训练脚本在当前进程内把 VAE 资产定位常量指向已校验的本地目录；配置也指向固定的 T5/tokenizer 目录。后续迁移推理环境时必须保留或重新配置这些冻结组件，单独复制 `model.safetensors` 不足以构建完整模型。

## 路径与检查

SSH：`37944.niuzhibin.ws@merlin-ssh-proxy-cn.byted.org`，端口 `22222`。先运行 `mlx worker list` 获取 worker 的当前状态和地址。

所有本次准备文件位于：

```text
/root/code/lerobot/outputs/fastwam_20261020/
  fastwam_so101_20261020.json        固定实验参数
  model_files.lock.json             官方文件校验值
  source.tar.gz / code/src/         源码快照
  assets/                          固定基础模型、VAE、T5、tokenizer
  preparation_status.json          下载和资产校验状态
  prepare.log / prepare.pid        资产准备进程
  data_preflight.json              数据预检与来源清单
  base_header_audit.json           共享权重覆盖检查
  cpu_tests.json / tests.log       CPU 回归结果
  preparation_finalize.log         准备门禁日志
  READY / PREPARATION_FAILED       准备成功或失败标记
  allocation.log / allocation.pid GPU 申请进程
  worker.log / worker.exitcode     GPU 训练全流程日志和退出码
  smoke/                          两步预检模型，禁止当作正式成果
```

正式输出：

```text
/root/code/lerobot/outputs/train/fastwam_so101_mixed_20261020/
  training_manifest.json
  checkpoints/005000/...
  checkpoints/010000/...
  checkpoints/015000/...
  checkpoints/020000/
    pretrained_model/             权重、配置、前后处理器和归一化统计
    optimizer_and_rng.pt
    validation.json
    reload_check.json
```

只有同时满足 `worker.exitcode=0`、正式清单 `completed=true, final_step=20000`、最终模型/配置/前后处理器文件完整，以及 `reload_check.json` 的 `passed=true`，才能报告训练流程成功。GPU 预检不能代替正式训练，存在目录也不能代替成功证据。

本次暂存 Merlin，不自动上传 Hub、启动机械臂或释放 worker。遇到失败保留日志和检查点，由后续明确处理决定是否恢复。

官方资料：[LeRobot FastWAM](https://huggingface.co/docs/lerobot/main/fastwam)、[基础权重](https://huggingface.co/lerobot/fastwam_base)、[FastWAM 项目代码](https://github.com/yuantianyuan01/FastWAM)。
