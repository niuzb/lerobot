# SO-101 胶带放入笔筒：ACT 模型训练完整流程

更新日期：2026-09-28。适用环境：Windows 本地 SO-101 主从机械臂、双摄像头、Merlin 云端 GPU、Hugging Face Hub。

本文以本次实际完成的训练为例，覆盖环境准备、数据录制与验证、申请 GPU、训练、断点恢复、模型上传及推理交接。命令标注了执行位置；Windows PowerShell 使用反引号续行，Linux Bash 使用反斜杠续行，不要混用。

## 1. 本次训练的输入与产物

| 项目 | 本次使用的值 |
| --- | --- |
| 任务英文描述 | `Put the tape into the pen holder` |
| HF 用户名 | `zhniu` |
| 训练数据集 ID | `zhniu/lerobot_zihao_dataset_a_20260927_202323` |
| 模型仓库 ID | `zhniu/act_tape_into_pen_holder` |
| 数据规模 | 40 条 episode，共 12,000 帧，30 FPS |
| 单条录制时长 | 10 秒，复位间隔 2 秒 |
| 从臂 | `so101_follower`，COM5，`zihao_follower_arm` |
| 主臂 | `so101_leader`，COM6，`zihao_leader_arm` |
| 摄像头 | `front`：编号 0；`side`：编号 2；均为 640×480、30 FPS、MJPG |
| 本地工作目录 | `C:\Users\Admin\code\lerobot` |
| 本地可用环境 | `C:\Users\Admin\miniconda3\envs\lerobot312` |
| 云端工作目录 | `/root/code/lerobot` |
| 云端 Conda 环境 | `/root/miniconda3/envs/lerobot` |
| 训练硬件 | 1 张 Tesla T4，8 CPU，32 GiB 内存 |
| 训练参数 | ACT，batch size 8，15,000 steps，每 5,000 steps 保存 |
| 最终 checkpoint | `/root/code/lerobot/outputs/train/act_tape_into_pen_holder/checkpoints/015000/pretrained_model` |

数据与模型是两种不同的 Hugging Face 仓库：

- [训练数据集](https://huggingface.co/datasets/zhniu/lerobot_zihao_dataset_a_20260927_202323)：保存示范视频、关节状态、动作和元数据。
- [训练好的 ACT 模型](https://huggingface.co/zhniu/act_tape_into_pen_holder)：保存权重、模型配置和归一化处理器。

不要把先前提供的 `zhniu/wav2vec2_ssd` 当成本次机器人训练的目标仓库。这里使用的是上表中独立的数据集和 ACT 模型仓库。

本次训练已在 2026-09-27 完成 15,000 steps，最终日志 loss 约 0.154，权重已上传并核验。训练循环耗时约 2 小时 5 分钟，上传等额外耗时另计。该时间是本次 T4 实测值，不是其他 GPU 或数据规模的固定耗时。

## 2. Windows：确认本地环境

已有可用环境时优先复用，不需要为了训练重装整套依赖。

在 PowerShell 执行：

```powershell
cd C:\Users\Admin\code\lerobot
& C:\Users\Admin\miniconda3\Scripts\conda.exe env list

$Py = 'C:\Users\Admin\miniconda3\envs\lerobot312\python.exe'
& $Py -c "import sys, torch, lerobot; print(sys.executable); print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(lerobot.__file__)"
& $Py -c "from serial.tools import list_ports; print([(p.device, p.description) for p in list_ports.comports()])"
```

本地环境用于录制，不要求 CUDA 可用；训练时才需要在云端 GPU worker 中验证 CUDA。

本机默认项目 `.venv` 曾出现 `c10.dll / WinError 1114`，而 Conda `lerobot312` 可以正常加载 PyTorch，因此下文 Windows 命令明确使用该环境。普通项目操作可优先使用 `uv run`，但不要让它意外切回这个尚未修复的 `.venv`。若 `conda` 没有加入 PATH，使用上面的完整路径即可。

记录版本，以便后续复现：

```powershell
git rev-parse HEAD
git status --short
& $Py -c "import importlib.metadata as m; print({k:m.version(k) for k in ['lerobot','torch','torchvision','huggingface-hub']})"
```

仓库存在本地改动时，仅记录 Git commit 不足以完整复现；还要保留本次所用配置和本地修改。不要在复现训练前随意更新仓库或升级依赖。

## 3. Windows：硬件与相机检查

1. 接好主臂 COM6、从臂 COM5、电源及两只外置摄像头。
2. 保留现有 ID 对应的校准文件。只有机械安装改变、舵盘重装或校准确实失效时才重新校准。
3. 实际查看编号 0、2 的画面，确认分别对应 `front` 和 `side`，且能看到夹爪、胶带和笔筒。
4. 固定机位、分辨率和光照，后续推理保持一致。

重启或重新插拔后，相机编号可能改变。本机曾只有内置摄像头可用；相机“打开成功”不代表打开了正确的任务相机。

查看 Windows 当前设备：

```powershell
Get-PnpDevice -PresentOnly -Class Camera | Select-Object FriendlyName, Status
& $Py -m lerobot.scripts.lerobot_find_cameras opencv
```

本机重新接好摄像头后，DirectShow 后端能稳定读取编号 0 和 2，因此下面命令显式指定 `backend: 700`。这是本机的实测设置。

先试主从遥操作，确认关节和相机都正常：

```powershell
& $Py -m lerobot.scripts.lerobot_teleoperate `
  --robot.type=so101_follower `
  --robot.port=COM5 `
  --robot.id=zihao_follower_arm `
  --robot.use_degrees=true `
  --robot.cameras='{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: MJPG, backend: 700}, side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG, backend: 700}}' `
  --teleop.type=so101_leader `
  --teleop.port=COM6 `
  --teleop.id=zihao_leader_arm `
  --teleop.use_degrees=true `
  --display_data=true
```

确认正常后停止遥操作，再开始录制；两个进程不能同时占用串口和摄像头。这里统一使用角度单位，夹爪仍使用 0–100 范围。录制、训练后的动作解释及推理必须保持同一单位约定。

## 4. Windows：登录 Hugging Face

```powershell
& C:\Users\Admin\miniconda3\envs\lerobot312\Scripts\hf.exe auth login
& C:\Users\Admin\miniconda3\envs\lerobot312\Scripts\hf.exe auth whoami
```

登录结果应为 `zhniu`。Token 需要有目标仓库的写权限；在交互提示中输入，不要把 Token 写入文档、脚本或聊天记录。本次曾发生过终端输出暴露 Token 的情况，如尚未撤销，应在 HF 设置中撤销并重新登录。

## 5. Windows：录制示范数据

如果只是重训本次已有数据，可以跳过录制，直接使用第 1 节的完整数据集 ID。

以下命令用于录制一批新数据：

```powershell
& $Py -m lerobot.scripts.lerobot_record `
  --robot.type=so101_follower `
  --robot.port=COM5 `
  --robot.id=zihao_follower_arm `
  --robot.use_degrees=true `
  --robot.cameras='{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: MJPG, backend: 700}, side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG, backend: 700}}' `
  --teleop.type=so101_leader `
  --teleop.port=COM6 `
  --teleop.id=zihao_leader_arm `
  --teleop.use_degrees=true `
  --display_data=true `
  --dataset.repo_id=zhniu/lerobot_zihao_dataset_a `
  --dataset.fps=30 `
  --dataset.num_episodes=40 `
  --dataset.single_task='Put the tape into the pen holder' `
  --dataset.episode_time_s=10 `
  --dataset.reset_time_s=2 `
  --dataset.push_to_hub=true
```

PowerShell 的续行反引号后不能有空格。相机配置使用外层单引号，避免内部引号被 Shell 误解析。

本仓库默认会给数据集名称追加时间戳。上次输入基础名称后，实际生成的 ID 是：

```text
zhniu/lerobot_zihao_dataset_a_20260927_202323
```

新一次录制会产生新的后缀。务必从录制日志或 Hugging Face 数据集页面抄下完整 ID，并在训练命令中使用它；不要把示例中的旧后缀当成这次新录制的结果。

每条示范都应完成“靠近胶带 → 抓取 → 移动到笔筒 → 放下”的完整过程。10 秒不够时增加录制时长；2 秒来不及复位时增加 `reset_time_s`。保留成功且动作连贯的示范，避免录入空抓、碰撞或人手遮挡画面。

录制快捷键为右方向键结束当前阶段、左方向键重录当前 episode、Esc 结束录制。按键行为以当前终端提示为准。停止录制后要等待视频编码和上传完成。

## 6. 验证数据确实生成并上传

本次理论帧数：`40 × 10 × 30 = 12,000`。提前结束 episode、重录或中断会影响实际数量，应以元数据为准。

在 Windows 读取 HF 上的元数据并检查文件：

```powershell
@'
import json
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

repo = "zhniu/lerobot_zihao_dataset_a_20260927_202323"
path = hf_hub_download(repo, "meta/info.json", repo_type="dataset")
info = json.loads(Path(path).read_text(encoding="utf-8"))
print({k: info.get(k) for k in ["total_episodes", "total_frames", "fps", "robot_type"]})
print("features:", info["features"])
files = HfApi().list_repo_files(repo, repo_type="dataset")
print("parquet files:", sum(f.endswith(".parquet") for f in files))
print("video files:", sum(f.endswith(".mp4") for f in files))
'@ | & $Py -
```

应看到关节状态 `observation.state`、动作 `action`、`observation.images.front` 和 `observation.images.side`。本次状态与动作维数均为 6。视频可打包存储，文件数量不一定等于 episode 数量。

再抽查开头、中间、结尾以及几条不同 episode 的视频：画面是否对应正确相机、是否模糊或冻结、是否完成任务、状态与动作是否随运动变化。元数据数量正确只能证明数据存在，不能证明示范质量合格。

录制时可以用 `--dataset.root` 显式设置保存路径；未设置时由 LeRobot 缓存配置决定，本机通常位于 `C:\Users\Admin\.cache\huggingface\lerobot` 下。保留原始数据，不要只保留模型。

## 7. SSH 主机：登录与申请 GPU

在 Windows PowerShell 登录：

```powershell
ssh 37944.niuzhibin.ws@merlin-ssh-proxy-cn.byted.org -p 22222
```

进入 SSH 主机后执行 Linux 命令：

```bash
mlx worker list
mlx worker quota
```

如果已有自己可用的空闲 GPU worker，可登录复用；否则根据配额申请一张卡。本次训练成功使用：

```bash
mlx worker launch --gpu 1 --type Tesla-T4 --cpu 8 --memory 32 --resourcetype workspace -- bash
```

GPU 型号需以当时配额输出为准。没有 T4 时，不要盲目重复提交多个申请。曾尝试 A100/V100 时，资源请求及可用资源影响了调度；显式指定 CPU 和内存可避免默认请求过大。

**拿到 worker ID 只表示申请已创建。** `mlx worker list` 中还没有 podIP、无法登录或 `nvidia-smi` 无法看到 GPU 时，不应启动训练。

以后重新连接，先从列表获取当前 ID，再执行：

```bash
mlx worker login <当前worker-ID> -- bash
```

历史训练 worker 为 `4362123`，已不是可以永久使用的地址。模型文件路径和 worker ID 是不同概念，不要硬编码历史 worker。

## 8. GPU worker：选择 Conda 环境并检查 CUDA

以下命令必须在 **GPU worker 内** 执行：

```bash
cd /root/code/lerobot
/root/miniconda3/bin/conda env list
source /root/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

which python
which lerobot-train
nvidia-smi
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPUs:', torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

本次云端环境为 Python 3.12.13、PyTorch 2.7.0+cu118，GPU 为 Tesla T4。`torch.cuda.is_available()` 必须是 `True`。

若 Conda 激活异常，可显式使用已有解释器：

```bash
/root/miniconda3/envs/lerobot/bin/python -c "import torch; print(torch.cuda.is_available())"
```

上述 Windows 与云端已有环境是本次复现路径；本文不要求重新安装 PyTorch。若必须重建环境，要记录并匹配项目代码、依赖与 CUDA/驱动兼容性，不能仅凭一个 LeRobot 包版本号认定环境一致。本地仓库当前代码与服务器训练时的安装元数据可能不同，以实际模块路径、commit、本地修改和保存的训练配置共同核对。

## 9. GPU worker：网络、HF 登录和数据预检

需要通过指定代理访问外网时，在 worker 当前 Shell 设置：

```bash
export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export HTTPS_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

hf auth login
hf auth whoami
```

云端需要单独确认登录身份为 `zhniu`；本地已登录不等于云端也已登录。旧推理环境可能设置了离线模式，下载新数据和上传模型前应取消。只在需要代理的云端环境设置以上地址。

确认能读取实际训练数据集：

```bash
python - <<'PY'
import json
from pathlib import Path
from huggingface_hub import hf_hub_download
repo = 'zhniu/lerobot_zihao_dataset_a_20260927_202323'
info = json.loads(Path(hf_hub_download(repo, 'meta/info.json', repo_type='dataset')).read_text())
print('episodes:', info['total_episodes'], 'frames:', info['total_frames'], 'fps:', info['fps'])
print('features:', list(info['features']))
PY

df -h /root/code/lerobot /root/.cache
```

数据下载缓存、视频、模型 checkpoint 和优化器状态都需要磁盘空间。首次训练会下载数据，ACT 视觉骨干也可能下载 ResNet18 初始权重；初始化期间没有训练步数不一定是卡死，应查看日志。

## 10. GPU worker：启动 ACT 训练

本次关键参数如下：

| 参数 | 值与含义 |
| --- | --- |
| `policy.type` | `act` |
| `policy.device` | `cuda` |
| `batch_size` | 8 |
| `steps` | 15,000 次参数更新 |
| `save_freq` | 每 5,000 steps 保存 checkpoint，结束时也保存 |
| `log_freq` | 每 100 steps 记录指标 |
| `num_workers` | 4 个数据加载进程 |
| `policy.repo_id` | 模型上传位置，不是数据集位置 |
| `policy.push_to_hub` | `true`，训练结束上传模型 |
| `wandb.enable` | `false`，日志保存在服务器 |

本次近似数据遍历次数为 `15,000 × 8 ÷ 12,000 = 10`。这是帧采样意义上的近似 epoch，不是“15,000 条示范”。

以下是已成功运行的训练命令。**原输出目录已存在，重训时必须改用新的输出目录和日志名；如果不想替换线上模型，也同时更换 `policy.repo_id`。** 例如新一次运行可用 `act_tape_into_pen_holder_v2`。

```bash
cd /root/code/lerobot

nohup /root/miniconda3/envs/lerobot/bin/lerobot-train \
  --dataset.repo_id=zhniu/lerobot_zihao_dataset_a_20260927_202323 \
  --policy.type=act \
  --policy.device=cuda \
  --output_dir=outputs/train/act_tape_into_pen_holder \
  --job_name=act_tape_into_pen_holder \
  --batch_size=8 \
  --steps=15000 \
  --save_freq=5000 \
  --log_freq=100 \
  --num_workers=4 \
  --wandb.enable=false \
  --policy.repo_id=zhniu/act_tape_into_pen_holder \
  --policy.push_to_hub=true \
  > /root/code/lerobot/act_tape_train.log 2>&1 < /dev/null &

TRAIN_PID=$!
echo "Training PID: $TRAIN_PID"
```

保存这次打印的 PID。`nohup` 使进程不依赖当前 SSH 连接，但它不能保证 worker 永不被回收；仍需检查 worker、日志和 checkpoint。

## 11. 监控训练

在 worker 内，使用本次实际 PID：

```bash
ps -p "$TRAIN_PID" -o pid,etime,stat
nvidia-smi
tr '\r' '\n' < /root/code/lerobot/act_tape_train.log | grep -E 'step:|checkpoint|End of training|Model pushed|Traceback|ERROR' | tail -n 15
```

重新登录后 `TRAIN_PID` Shell 变量不会自动保留，需使用先前记录的数字。历史 PID `1125` 只属于上次训练，不能直接拿来判断新任务。

训练进度条使用回车刷新，直接 `tail -n 40` 可能输出大量内容；上面的 `tr` 命令会把它转为便于筛选的行。

本次指标变化供参考：

| Step | 日志 loss |
| --- | --- |
| 800 | 2.240 |
| 2,800 | 0.876 |
| 6,400 | 0.299 |
| 10,000 | 0.200 |
| 15,000 | 0.154 |

训练速度约 2 steps/s，日志显存指标约 3.73 GiB；`nvidia-smi` 显示的总占用可能更高。不同机器、数据读取速度和配置都会改变结果。

健康运行的依据是步数持续增加、loss 有限、无异常堆栈且 checkpoint 按计划生成。进程消失不代表成功，要继续看最后日志。loss 不是成功率；本次命令未设置独立验证集，不能据此宣称泛化性能达标。

## 12. 断点恢复与上传补救

### 12.1 从 checkpoint 恢复中断训练

先确认原进程已停止，并找到完整 checkpoint。例如从 10,000 步恢复：

```bash
cd /root/code/lerobot
ls outputs/train/act_tape_into_pen_holder/checkpoints/010000

nohup /root/miniconda3/envs/lerobot/bin/lerobot-train \
  --config_path=/root/code/lerobot/outputs/train/act_tape_into_pen_holder/checkpoints/010000/pretrained_model/train_config.json \
  --resume=true \
  --steps=15000 \
  > /root/code/lerobot/act_tape_resume_010000.log 2>&1 < /dev/null &
```

`steps=15000` 是目标总步数，不是在原 10,000 步之外再训练 15,000 步。精确续训需要同一 checkpoint 的模型与 `training_state`，其中包含优化器等状态。只有 Hub 上的推理权重，不一定能完整恢复原训练过程。

### 12.2 训练完成但上传失败

保留本地最终 checkpoint，修复网络或 HF 登录权限后补传，无需重新训练。以下操作会更新目标模型仓库，先确认要上传的版本：

```bash
hf upload zhniu/act_tape_into_pen_holder \
  /root/code/lerobot/outputs/train/act_tape_into_pen_holder/checkpoints/015000/pretrained_model \
  . --repo-type model
```

## 13. 验证训练完成与模型完整性

本次成功日志包含：

```text
step:15K ... loss:0.154
End of training
Model pushed to https://huggingface.co/zhniu/act_tape_into_pen_holder
```

检查服务器文件：

```bash
ls -lh /root/code/lerobot/outputs/train/act_tape_into_pen_holder/checkpoints/015000/pretrained_model
```

再独立检查 HF 模型仓库，不能只依据上传日志：

```bash
python - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
repo = 'zhniu/act_tape_into_pen_holder'
info = api.model_info(repo)
files = set(api.list_repo_files(repo, repo_type='model'))
required = {
    'model.safetensors', 'config.json', 'train_config.json',
    'policy_preprocessor.json', 'policy_postprocessor.json',
    'policy_preprocessor_step_3_normalizer_processor.safetensors',
    'policy_postprocessor_step_0_unnormalizer_processor.safetensors',
}
print('repo:', info.id, 'revision:', info.sha)
print('files:', sorted(files))
missing = required - files
assert not missing, f'Missing: {missing}'
print('Model files verified')
PY
```

以上处理器状态文件名是本次 ACT 产物的实际名称。其他模型或版本的处理器步骤编号可能不同，检查时应以其 JSON 配置引用的文件为准。

保存 `train_config.json`、模型 revision、数据集 revision、代码版本和训练日志，才能明确下一次推理到底使用哪份模型。上传模型目录时需要一并上传归一化状态，不能只复制 `model.safetensors`。

## 14. 训练后进入推理测试

本项目当前把遥操作数据采集放在 `lerobot-record`，模型实机运行放在 `lerobot-rollout`。旧文档中的 `lerobot-record --policy.path=...` 不适用于当前本地入口。

本次模型已经做过一次本地 CPU 的 10 秒限速试跑：夹爪抬起并张开，但未抓到胶带、未完成放入笔筒。控制频率约 20.5 Hz，存在约 1.1 秒推理停顿。它证明模型可加载并驱动硬件，不能证明任务成功，也不能据一次限速试跑判断模型最终能力。

下一阶段采用“本地相机和关节状态 → 云端 ACT GPU 推理 → 本地执行 action”的方式。详细步骤见同目录的 [ACT_REMOTE_GPU_RUNBOOK.md](ACT_REMOTE_GPU_RUNBOOK.md)，客户端为 [act_remote_rollout.py](act_remote_rollout.py)。

在云端服务和 SSH 隧道就绪后，先做不执行动作的通信验证：

```powershell
cd C:\Users\Admin\code\lerobot
& C:\Users\Admin\miniconda3\envs\lerobot312\python.exe -u act_remote_rollout.py
```

验证通过且有人现场监护，再执行限时试跑：

```powershell
& C:\Users\Admin\miniconda3\envs\lerobot312\python.exe -u act_remote_rollout.py --execute --seconds 10
```

本文编写时，远程 GPU 推理尚未完成实机闭环验证，GPU 申请仍待就绪。推理成功率应通过多次完整试验记录；分别统计抓取成功、放置成功和整个任务成功，记录物品初始位置及失败画面，再决定补数据或调整训练。

训练用 GPU 在模型及恢复文件核验、备份完成后，可释放明确属于本次任务且不再用于推理的 worker；不要释放仍有其他任务运行的 worker。需要马上继续云端推理时可以复用可用 worker。

## 15. 常见问题速查

| 现象 | 优先检查 |
| --- | --- |
| 找不到 COM5/COM6 | USB、电源、驱动、设备管理器及实际串口枚举 |
| 相机 2 无法打开 | 外置相机是否接好、编号是否变化、是否被其他进程占用；本机使用 DirectShow |
| `c10.dll` 初始化失败 | 是否误用了项目 `.venv`；先用已验证的 Conda `lerobot312` |
| HF 401/403 | 当前环境登录账号、Token 权限、目标仓库类型及代理连接 |
| 数据集找不到 | 使用完整时间戳 ID，确认 `repo_type=dataset` |
| worker 有 ID 但无 podIP | 仍在等待资源；查询 quota，避免反复提交重复申请 |
| CUDA 为 False | 是否在 GPU worker 内、是否用了正确 Conda 环境和 CUDA 版 PyTorch |
| 启动时下载很久 | 数据/ResNet 权重首次下载、代理或误开启 HF 离线模式 |
| 显存不足 | 降低 batch size；注意同一步数下样本更新量也会变化 |
| 输出目录已存在 | 新训练换目录；中断恢复使用完整 checkpoint 和 `resume=true` |
| 训练进程退出 | 查看日志末尾，区分正常完成、异常退出和上传失败 |
| loss 很低但抓取失败 | 检查示范质量、相机映射、关节单位、物体位置和推理时延；需要实机评估 |

完成一次训练的验收点是：数据内容已抽查、GPU 和环境已验证、目标步数已达到、最终 checkpoint 完整、HF 模型文件可访问。机械臂成功完成任务是后续实机评估的独立验收点。
