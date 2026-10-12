# SO-101 VLA-JEPA 仿真运行指南

本文说明如何运行仓库根目录的 `so101_jepa_sim_eval.py`，使用训练好的 VLA-JEPA 模型控制 MuJoCo 中的 SO-101。

脚本后台执行仿真并保存视频，不弹出交互窗口，不连接真实机械臂。

## 1. 运行环境

已验证的运行环境为远端 Linux，使用 MuJoCo 3.3.7、OSMesa 软件渲染，以及现有 `lerobot` Python 环境。CPU/float32 推理已跑通。

本机 Windows 曾在 MuJoCo 3.3.7 和 3.15.0 上遇到 DLL 初始化失败，因此下面采用已配置好的远端环境。这些命令用于复用现有环境，并不是一台新机器的完整安装流程。

远端需要保留以下文件：

- 脚本：`/root/code/lerobot/so101_jepa_sim_eval.py`
- SO-101 模型：`/root/code/lerobot/outputs/so101_sim_assets_20261012/robotstudio_so101/so101.xml` 及其 `assets` 目录。
- 最终检查点：`/root/code/lerobot/outputs/train/vla_jepa_tape_20261011_r1/checkpoints/040000/pretrained_model`
- 模型配置引用的 Qwen/V-JEPA 本地缓存。仅复制检查点目录到另一台机器，不一定能离线运行。

机器人模型来源：<https://github.com/SinfonIAUniandes/lerobot_robot_so101_mujoco>，使用提交 `cc577ad7f35a7ad477f6bb7695390ec6da54d46f` 的资产。

## 2. 登录服务器

在本机 PowerShell 中执行：

```powershell
ssh -p 22222 37944.niuzhibin.ws@merlin-ssh-proxy-cn.byted.org
```

以下命令均在登录后的 Linux shell 中执行。

## 3. 使用训练好的模型运行一轮

复制整段执行：

```bash
cd /root/code/lerobot

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

/root/.local/bin/uv run --no-project \
  --python /root/miniconda3/envs/lerobot/bin/python \
  python so101_jepa_sim_eval.py \
  --asset outputs/so101_sim_assets_20261012/robotstudio_so101/so101.xml \
  --checkpoint outputs/train/vla_jepa_tape_20261011_r1/checkpoints/040000/pretrained_model \
  --device cpu \
  --steps 300 \
  --episodes 1 \
  --output "outputs/jepa_sim_$(date +%Y%m%d_%H%M%S)"
```

运行时保持 SSH 窗口打开。终端每 30 步输出一次进度，一轮结束时输出 `EPISODE_COMPLETE`。

本命令加载已经训练完成的 40,000 步检查点，进行推理，不重新训练模型。

## 4. 参数说明

| 参数 | 含义 |
| --- | --- |
| `--asset` | 必填，SO-101 的 MuJoCo 模型 XML 路径 |
| `--checkpoint` | 训练模型目录；省略时只做物理保持测试 |
| `--device` | `cpu` 或 `cuda`；脚本默认 `cuda`，上方命令显式选择 `cpu` |
| `--steps` | 每轮控制步数，默认 300 |
| `--episodes` | 测试轮数，默认 3；上方命令显式选择 1 |
| `--output` | 必填，输出目录，必须是尚不存在的新目录 |

控制频率为 30 Hz，物理步长为 1/300 秒，每个控制步执行 10 个物理步。因此：

- `--steps 300`：10 秒仿真时间。
- `--steps 9000`：300 秒仿真时间。
- `--episodes 3`：执行三轮，每轮重置环境并使用不同种子轻微随机化物体位置。

仿真时间不等于实际计算时间。上一轮 CPU 测试的 300 步循环耗时约 393.94 秒，另需模型加载时间；更长测试的耗时会相应增加。

使用 GPU 时，应先在入口查询 `mlx worker list`，确认当前 GPU worker 的实际地址和端口，再连接对应 worker 验证 CUDA。不要沿用历史端口，也不要直接假定登录节点有 GPU。确认可用后，使用相同命令并将 `--device cpu` 改为 `--device cuda`。

## 5. 只验证物理环境和渲染

在第 3 节环境变量已设置的前提下执行：

```bash
cd /root/code/lerobot

/root/.local/bin/uv run --no-project \
  --python /root/miniconda3/envs/lerobot/bin/python \
  python so101_jepa_sim_eval.py \
  --asset outputs/so101_sim_assets_20261012/robotstudio_so101/so101.xml \
  --device cpu \
  --steps 60 \
  --episodes 1 \
  --output "outputs/jepa_physics_$(date +%Y%m%d_%H%M%S)"
```

这里没有 `--checkpoint`，机械臂保持初始关节目标，不调用训练模型。这只能验证场景、物理和渲染链路。

## 6. 输出文件

| 文件 | 内容 |
| --- | --- |
| `episode_00.mp4` | 第一轮双相机视频，左侧腕部相机、右侧外部相机，1280×480、30 fps |
| `episode_00.json` | 每步关节状态、模型动作和物体位置 |
| `report.json` | 每轮耗时、物体最高位置、夹爪接触计数、命令裁剪计数和入筒判定 |
| `initial_front.png` | 初始腕部相机画面 |
| `initial_side.png` | 初始外部相机画面 |
| `scene.xml` | 自动生成的场景，包含当前远端资产的绝对路径 |

多轮测试的视频及轨迹依次命名为 `episode_01`、`episode_02` 等。`report.json` 在每轮完整结束后写入；运行被提前打断时，不能假定已有完整报告。

若需要把新结果取回本机，在另一个 PowerShell 窗口中执行以下命令，并将时间戳替换为实际输出目录名：

```powershell
scp -P 22222 -r "37944.niuzhibin.ws@merlin-ssh-proxy-cn.byted.org:/root/code/lerobot/outputs/jepa_sim_YYYYMMDD_HHMMSS" "C:\Users\Admin\code\lerobot\outputs\"
```

此前已完成的一轮结果已经保存在本机：

```text
C:\Users\Admin\code\lerobot\outputs\jepa_sim_cpu_20261012\
```

## 7. 已验证结果与限制

2026-10-12 的首轮测试使用最终 040000 模型，完成了 300 步闭环运行，退出码为 0。动作形状为 6 维且数值有限，物理状态有限，关节命令裁剪次数为 0。

该轮机械臂和夹爪有运动，但没有抓起物体：夹爪接触计数为 0，没有抬升，也没有放入容器。

当前场景使用圆柱体代替胶带、开放方形容器代替笔筒，尚未拟合真实机械臂的关节校准零位、相机外参和场景外观。该测试用于验证模型调用与反馈闭环，不能据此估计实机抓取成功率。

## 8. 常见问题

- **`FileExistsError`**：`--output` 指向已有目录。换一个新目录名即可，不需要删除旧结果。
- **`CUDA unavailable`**：当前机器没有可用 CUDA。使用 `--device cpu`，或连接已确认可用的 GPU worker。
- **离线模型加载失败**：检查检查点和配置引用的基础模型缓存是否完整，单独复制权重文件不够。
- **Windows DLL 初始化失败**：本机此前出现过该问题，本文使用已验证的远端 Linux 环境。
- **没有弹出模拟器窗口**：这是脚本的正常行为。脚本采用离屏渲染，结果写入输出目录的视频和图片。
