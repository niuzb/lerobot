# 使用远端 GPU 的 SmolVLA 控制本地 SO-101

本文记录 2026-09-25 在 Windows 本地连接 SO-101、在远端 MLX worker 使用 GPU 运行 SmolVLA，并通过异步推理驱动机械臂完成“把方块放进杯子”任务的完整链路。

本文既是本次实验记录，也是下一次复现时的操作手册。命令中的 worker IP、摄像头编号和模型路径必须按实际环境核对，不能直接假定永久不变。

## 1. 系统结构

```text
本地 Windows
  SO-101 / COM5
  camera1 = 外部固定相机（本次为 OpenCV index 2）
  camera2 = 腕部相机（本次为 OpenCV index 0）
  conda env = lerobot312
          │
          │ gRPC 127.0.0.1:18080
          ▼
本地 SSH 端口转发
          │
          ▼
merlin SSH 主机 127.0.0.1:50052
  Python TCP relay
          │
          │ worker IPv6:50051
          ▼
MLX GPU worker
  A100-SXM-80GB
  LeRobot PolicyServer
  SmolVLA checkpoint
```

模型推理在远端 GPU 上完成；摄像头采集、机械臂状态读取和电机控制始终在本地机器上完成。

## 2. 本次验证过的配置

| 项目 | 值 |
| --- | --- |
| SSH | `37944.niuzhibin.ws@merlin-ssh-proxy-cn.byted.org:22222` |
| GPU worker | `A100-SXM-80GB`，1 张 GPU，4 CPU，32 GB RAM |
| worker ID | `4340276`（仅本次有效） |
| PolicyServer | worker TCP `50051` |
| 主机 relay | SSH 主机 `127.0.0.1:50052` |
| 本地入口 | `127.0.0.1:18080` |
| 本地 Python | conda 环境 `lerobot312`，Python 3.12 |
| 机械臂 | SO-101 follower，`COM5`，校准 ID `zihao_follower_arm` |
| 外部固定相机 | OpenCV index `2`，对应模型 `camera1` |
| 腕部相机 | OpenCV index `0`，对应模型 `camera2` |
| 图像 | 640 × 480，30 FPS，MJPG |
| 策略 | SmolVLA，action chunk 50；异步服务每次返回 10 个动作 |
| checkpoint | `/root/code/lerobot/output_lerobot_train_20260704_140211/checkpoints/040000/pretrained_model` |

该 checkpoint 来自 `youliangtan/so101-table-cleanup`，训练任务是把笔、马克笔、胶带和剪刀放进笔筒。方块放进杯子属于跨物体、跨容器泛化，不保证一次成功。要获得稳定成功率，最终应采集方块与杯子的本机数据并微调模型。

## 3. 机械臂和 5 号电机结论

5 号 `wrist_roll` 电机按正常硬件处理，依据如下：

- 断电后能够顺畅转动，没有明显卡滞或异响。
- 线缆没有被拧紧。
- 指示灯是常亮红灯，而不是故障时常见的闪烁状态。
- 本次异常与整圈位置编码跨越 `0/4095` 边界时的软件处理有关，不能仅凭一次 `Overload error` 判定电机损坏。

STS3215 是 12 位、每圈 4096 个计数的绝对位置舵机。`wrist_roll` 是整圈关节；控制程序如果把 `-180°` 和 `+180°` 当作普通线性区间，跨零点时就可能把很小的实际变化解释成接近整圈的跳变。

因此正常运行时可以控制 5 号电机，但需要满足以下条件：

1. 开始前把腕旋关节放在远离 `-180°/+180°` 接缝的中间区域。
2. 首次动作使用较小的单周期限幅。
3. 控制代码必须对整圈角度使用环绕感知的差值，或在原始编码器空间检查目标步长。
4. 一旦出现突跳、异响、红灯闪烁或线缆开始缠绕，立即切断机械臂电源。

## 4. 安全规则

- 机械臂运动时必须有人在电源开关旁观察，不做无人值守运行。
- 杯子、方块和机械臂周围不能有人手、松散线缆或其他障碍物。
- 先完成网络测试、模型推理测试和无扭矩 dry-run，再启用运动。
- 第一次运动只运行 2–5 秒，确认方向正确后再增加时间。
- 使用 `max_relative_target` 限制每个控制周期的最大关节变化。
- 使用 `max_position_jump=5.0` 拦截单周期超过 5° 的异常反馈；触发时冻结上一帧稳定姿态并跳过本周期写目标。
- 如果日志反复出现 `Ignoring implausible Present_Position jump`，只读检查总线和编码器，不得继续延长运动时间。
- 不要依靠 `Ctrl+C` 作为唯一急停方式；真正异常时直接切断机械臂电源。
- 远端 GPU worker 可以继续运行，机械臂断电不会影响模型服务。

## 5. 本地环境准备

在 PowerShell 中进入仓库并激活环境：

```powershell
cd C:\Users\Admin\code\lerobot
conda activate lerobot312
```

安装本地项目和异步推理依赖：

```powershell
C:\Users\Admin\.local\bin\uv.exe pip install `
  --python C:\Users\Admin\miniconda3\envs\lerobot312\python.exe `
  -e ".[async]"
```

确认串口和摄像头：

```powershell
lerobot-find-port
lerobot-find-cameras opencv
```

本次检测结果：

- `COM5`：SO-101 电机总线。
- index `0`：腕部相机，应作为 `camera2`。
- index `1`：笔记本内置相机，不使用。
- index `2`：外部固定相机，应作为 `camera1`。

Windows 下同时打开两路相机时，本次需要先创建 index `0`，再创建 index `2`，并设置 `fourcc="MJPG"`。

校准文件应存在：

```text
C:\Users\Admin\.cache\huggingface\lerobot\calibration\robots\so_follower\zihao_follower_arm.json
```

不要在每次运行前重新校准。只有在机械安装位置发生变化、舵盘拆装或校准文件确认错误时才重新校准。

## 6. 登录远端并申请 GPU worker

在本地 PowerShell 的终端 A 中登录 SSH 主机：

```powershell
ssh 37944.niuzhibin.ws@merlin-ssh-proxy-cn.byted.org -p 22222
```

申请 worker：

```bash
mlx worker launch \
  --cpu 4 \
  --memory 32 \
  --gpu 1 \
  --type A100-SXM-80GB \
  --resourcetype workspace \
  -- bash
```

本次没有使用最初计划的 V100：默认资源申请量超过当时可用资源；A100 若不显式降低 CPU 数量也会申请失败。因此把 CPU 和内存明确写出。

进入 worker 后检查 GPU：

```bash
nvidia-smi
```

记录 worker 的 IPv6 地址：

```bash
ip -6 -brief address
```

本次使用过的实际地址是：

```text
fdbd:dc03:16:341:2000::31
```

该地址是动态值，下次必须重新读取。

## 7. 在 worker 中启动 SmolVLA PolicyServer

仍在 worker 中执行：

```bash
cd /root/code/lerobot
source /root/code/lerobot/active_lerobot.sh

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
```

离线变量很重要。本次 checkpoint 和基础模型已经在缓存中；不设置离线模式时，`AutoProcessor` 曾尝试访问 Hugging Face Hub 并长时间等待。

启动服务：

```bash
python -m lerobot.async_inference.policy_server \
  --host='[::]' \
  --port=50051 \
  --fps=30 \
  --inference_latency=0 \
  --obs_queue_timeout=5
```

不要使用本次环境中的 `8080`，因为该端口已被平台 sidecar 占用。服务器启动时并不立即加载模型；本地客户端第一次握手后，服务器才会按客户端给出的 checkpoint 和设备加载策略。

已验证的性能：

- 模型放入 A100 约 10–12 秒。
- 单次异步推理约 0.27–0.31 秒。
- 返回动作形状为 `[1, 10, 6]`。

### 7.1 观测传输压缩与最新帧队列

客户端发送观测时会自动使用 JPEG wire format（默认质量 `80`）。远端
`PolicyServer` 收到后解码回原来的 RGB `uint8` 数组，因此策略预处理接口不变。

观测发送由独立线程执行，待发送队列长度固定为 `1`：网络发送较慢时，新观测会替换尚未发送的旧观测，控制循环不会被 gRPC 阻塞。日志会周期性打印：

- `interval_fps` / `avg_fps`：实际发送帧率；
- `payload`、`jpeg_images`、`send_ms`：传输大小、压缩图像数和耗时；
- `dropped_pending`：因“只保留最新帧”被丢弃的旧观测数量。

只读验证（不调用 `robot.connect()`、不使能扭矩）：

```powershell
C:\Users\Admin\miniconda3\envs\lerobot312\python.exe -u observation_stream_test.py
```

详细发送日志写入 `outputs/observation_stream.log`。本次双路 640×480 图像的 payload 从约 1.80 MiB 降至约 35.6 KiB，观测发送速率从约 2.6 FPS 提升至约 13.4 FPS；远端成功解码并返回 `[1, 10, 6]` action chunk。

## 8. 建立 worker 到 SSH 主机的 TCP relay

平台不允许从本地直接 SSH 转发到 worker，因此需要在 SSH 主机上增加一跳。

在本地 PowerShell 的终端 B 中再次登录 SSH 主机：

```powershell
ssh 37944.niuzhibin.ws@merlin-ssh-proxy-cn.byted.org -p 22222
```

在 SSH 主机中，把 `WORKER_IPV6` 替换为刚刚读取的实际值，然后运行 relay：

```bash
export WORKER_IPV6='fdbd:dc03:16:341:2000::31'

python - "$WORKER_IPV6" <<'PY'
import socket
import socketserver
import sys
import threading

worker_host = sys.argv[1]
worker_port = 50051


def copy_stream(source, destination):
    try:
        while data := source.recv(1024 * 1024):
            destination.sendall(data)
    finally:
        try:
            destination.shutdown(socket.SHUT_WR)
        except OSError:
            pass


class Relay(socketserver.BaseRequestHandler):
    def handle(self):
        with socket.create_connection((worker_host, worker_port)) as upstream:
            reverse = threading.Thread(
                target=copy_stream,
                args=(upstream, self.request),
                daemon=True,
            )
            reverse.start()
            copy_stream(self.request, upstream)
            reverse.join()


class ThreadingTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


with ThreadingTCPServer(("127.0.0.1", 50052), Relay) as server:
    print(f"relay 127.0.0.1:50052 -> [{worker_host}]:{worker_port}", flush=True)
    server.serve_forever()
PY
```

保持终端 B 打开。

## 9. 建立本地 SSH 端口转发

在本地 PowerShell 的终端 C 中执行：

```powershell
ssh -N `
  -L "18080:127.0.0.1:50052" `
  37944.niuzhibin.ws@merlin-ssh-proxy-cn.byted.org `
  -p 22222
```

保持终端 C 打开。此后本地客户端只需要连接 `127.0.0.1:18080`。

另开一个 PowerShell 验证 TCP：

```powershell
Test-NetConnection 127.0.0.1 -Port 18080
```

验证 gRPC channel：

```powershell
conda run -n lerobot312 python -c "import grpc; c=grpc.insecure_channel('127.0.0.1:18080'); grpc.channel_ready_future(c).result(timeout=10); print('gRPC ready')"
```

只有看到 `gRPC ready` 后才继续。

## 10. 摄像头摆放和特征映射

该 checkpoint 的预处理器使用下面的图像特征：

```text
camera1 = 外部固定视角
camera2 = 腕部视角
```

本地配置必须直接使用这两个名字。不要把摄像头命名为 `front` 和 `wrist` 后期待远端自动重命名，因为当前异步握手会用客户端发送的 feature spec 覆盖 checkpoint 中的 rename map。

摄像头字典按以下顺序创建：

```python
camera_cfg = {
    "camera2": OpenCVCameraConfig(
        index_or_path=0,
        width=640,
        height=480,
        fps=30,
        fourcc="MJPG",
    ),
    "camera1": OpenCVCameraConfig(
        index_or_path=2,
        width=640,
        height=480,
        fps=30,
        fourcc="MJPG",
    ),
}
```

执行前必须确认：

- `camera1` 同时看见机械臂工作区、方块和杯子。
- `camera2` 能看见夹爪前方的方块或杯口。
- 杯子与方块尽量使用与训练数据相似的尺寸、对比度和桌面光照。
- 外部相机不能只拍到空桌面；本次初始摆放曾出现该问题。

## 11. 第一阶段：不接机械臂的模型检查

在真正连接机械臂前，先确认以下条件：

1. checkpoint 路径存在。
2. server 日志显示模型加载到 `cuda`。
3. 一份 6 维状态和两幅图像能够得到 `[1, 10, 6]` 的动作。
4. 推理过程中显存稳定，没有重复下载模型。

本次已经完成这一步：40k checkpoint 可以在 A100 上推理，GPU 占用约 0.9 GB。

## 12. 第二阶段：只读观测和 prediction-only dry-run

先验证本地串口与相机，且不调用 `robot.connect()` 或 `send_action()`。直接连接底层总线不会执行 `configure()`，因此不会启用扭矩：

```python
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

camera_cfg = {
    "camera2": OpenCVCameraConfig(
        index_or_path=0, width=640, height=480, fps=30, fourcc="MJPG"
    ),
    "camera1": OpenCVCameraConfig(
        index_or_path=2, width=640, height=480, fps=30, fourcc="MJPG"
    ),
}

robot = SO101Follower(
    SO101FollowerConfig(
        port="COM5",
        id="zihao_follower_arm",
        cameras=camera_cfg,
    )
)

robot.bus.connect()
for camera in robot.cameras.values():
    camera.connect()

try:
    observation = robot.get_observation()
    print({key: value for key, value in observation.items() if key.endswith(".pos")})
    print("camera1:", observation["camera1"].shape)
    print("camera2:", observation["camera2"].shape)
finally:
    robot.bus.disconnect(disable_torque=True)
    for camera in robot.cameras.values():
        camera.disconnect()
```

这一步只验证状态和图像，不会把观测发送给远端模型。远端 prediction-only 测试应使用已保存的观测或数据集样本调用策略，打印预测动作，不能进入 `RobotClient.control_loop()`，因为该循环会自动调用 `robot.send_action()`。

需要使用的策略配置如下；它用于检查 checkpoint、设备和 action chunk 参数，不代表此时可以启动完整控制循环：

```python
client_cfg = RobotClientConfig(
    robot=robot_cfg,
    server_address="127.0.0.1:18080",
    policy_type="smolvla",
    pretrained_name_or_path=(
        "/root/code/lerobot/"
        "output_lerobot_train_20260704_140211/"
        "checkpoints/040000/pretrained_model"
    ),
    policy_device="cuda",
    client_device="cpu",
    actions_per_chunk=10,
    chunk_size_threshold=0.5,
    fps=30,
    aggregate_fn_name="weighted_average",
    task="Grab the cube and place it into the cup",
)
```

检查至少一个 action chunk：

- 动作都是有限数值，没有 `NaN` 或 `inf`。
- 关节顺序是 6 维 SO-101 顺序。
- 初始目标与当前姿态差距是否过大。
- 相机画面与任务描述是否匹配。

本次模型曾给出较大的初始姿态差，例如肘关节目标与当前值相差约 37°。这说明必须使用逐周期限幅，不能直接把第一帧策略目标原样写给电机。

## 13. 第三阶段：短时受限运动

在 5 号电机按正常硬件参与控制时，正常配置使用：

```python
robot_cfg = SO101FollowerConfig(
    port="COM5",
    id="zihao_follower_arm",
    cameras=camera_cfg,
    max_relative_target={
        "shoulder_pan": 0.10,
        "shoulder_lift": 0.10,
        "elbow_flex": 0.10,
        "wrist_flex": 0.10,
        "wrist_roll": 0.05,
        "gripper": 0.10,
    },
    max_position_jump=5.0,
)
```

这里的数值是每个控制周期的最大变化，不是整个动作的最大变化。30 FPS 下，`0.10°/周期` 的理论上限约为 `3°/秒`。

首次动作流程：

1. 机械臂断电，把姿态放在工作区中间，确保腕旋关节远离编码器接缝。
2. 方块和杯子放入两路摄像头视野。
3. 人站在电源开关旁，先给机械臂上电。
4. 确认 6 个电机均为稳定常亮，没有闪烁。
5. 启动客户端，只运行 2 秒。
6. 观察运动方向、线缆、声音和指示灯。
7. 无异常时停机检查，再逐步增加到 5 秒、10 秒。

如果终端出现 `Ignoring implausible Present_Position jump`，说明控制周期读到了不可信的反馈；程序会跳过该周期的写入，但这不是已经修复硬件或总线问题的证明。立即停止本次实验，断电后重新做原始位置读取。

任务文本使用：

```text
Grab the cube and place it into the cup
```

如果模型对该文本反应不稳定，可以尝试更接近训练分布的表达，但必须通过 dry-run 比较动作，不能边运动边随意改 prompt：

```text
Grab the object and place it into the holder
```

## 14. 客户端命令参考

在摄像头映射、整圈安全处理和短时停止机制都确认后，可用下面的 PowerShell 命令作为正式客户端的基础：

```powershell
conda run -n lerobot312 python -m lerobot.async_inference.robot_client `
  --server_address=127.0.0.1:18080 `
  --robot.type=so101_follower `
  --robot.port=COM5 `
  --robot.id=zihao_follower_arm `
  --robot.cameras="{camera2: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: MJPG}, camera1: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}}" `
  --robot.max_relative_target=0.1 `
  --task="Grab the cube and place it into the cup" `
  --policy_type=smolvla `
  --pretrained_name_or_path=/root/code/lerobot/output_lerobot_train_20260704_140211/checkpoints/040000/pretrained_model `
  --policy_device=cuda `
  --client_device=cpu `
  --actions_per_chunk=10 `
  --chunk_size_threshold=0.5 `
  --aggregate_fn_name=weighted_average `
  --fps=30
```

该标准 CLI 会持续运行，适合完成所有短时安全验证之后使用。第一次带电实验应使用带自动超时的包装脚本，不要直接无限运行。

## 15. 异常处理

### 15.1 电机红灯闪烁、机械臂卡住或突然快速旋转

1. 立即切断机械臂电源。
2. 不要依赖程序发送 torque disable。
3. 保留 PolicyServer 日志和本地客户端日志。
4. 断电检查关节是否顺畅、线缆是否缠绕。
5. 下次只能从无扭矩 dry-run 重新开始。

### 15.2 `Overload error`

它表示舵机检测到过载保护条件，不等同于电机已经损坏。检查机械阻挡、控制目标跳变、电压和编码器跨零点问题。机械检查正常且灯不闪烁时，可按正常硬件继续排查控制软件。

### 15.3 远端模型加载时卡住

确认设置：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

并确认基础模型已经在远端缓存。

### 15.4 本地无法连接 `18080`

按顺序检查：

1. worker 中 `PolicyServer` 是否仍在运行。
2. worker IPv6 是否变化。
3. SSH 主机的 relay 是否仍在运行。
4. 本地 `ssh -N -L` 是否仍在运行。
5. `Test-NetConnection 127.0.0.1 -Port 18080` 是否成功。

### 15.5 两路摄像头无法同时打开

- 先创建 index `0`，再创建 index `2`。
- 使用 640 × 480、30 FPS、MJPG。
- 关闭占用摄像头的 Camera、Teams、Zoom 或浏览器页面。

## 16. 正常停机顺序

1. 停止本地 RobotClient。
2. 确认机械臂不再运动，然后切断机械臂电源。
3. 停止本地 SSH tunnel（终端 C）。
4. 停止 SSH 主机 relay（终端 B）。
5. 停止 worker 中的 PolicyServer。
6. 退出或释放 MLX worker，避免继续占用 GPU。

如果只是暂时调整桌面物体，可以保持远端 GPU 服务和两层网络转发，只关闭本地 RobotClient 并切断机械臂电源。

## 17. 下一步建议

当前链路已经证明：本地真实相机和 SO-101 状态可以送到远端 A100，SmolVLA 可以返回 6 维动作，本地也可以接收并执行动作。下一步应按以下顺序推进：

1. 查明 `Goal_Position` 写入后 `elbow_flex` 反馈从约 3042 计数连续降到约 2670 计数的现象；重复出现时不得进入长时间动作。
   当前只读扭矩测试稳定，而写回当前目标也能触发该现象，因此不能仅归因于策略输出或单次编码器读数。
2. 完成 `wrist_roll` 的环绕感知限幅或原始编码器步长检查。
3. 调整 `camera1`，确保固定视角同时覆盖方块、杯子和夹爪。
4. 进行全电机无扭矩 dry-run。
5. 进行 2 秒、低速、有人值守的受限动作。
6. 如果零样本 SmolVLA 无法稳定完成任务，采集本机“方块放入杯子”示范并微调 checkpoint。

## 2026-09-27 更新：按时间推进目标

`short_rollout.py` 已用时间斜坡替换每次相对反馈的 `0.5°/action` 限幅。目标从上次成功写出的目标出发，按速度 × 经过时间逼近模型目标，同时限制它与当前实测位置的偏差。

| 关节 | 目标推进速度 | 最大跟踪误差 |
| --- | --- | --- |
| 肩旋、肩抬、肘、腕俯仰 | 3°/秒 | 3° |
| 腕旋 | 1°/秒 | 1° |
| 夹爪（0～100 开度） | 3 个百分点/秒 | 3 个百分点 |

- 首次指令保持实测位置；重连后重新初始化。
- 单次时间步长最多按 0.1 秒计算，通信暂停不会积累大幅跳变。
- 目标到达跟踪误差上限后不继续累积；反馈跟上后再推进。
- 保留位置跳变检查；无法同时满足速度和跟踪误差约束时抛错，短时运行脚本结束控制并断开连接。
- 日志分别记录 `requested`（模型目标）、`performed`（写出目标）、`feedback_before_write`（写入前实测位置）。写出目标不代表实际运动。

本次只完成代码和模拟反馈测试，未执行实机动作。短时脚本仍为 20 秒；本节参数取代之前该脚本的逐 action 限幅参数。

### 短时运行结束后复位

`short_rollout.py` 在模型控制开始前读取并记录本次初始关节位置（包含夹爪）。正常计时结束后，先停止控制线程，再通过相同的时间斜坡和跟踪误差上限返回初始位置，最后断开机械臂。复位最多 60 秒，额外计入运行时间；要求实测误差连续 0.5 秒保持在普通关节 0.5°、夹爪 1 个开度百分点以内。`RESET` 日志记录初始位置、复位目标、反馈和结果。

Ctrl+C、提前退出、控制异常或控制线程未停止时跳过自动复位；复位中反馈保护拒绝、通信报错或超时则停止复位并清理连接。复位是关节空间返回，不是避障轨迹，也不保证重走原路径；需保持返回路径通畅，注意夹爪恢复原开度可能释放物体。断开连接后按现有配置卸扭矩，因此不保证继续保持初始姿态。

### 策略单位转换（2026-09-27）

`short_rollout.py` 已显式启用 `policy_joint_units="range_m100_100"`，并保持机器人 `use_degrees=True`：

1. 状态输入：实测角度通过本机校准范围换算为前五维 -100～100；夹爪保持 0～100。
2. 模型返回及队列融合：继续使用策略范围单位。服务端 MEAN_STD 反归一化不等同于这里的物理单位转换，不修改服务端处理器。
3. 执行输出：策略范围先裁至合法范围，再按本机校准换算回角度，之后调用原有 send_action 进行时间平滑与跟踪误差限制。
4. 初始位置记录、复位和到位误差仍是驱动单位，不经过策略转换。因此现有 3°/秒、腕旋 1°/秒等保护参数不变。

换算因子为 `(range_max-range_min)*360/(resolution-1)/200`，支持驱动方向标志；夹爪不做二次归一化。禁止把已经使用范围单位的驱动再次接入转换器。非有限值、缺失校准、无效范围和动作维度错误会拒绝执行。

动作日志新增 `policy_action`（队列融合后的策略单位），`requested` 改为转换后的驱动目标，`performed` 是限幅后写出目标，`feedback_before_write` 是实测位置。后三者前五维均为度、夹爪为开度百分点。其他客户端默认 `policy_joint_units="robot"`，不启用转换。

本修改已做离线转换及模拟控制测试，未运行实机。它修正单位边界，不保证消除训练/本机校准端点、相机视角和机械结构差异。

## 18. 参考资料

- [LeRobot asynchronous inference 文档](https://huggingface.co/docs/lerobot/main/en/async)
- [LeRobot SO-101 文档](https://huggingface.co/docs/lerobot/main/en/so101)
- [Feetech STS3215 官方规格书](https://www.feetechrc.com/Data/feetechrc/upload/file/20200611/6372749961523760249976542.pdf)
- 本仓库示例：`examples/tutorial/async-inf/robot_client.py`
- 本仓库实现：`src/lerobot/async_inference/policy_server.py`、`src/lerobot/async_inference/robot_client.py`
