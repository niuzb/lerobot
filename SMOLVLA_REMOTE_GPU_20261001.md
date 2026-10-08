# 本次胶带放入笔筒：SmolVLA 远程 GPU 推理

模型：`zhniu/smolvla_tape_into_pen_holder_20261001`，20000 步。

## 本次验证结果（2026-10-01 12:43）

后续 12:46–12:47 已按用户要求完成 60 秒限速运行：1772 条动作、204 批云端动作、786 次观测，motion_seconds=60.0，约 29.5 Hz；进程正常退出并断开机械臂和摄像头。记录目录为 `outputs/smolvla_remote_20261001_124554`。客户端默认时长现为 60 秒；默认仍为只读模式，运动需 `--execute`。

- 只读检查三次均返回 30 条有限值的 6 维 action；往返时间约 966、320、511 ms。
- 用户确认现场就位后完成 5 秒限速试跑，发送 148 条动作，收到 15 批云端动作，发送 68 次观测。
- 动作日志首末时间差为 5.00 秒，约 29.4 Hz；正常结束并断开 COM5 和摄像头，无自动复位。
- 原试跑 summary 的 motion_seconds=5.687 包含清理时间，实际动作跨度以 actions.jsonl 为准；客户端现已修正后续汇总计时。
- 运行目录：`outputs/smolvla_remote_20261001_124321`，只读快照：`outputs/smolvla_preflight_20261001_124201`。
- JPEG/观测相关测试 17 项通过。已验证远程闭环运行，尚未确认任务抓取和放置成功。
- 云端服务、relay 和本地隧道保留，可用于下一次手动试跑。

## 配置

- 本地：Windows，`C:\Users\Admin\miniconda3\envs\lerobot312\python.exe`。
- 机械臂：COM5，SO-101，校准 ID `zihao_follower_arm`。
- 关节单位：身体关节为度，夹爪为 0–100，与本次训练数据一致；不启用旧模型的范围转换。
- 相机 0 是本次训练 front，传输名 `camera1`；相机 2 是 side，传输名 `camera2`。不要套用旧 SmolVLA 文档相反的相机顺序。
- 摄像头：DirectShow，MJPG，640×480，30 FPS；网络传输 JPEG quality 80。
- 云端 worker：`4374960`，NVIDIA A10。`mlx worker list` 中的 `fdbd:dc03:14:428::38:10411` 是登录入口，推理 relay 使用容器实际 IPv6 `fdbd:dc03:14:428:2600::6c`，以 `/proc/net/if_inet6` 为准。
- 云端 checkpoint：`/root/code/lerobot/outputs/train/smolvla_tape_into_pen_holder_20261001/checkpoints/020000/pretrained_model`。
- 链路：本地 `127.0.0.1:18081` → SSH 主机 `127.0.0.1:50054` → GPU 容器 `50053`。
- 服务保留 640×480 图像比例，使用 SmolVLA 自身的 512×512 padding；不会先拉伸为 256×256。
- 目标控制频率 30 Hz，每批取 30 个动作；限速 10 度/秒（夹爪百分比点/秒），跟踪误差限值 5，2 秒无新动作停止。
- 首次运动 5 秒，结束断开机械臂，不自动复位。需要现场监护。

## 服务启动

先检查现有进程，避免重复启动。文件均在 `/root/code/lerobot`，已有服务无需再次执行。

GPU worker 内：

```bash
cd /root/code/lerobot
nohup /root/miniconda3/envs/lerobot/bin/python -u smolvla_cloud_server.py > outputs/smolvla_cloud_server_20261001.log 2>&1 < /dev/null &
```

SSH 主机内：

```bash
cd /root/code/lerobot
nohup /root/miniconda3/envs/lerobot/bin/python -u smolvla_cloud_relay.py fdbd:dc03:14:428:2600::6c > outputs/smolvla_cloud_relay_20261001.log 2>&1 < /dev/null &
```

本地 PowerShell 隧道：

```powershell
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L 127.0.0.1:18081:127.0.0.1:50054 37944.niuzhibin.ws@merlin-ssh-proxy-cn.byted.org -p 22222
```

云端已备份并同步 JPEG 传输需要的 `async_inference/helpers.py` 和 `policy_server.py`；备份后缀为 `.before_smolvla_remote_20261001`。

## 本地验证和运行

```powershell
cd C:\Users\Admin\code\lerobot
& C:\Users\Admin\miniconda3\envs\lerobot312\python.exe -u smolvla_remote_rollout.py
# 现场就位且只读验证通过后：
& C:\Users\Admin\miniconda3\envs\lerobot312\python.exe -u smolvla_remote_rollout.py --execute --seconds 5
```

只读验证不写电机目标，保存两张相机快照到 `outputs/smolvla_preflight_时间戳`。运动记录、相机快照和汇总保存在 `outputs/smolvla_remote_时间戳`。
