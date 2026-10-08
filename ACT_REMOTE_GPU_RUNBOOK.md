# ACT 云端推理（2026-09-28）

本地入口 `act_remote_rollout.py` 参考 `short_rollout.py`，保留原来的 SmolVLA 文件。

- 模型：`zhniu/act_tape_into_pen_holder`，15,000 步 checkpoint。
- 服务端 checkpoint：`/root/code/lerobot/outputs/train/act_tape_into_pen_holder/checkpoints/015000/pretrained_model`。
- 本地：Conda `lerobot312`、COM5、`zihao_follower_arm`。
- 图像：front=0、side=2，DirectShow、MJPG、640×480、30 FPS；RGB 以 JPEG quality=80 传输。
- 关节单位：身体关节为角度，夹爪为 0–100；`policy_joint_units=robot`，不使用旧 SmolVLA 的范围换算。
- 本地动作循环目标 30 Hz，云端每批返回 30 个动作，队列重叠使用 weighted_average。
- 目标速度限制 10 度/秒（夹爪为百分比点/秒），跟踪误差限制 5；2 秒未收到新动作即停止。
- 试跑结束断开机械臂，不自动复位。动作和快照保存在 `outputs/act_remote_时间戳`。

## 当前进度

已提交 T4 worker `4367459`（1 GPU、8 CPU、32 GiB），尚在等待资源。当前 `mlx worker quota` 仅显示 CPU-only，服务及实机闭环尚未验证。

## 云端启动

登录：

```powershell
ssh 37944.niuzhibin.ws@merlin-ssh-proxy-cn.byted.org -p 22222
```

先执行 `mlx worker list`，确认 worker 获得 podIP，再执行：

```bash
mlx worker login 4367459 -- bash
cd /root/code/lerobot
nvidia-smi
nohup /root/miniconda3/envs/lerobot/bin/python -u outputs/act_cloud_server_20260928.py > outputs/act_cloud_server_20260928.log 2>&1 < /dev/null &
```

服务启动文件已上传，使用现有 JPEG-capable PolicyServer，监听 worker 的 50051 端口。

在 SSH 主机而非 worker 上启动 relay；把 `<worker-podIP>` 替换为这次 worker 的实际地址：

```bash
cd /root/code/lerobot
nohup /root/miniconda3/envs/lerobot/bin/python -u outputs/act_cloud_relay_20260928.py '<worker-podIP>' > outputs/act_cloud_relay_20260928.log 2>&1 < /dev/null &
```

启动前检查端口和进程，避免重复运行或覆盖已有服务日志。Relay 只监听 SSH 主机的 `127.0.0.1:50052`。

本地 PowerShell 保持以下隧道运行：

```powershell
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L 127.0.0.1:18080:127.0.0.1:50052 37944.niuzhibin.ws@merlin-ssh-proxy-cn.byted.org -p 22222
```

## 验证与运行

只读验证（读取关节和相机，检查三次云端返回的动作，不发送电机运动指令）：

```powershell
cd C:\Users\Admin\code\lerobot
& C:\Users\Admin\miniconda3\envs\lerobot312\python.exe -u act_remote_rollout.py
```

确认只读验证通过后，人工现场监护下运行 10 秒（从第一条执行动作计时）：

```powershell
& C:\Users\Admin\miniconda3\envs\lerobot312\python.exe -u act_remote_rollout.py --execute --seconds 10
```

相机机位和物品布置应与录制数据一致。异常时可 Ctrl+C 终止，机械臂电源切断是独立急停手段。申请等待结束后不要在无人现场监护时自动启动运动。
