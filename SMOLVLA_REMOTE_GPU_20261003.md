# 清洗数据模型：Hub 上传与异步推理

模型：`zhniu/smolvla_tape_curated_20261002`，88 条清洗示范，20,000 steps。

## 本次推理验证

- 本地 SO-101 follower：COM5，校准 ID `zihao_follower_arm`，身体关节使用度，夹爪 0–100。
- 相机 0 → camera1（训练 front，腕部）；相机 2 → camera2（训练 side，外部）。640×480、30 FPS、JPEG quality 80。
- 远端 worker `4377679`，Tesla T4；登录 IP `fdbd:dc03:6:622::69`，SSH 11160；容器实际 IPv6 `fdbd:dc03:6:622:4500::84`。
- 模型目录：`/root/code/lerobot/outputs/train/smolvla_tape_curated_20261002/checkpoints/020000/pretrained_model`。
- 链路：本地 `127.0.0.1:18081` → Merlin 登录机 `127.0.0.1:50054` relay → GPU 容器 50053。
- worker 服务 PID `935915`，本地 SSH 隧道 PID `10708`。PID 仅对应本次启动。
- 已切换 `smolvla_cloud_server.py` 和 `smolvla_remote_rollout.py` 的默认 checkpoint；服务拒绝其他模型路径。
- 三次只读推理通过：每批 30 个六维有限动作，往返约 1063/836/675 ms；稳态服务端预处理与推理约 450 ms。
- 2026-10-03 10:30 完成 5 秒限速实机试跑：125 条动作、9 批动作、75 次观测；动作时间跨度 4.985 秒，正常退出并断开机械臂和相机。
- 目标速度限制 10 度/秒（夹爪百分点/秒），跟踪误差上限 5，2 秒没有新动作停止。未自动复位。
- 观测发送约 12–16 FPS，单包约 48–54 KiB，待发送队列只保留最新帧。125 条动作/4.985 秒约 25 Hz，未达到完整 30 Hz。
- 只读快照：`outputs/smolvla_preflight_20261003_102855`。
- 实机动作/快照/汇总：`outputs/smolvla_remote_20261003_103021`。
- 完成链路验证不等于抓取成功，尚未确认完整任务成功。

## 再次运行

### 200 秒试跑（2026-10-03 10:34–10:38）

按用户要求将客户端允许的时长上限改为 200 秒，使用 `--execute --seconds 200`；限速和跟踪误差配置保持不变。完成 200.016 秒运行，5,052 条动作（约 25.3 Hz）、361 批远端动作、3,171 次观测。进程退出码 0，机械臂与相机均正常断开。未自动复位，远端服务与隧道保留。记录目录 `outputs/smolvla_remote_20261003_103426`，日志 `outputs/smolvla_execute_200s_20261003.log`。本记录确认执行完成，不代表抓取任务成功。

服务、relay 和 SSH 隧道保留。本地客户端结束后机械臂已断开。

```powershell
cd C:\Users\Admin\code\lerobot
# 只读预测，不写电机目标
& C:\Users\Admin\.local\bin\uv.exe run --no-project --python C:\Users\Admin\miniconda3\envs\lerobot312\python.exe --no-managed-python python -u smolvla_remote_rollout.py
# 现场监护下运行，5 秒后停止
& C:\Users\Admin\.local\bin\uv.exe run --no-project --python C:\Users\Admin\miniconda3\envs\lerobot312\python.exe --no-managed-python python -u smolvla_remote_rollout.py --execute --seconds 5
```

## 上传

使用 `upload_smolvla_curated_20261003.py`，通过 worker 现有 HF 登录上传完整 pretrained_model 目录。登录机外网连接不可达，因此上传在 worker 内执行。

上传成功并校验通过：所有本地模型文件在 Hub 对应提交中均存在，大小一致，LFS 文件 SHA-256 一致。

- Hub：https://huggingface.co/zhniu/smolvla_tape_curated_20261002
- 提交：`f9c77976166d68f4416f6d2038f31574f33281ee`
