# Pi 0.5 远端推理与首次实机试跑（2026-10-06）

## 200 秒试跑（用户追加授权）

北京时间 10:42:51–10:46:12 完成 200 秒控制，5,590 条动作（约 27.95 Hz）、422 批预测、3,251 次观测发送。
退出码 0，机械臂与相机正常断开，不自动复位；日志无失联超时或执行异常。
用户要求将 `--seconds` 上限提高到 200，默认仍为 5 秒；速度 5 度/秒、跟踪误差 3 及失联保护保持不变。

已检查过程与结束前画面：夹爪接近/接触胶带，但最终胶带仍在桌面上，未完成放入笔筒。
阶段抽帧不能排除中间短暂提起过胶带，因此不能据此断言从未抓起。
动作、每两秒相机快照、汇总：`outputs/pi05_remote_20261006_104241`。
执行日志：`outputs/pi05_execute_200s_20261006.log`。

## 实际结果

用户确认现场在电源开关旁、工作区清空，并授权 5 秒试跑后执行。
只读本地检查和三次远端预测先行通过，未写电机目标。

- 最终模型：`zhniu/pi05_tape_curated_20261004`，远端使用已验证的 `checkpoints/080000/pretrained_model`。
- 只读预测：每批 30 个有限六维动作，往返 1789.96 / 541.50 / 513.67 ms。
- 试跑：北京时间 10:38:48–10:38:55，动作时间跨度 5.031 秒，146 条动作、11 批预测、80 次观测发送。
- 退出码 0，机械臂与两路相机正常断开，不自动复位。
- 日志无失联超时或执行异常。链路试跑完成不代表胶带抓放任务成功。
- 只读快照：`outputs/pi05_local_readonly_20261006_103523`。
- 远端预测快照：`outputs/pi05_preflight_20261006_103752`。
- 实机动作、快照、汇总：`outputs/pi05_remote_20261006_103836`。
- 执行日志：`outputs/pi05_execute_5s_20261006.log`。

## 配置

- 本地 SO-101：COM5，校准 `zihao_follower_arm`，use_degrees=true，夹爪 0–100。
- front = 相机 0（腕部），side = 相机 2（外部）；640×480、30 FPS、MJPG、DirectShow。
- 使用采集时的驱动单位，policy_joint_units=robot；QUANTILES 反归一化由保存的处理器完成。
- 目标速度 5 度/秒（夹爪百分点/秒），最大跟踪误差 3。当前驱动的 max_position_jump 已弃用并忽略，不能把配置值 5 当作生效的跳变保护；此前记录对此有误，已更正。
- 2 秒未收到新动作即停止，动作批大小 30，队列重叠使用 weighted_average。
- 任务文本：`Put the tape into the pen holder`。
- 新脚本 `pi05_remote_rollout.py` 默认只读预测，带 `--execute` 才运动；默认时长 5 秒，用户追加授权后时长上限 200 秒。

## 远端服务与连接

- Worker `4380064`，A10；Merlin 入口 `37944.niuzhibin.ws@merlin-ssh-proxy-cn.byted.org:22222`。
- worker SSH `root@fdbd:dc03:14:344::15:10471`；容器 IPv6 `fdbd:dc03:14:344:f00::ea`。
- 本地 127.0.0.1:18082 → Merlin 127.0.0.1:50056 → 容器 50055。
- `pi05_cloud_server.py` 保留原图比例，由 Pi05 自身 resize/pad；排除配置中的 empty camera。
- 服务仅接受 pi05 和指定的最终模型目录，加载后 eval/reset，推理使用 inference_mode。
- 启动入口 `start_pi05_inference_20261006.sh`，日志远端 `pi05_inference_20261006.log`。
- Relay `pi05_cloud_relay.py`，日志远端 `pi05_relay_20261006.log`。
- 试跑结束保留远端模型服务和隧道。机械臂已断开。

只读再次检查：

```powershell
& C:\Users\Admin\.local\bin\uv.exe run --no-project --python C:\Users\Admin\miniconda3\envs\lerobot312\python.exe --no-managed-python python -u pi05_remote_rollout.py
```

现场观察、电源开关可立即操作、工作区无人手且相机/物体布置确认后，执行新的 5 秒试跑：

```powershell
& C:\Users\Admin\.local\bin\uv.exe run --no-project --python C:\Users\Admin\miniconda3\envs\lerobot312\python.exe --no-managed-python python -u pi05_remote_rollout.py --execute --seconds 5
```

## 200 秒日志复核

分析文件：`outputs/pi05_rollout_analysis_20261006.json`，脚本 `outputs/analyze_pi05_rollout_20261006.py`。
客户端与该次服务端日志没有 ERROR、Traceback、过载或失联记录；全部已写目标和反馈有限。
422 次服务端推理，中位数 363.6 ms、P95 371.3 ms、最大 530.5 ms；网络发送中位数 55.4 ms、P95 249.3 ms、最大 267.9 ms。

动作节奏存在抖动：实际约 27.95 Hz，间隔中位数 31 ms、P95 47 ms、P99 78 ms；45 次间隔超过 100 ms，最大 516 ms。
六次重复动作 timestep，但没有倒退；3 对动作日志时间戳相同，写后时间戳不足以精确重建驱动计算限速的时间。
单相邻样本计算的目标速度可出现较高尖峰，0.5 秒窗口最高约 5.16 度/秒；不能据此将采样尖峰直接判定为电机超速。

驱动限幅明显影响轨迹：各关节约 73%–90% 的正时间间隔样本目标推进接近 5 单位/秒。
肩抬关节约 19.84% 的样本跟踪差接近 3 度限制（最大 3）；肘关节约 2.77%；其余关节没有触及该限制。
夹爪实测开度 6.72–39.85，20–110 秒主要在约 14–26；训练 action 的 q99 约 96.39，但这不是每次抓取必须到达的开度。
现有动作日志只保存限幅后的写出目标与写入前反馈，没有原始预测/队列融合目标，不能区分是模型预测、队列融合还是速度限制导致开度不足。
当前可支持的结论是控制链路正常，存在执行时序/轨迹跟随问题；仅凭日志不能确定抓取失败根因。
下一次诊断应先补录模型原始动作、队列融合动作、限幅前目标、限幅后目标及同步画面，不直接放宽保护或继续长时运行。
