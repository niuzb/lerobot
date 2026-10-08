# 本次修改概况

整理日期：2026-10-08。

本文汇总本次提交包含的已有代码、测试和运行文档，主要围绕 SO-101「胶带放入笔筒」任务的数据录制、远程训练和实机推理。仓库开发约定见 `AGENTS.md`，用户操作指南见 `AGENT_GUIDE.md`。

## 1. SO-101 反馈、运动保护与录制

- `src/lerobot/robots/so_follower/so_follower.py`：校验反馈是否包含全部电机及有限数值；读取失败或反馈无效时锁存故障，阻止后续动作，重新连接后才能继续。有效反馈保留实际测量值，不再用旧姿态替换较大的位置变化。
- `src/lerobot/robots/so_follower/config_so_follower.py`：增加按时间限制目标速度的配置 `max_target_velocity`、跟踪误差限制 `max_tracking_error` 和时间间隔上限 `target_ramp_max_dt`。速度与跟踪限制必须成对配置，且不能与 `max_relative_target` 同时使用；旧的位置跳变参数保留为兼容字段。
- 目标速度保护以实际反馈及上一次成功写入的目标为依据，限制通信暂停后的单次运动额度，并在反馈超出可跟踪范围时抛出异常。
- `src/lerobot/scripts/lerobot_record.py`：退出时优先断开机械臂与遥操作设备，再进行汇总、语音、数据收尾等工作；单个设备断开失败仍会继续清理其他设备。
- `compensated_teleop.py`：增加显式启用的腕部参考姿态补偿，支持只读检查、短时测试和录制；录制保存从臂坐标中的实际下发动作。
- `check_recorded_dataset.py`：增加录制数据数值审计，检查非有限值，以及动作变化但状态长时间冻结等问题。

## 2. 远程异步推理与关节单位

- `src/lerobot/async_inference/helpers.py`：增加观测图像 JPEG 编解码，保留观测时间戳、步号与强制处理标记，并兼容未压缩观测。
- `src/lerobot/async_inference/configs.py`：增加 `observation_jpeg_quality`（默认 80）和 `policy_joint_units`（默认使用机器人单位）及参数校验。
- `src/lerobot/async_inference/robot_client.py`：通过独立线程发送观测，使用容量为 1 的队列保留最新观测，增加发送与丢弃统计、压缩载荷信息；观测异常时停止客户端。
- `src/lerobot/async_inference/policy_server.py`：解码压缩观测，记录接收速率和解码耗时；重连时重置观测状态，并复用同一路径的已加载模型。
- 新增 `src/lerobot/async_inference/joint_units.py`：根据本机标定，在策略的身体关节 `[-100, 100]` 坐标与驱动的角度坐标之间显式转换；夹爪保持 `[0, 100]`，检查标定、关节集合、驱动单位及非有限输入。动作转换发生在驱动运动保护之前。

## 3. 推理服务、试跑与诊断工具

- `act_remote_rollout.py`、`smolvla_remote_rollout.py`、`pi05_remote_rollout.py`：提供远程模型的只读预检与有时长限制的执行入口，检查预测值并记录运行结果。
- `smolvla_cloud_server.py`、`pi05_cloud_server.py`、`pi05_v2_cloud_server.py` 及对应 `*_cloud_relay.py`：提供专用模型服务和连接中继；π0.5 服务保留训练所需的相机宽高比，并预加载模型。
- `pi05_remote_rollout.py`：分别记录原始动作块、融合后的动作、单位转换后的请求、实际发送值和反馈，便于定位模型输出与执行结果的差异。
- `short_rollout.py`：增加相机快照和试跑日志；正常停止后依据实际测量位置返回初始姿态，并处理复位超时、反馈异常及控制线程未退出等情况。
- `dryrun_observation.py`、`observation_stream_test.py`、`prediction_only_test.py`、`pi05_local_observation.py`、`diagnose_policy_units.py`：补充观测链路、只读预测及单位匹配诊断入口。

## 4. 训练、数据准备与模型发布脚本

- `src/lerobot/configs/policies.py`：修正 `draccus.encode` 调用方式，保存配置时显式写入策略 `type`，保证后续能够识别具体策略类型。
- SmolVLA：新增训练、恢复训练、清洗数据准备、预检、训练启动、推理启动和 Hub 上传校验脚本。
- π0.5：新增基础权重下载、数据与依赖预检、正式训练、推理启动、模型上传和上传结果核验脚本。
- π0.5 v2：新增独立数据视图及训练脚本；按完整 episode 划分 79 条训练、9 条验证，仅使用训练帧重算关节统计量，并从固定的基础权重开始训练。
- 这些脚本包含具体实验日期、路径、模型和运行环境配置，复用时需要按目标环境核对参数。

## 5. 操作文档与实验记录

- ACT：`ACT_TRAINING_COMPLETE_GUIDE.md`、`ACT_REMOTE_GPU_RUNBOOK.md`。
- 录制恢复与清洗：`RECORDING_RECOVERY_20261001.md`、`DATASET_CLEANING_20261002.md`。
- SmolVLA：`SMOLVLA_REMOTE_GPU_SO101_RUNBOOK.md`，以及 2026-10-01 至 2026-10-03 的训练、推理和失败诊断记录。
- π0.5：`PI05_REMOTE_GPU_20261004.md`、`PI05_REMOTE_ROLLOUT_20261006.md`、`PI05_V2_20261006.md`、`PI05_V2_ROLLOUT_20261007.md`。
- 文档中的训练指标、实机结果和资源地址属于各次实验记录；本次提交整理没有重新运行训练或实机操作。

## 6. 测试覆盖与本次验证

新增或扩展的测试覆盖 JPEG 观测传输、观测失败停止、关节单位转换、SO follower 反馈校验与目标速度保护、遥操作补偿、数据冻结审计、π0.5 分层动作日志和试跑复位逻辑。

本次验证结果：

- 43 个新增或修改的 Python 文件通过 AST 语法检查。
- `git diff HEAD --check` 通过。
- 已尝试执行以下 8 个测试文件；pytest 加载仓库 `conftest.py` 时，PyTorch 的 `c10.dll` 初始化失败（Windows `WinError 1114`），因此没有执行到测试用例，不能将本次回归测试记为通过。

```powershell
uv run --no-sync pytest tests/async_inference/test_helpers.py tests/async_inference/test_robot_client.py tests/async_inference/test_joint_units.py tests/robots/test_so100_follower.py tests/robots/test_compensated_teleop.py tests/robots/test_pi05_rollout_logging.py tests/robots/test_recorded_dataset_audit.py tests/robots/test_short_rollout_reset.py -q --maxfail=5
```

本次未运行完整测试集、GPU 训练或实机验证。修复本机 PyTorch DLL 环境后，可使用上述命令重新执行相关回归测试。
