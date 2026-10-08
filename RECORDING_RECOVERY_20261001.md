# 状态冻结修复后：重新录制

旧数据 `zhniu/lerobot_zihao_dataset_a_20260927_202323` 的 97.47% 状态被冻结；保留供诊断，不继续用于正式训练，不覆盖原仓库。此次修复没有自动操作机械臂或启动录制。

## 修复行为

- 已移除相邻反馈位置的跳变阈值和时间补偿逻辑。观测使用本次真实读数，不截断、不平滑，也不返回缓存位置；数据质量在录制完成后检查。
- NaN/Inf、缺失关节或读总线失败仍报错并锁定本次连接，断开重连后才能继续。
- 发送目标前仍检查反馈完整性和有限性。动作速度、相对目标和跟踪误差限制继续由原配置控制。
- 录制异常会退出循环，当前未完成 episode 不会调用 save_episode；已保存 episode 保留。清理时优先断开机械臂/主臂，再做语音、文件收尾和上传。
- 远程客户端读观测异常会停止，不再忽略异常继续消耗动作队列。

## 先录 3 条检查，再正式录制

连接主臂 COM6、从臂 COM5，保持训练用相机 0=front、2=side。使用既有校准文件，不因重新录制而重做校准。先安排 15 秒每条、8 秒复位，避免原先 10 秒截断动作；操作仍需现场监护，先平缓遥操作确认方向与反馈正确。

PowerShell：

```powershell
cd C:\Users\Admin\code\lerobot
$Py = 'C:\Users\Admin\miniconda3\envs\lerobot312\python.exe'
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
  --dataset.repo_id=zhniu/smolvla_tape_clean_pilot `
  --dataset.fps=30 `
  --dataset.num_episodes=3 `
  --dataset.episode_time_s=15 `
  --dataset.reset_time_s=8 `
  --dataset.single_task='Put the tape into the pen holder' `
  --dataset.push_to_hub=false `
  --play_sounds=false `
  --display_data=true
```

录制日志会显示追加时间戳后的真实数据集名。完整结束并等待文件收尾后，将下方占位路径换成日志里的真实目录：

```powershell
& $Py check_recorded_dataset.py 'C:\Users\Admin\.cache\huggingface\lerobot\zhniu\smolvla_tape_clean_pilot_实际时间戳'
```

脚本返回 `passed: true`、退出码 0 才通过数值体检。它检查非有限值，以及连续至少 2 秒六维状态完全不变但 action 变化超过 5 单位的片段。允许正常静止等待；失败表示必须检查，不能单凭这一启发式断言电机或传感器坏了。体检不代替视频审核：需逐条确认确实夹住胶带、送到笔筒、释放成功，图像流与关节运动一致。

若报 `Feedback invalid` / `Present_Position read failed`：停止本轮，检查姿态、线缆、标定和关节跨界，不要放大阈值或关闭保护来强行继续。重连后另起测试，失败的未保存片段应重录。

旧命令中的 `max_position_jump`、`max_gripper_position_jump`、`feedback_reference_dt` 和 `feedback_max_dt` 仅保留解析兼容，已不参与运行判断。请保留 `--play_sounds=false`，避免语音调用超时。

3 条通过后，用新基础名称 `zhniu/smolvla_tape_clean` 录正式数据，录制中只保留完整成功的示范，完成后同样做体检、视频检查，再上传和训练。旧数据无法仅靠复制 action 列恢复真实 state。
