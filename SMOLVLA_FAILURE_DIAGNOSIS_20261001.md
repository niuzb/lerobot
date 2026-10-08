# SmolVLA 抓取失败诊断

检查对象：60 秒试跑 `outputs/smolvla_remote_20261001_124554`；本次训练数据 `zhniu/lerobot_zihao_dataset_a_20260927_202323`，revision `01241e32b3b32f043ffb786a4d9afb264b4f981a`。

## 1. 已确认的首要问题：训练状态被冻结

12000 帧中，从全局 index 304（episode_index=1、frame_index=4、该 episode 的 0.133 秒）开始，剩余 11696 帧的六维 observation.state 完全相同，占 97.47%。Episode 2–39 全部状态无变化，但视频显示机械臂运动，action 也在变化。

冻结向量为 `[17.846153, -36.351650, 45.450550, 72.351646, 3.472528, 34.626865]`，关节顺序为 shoulder_pan、shoulder_lift、elbow_flex、wrist_flex、wrist_roll、gripper。

本地 parquet SHA-256 `f7adb1379ef1b9bfdb24cea382e2145cbaf48b7db24e801bf5d5babca11f051c` 与 Hugging Face 指定 revision 的 LFS oid 一致，确认问题存在于训练源数据。

冻结使状态与图像/动作不一致，并污染归一化。腕弯 wrist_flex 的训练标准差仅 0.6027 度，本次运行该输入相对训练统计的中位绝对 z-score 为 36.23；shoulder_lift 和 wrist_roll 在本次运行中全部超出训练 state 的最小/最大范围。低训练 loss 不能证明此模型具备有效闭环控制能力。

## 2. 冻结的高度可疑代码路径

`src/lerobot/robots/so_follower/so_follower.py` 的 `_filter_present_position_jumps`：任一关节与旧基准相差超过 max_position_jump（5 度），返回整组旧状态，且不更新基准。如果实际姿态持续偏离该基准，后续每帧都可被拒绝，形成永久冻结。`get_observation` 将该结果交给录制脚本，录制端仍保存图像和 action，因此可以静默生成不一致数据。

这是根据代码和数据模式得出的高度可疑机制，不是已经找回当时录制日志后的确认。没有证据认定某个电机损坏。下一步应让无效反馈明确中止录制/运动，避免把缓存姿态当成新测量保存，并验证异常后的恢复策略；不能简单删除保护或自动接受大跳变。

## 3. 独立的执行问题：速度限制明显改变示范时序

本次统一限制 10 度/秒，夹爪为 10 百分比点/秒。按 action 日志差分，shoulder_lift、elbow_flex、gripper 分别有约 82%、79%、77% 的控制间隔达到至少 9.5 的速度边界附近。

示范 action 相邻帧差分的速度 P95：shoulder_lift 84.4 度/秒、elbow_flex 79.1 度/秒、gripper 73.0 百分比点/秒。它们是示范目标变化速率，不等于已经测得的物理关节速度。限速使实际轨迹严重慢于模型动作时序，是另一项重要影响；不能据此直接解除限速，因为当前模型的训练状态已损坏。

## 4. 其他检查与限制

- 相机映射没有发现互换：训练 front / 当前 camera1 都为近夹爪视角，side / camera2 都为外部视角。
- 摄像头画面布局与示范存在差异；训练中部分示范片段未完成任务，也存在明显成功放置的片段，不能把全部 40 条都视为已验证的成功示范。
- 本地最新帧机制和容量 1 的发送队列不会无限积压；本次每路约 13.1 FPS，控制约 29.5 Hz，网络延迟和异步动作处理仍会影响表现，但不足以解释/修复训练 state 冻结。
- 原运行只记录了限幅后的 sent 和 measured，未保存每条原始预测动作，因此不能精确追溯每次限幅前后的差值和抓取失败瞬间的全部模型输出。
- 本次诊断未启动机械臂、未修改控制保护、未重训。

## 建议处理顺序

1. 修复状态冻结的处理及录制质量检查，做不运动的反馈校验，再做短遥操作录制验证六个状态跟随实际变化。
2. 重新录制经过核验的成功示范；不要将 action 直接复制成 observation.state 来补救，二者物理含义不同。除非另有同步原始反馈日志，否则旧数据不能可靠还原。
3. 用干净数据重训，并保存原始模型动作、实际发送动作、实测状态和帧/动作时龄。
4. 在现场监护下校准速度与任务时序，按关节及夹爪分别设置限制，逐步验证完整抓取和放置。

证据文件：`outputs/smolvla_diagnosis_20261001/frozen_state.json`、`metrics.json`、`demo_contact_sheet.jpg`、`demo_multiple.jpg`、`rollout_contact_sheet.jpg`。
