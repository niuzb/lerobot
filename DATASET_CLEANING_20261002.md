# 正式数据清洗结果

清洗版本：`zhniu/smolvla_tape_clean_20261002_142151_curated_v1`

本地目录：`C:\Users\Admin\.cache\huggingface\lerobot\zhniu\smolvla_tape_clean_20261002_142151_curated_v1`

## 结果

| 项目 | 原始 | 清洗后 |
|---|---:|---:|
| Episode | 100 | 88 |
| 帧数 | 58,849 | 27,632 |
| 累计时长（30 FPS） | 32.69 分钟 | 15.35 分钟 |
| 六维状态完全相同的相邻帧占比 | 53.50% | 13.89% |

原始数据保留在 `C:\Users\Admin\.cache\huggingface\lerobot\zhniu\smolvla_tape_clean_20261002_142151`。清洗前后逐文件 SHA-256 一致，原始文件没有修改。没有上传 Hub，也没有启动训练。

## 清洗范围

从训练副本中排除原始 episode_index：`0, 4, 30, 38, 43, 44, 47, 65, 86, 90, 93, 94`。编号从 0 开始。包含空录、开始就已完成任务、录制在搬运中截断、人工干预以及未能确认完整成功的片段；保守排除不等于认定每个片段都存在采集损坏。

其余 episode 裁去开头等待和末尾等待/人工复位，使用身体关节运动确定候选窗口，并检查每条候选窗口的起始、中间、终止图像。一般保留约 0.75 秒前置、1 秒后置余量。没有删除轨迹内部静止帧，没有改变动作速度、重采样、平滑、插值、限幅或复制 action 到 state。

- 原始 episode 27：开始的局部状态静止告警位于实际动作之前，裁剪从 frame 140 开始，保留后续抓放。
- 原始 episode 48：手动调整结束到 frame 330（包含），排除随后人手复位。
- 原始 episode 98：手动调整起点到 frame 185，排除前面人手摆放物体。
- episode 2、7、10、26、46 等的人工复位尾段由裁剪排除。

视觉审核采用阶段抽帧与疑点加密抽帧，并非逐帧人工认证，因此这是首版训练用清洗集，不是 88 条零失误示范的保证。每条原始编号、保留范围和新编号都可在新数据集的 `cleaning_manifest.json` 中追溯。

## 重建与验证

- 重建 LeRobot v3 parquet、全局索引、episode 索引、帧索引、时间戳、episode metadata。
- 重算每条和全局数值统计量；全局数值分位数直接按清洗后的数值计算。
- 图像统计按每条每相机 32 个均匀采样画面、160×120 缩略图重新计算。用于图像统计的缩略图不会替换训练视频。
- 两路视频保持 640×480、30 FPS，输出 176 个独立 episode 视频，编码为 H.264 CRF 0，保留原视频解码后的像素；这不能恢复原始 AV1 编码已经损失的信息。
- 所有 176 个视频全量解码通过，帧数和时间戳匹配；每个视频首/中/末帧与原始指定帧 RGB 像素完全一致。
- 所有保留的 state/action 数值逐帧与源数据对应区间一致。
- 数值审计通过：`passed=true`，没有非有限值或状态冻结启发式告警。
- 使用 `LeRobotDataset(..., video_backend='pyav')` 成功读取首、中、末 episode 的首/中/末帧，图像形状及 state/action 匹配。

## 后续训练使用

训练时使用新的 ID 和本地目录；当前只是本地数据集，Hub 上尚未创建同名仓库。远端训练需要先上传或完整复制新目录。

```text
--dataset.repo_id=zhniu/smolvla_tape_clean_20261002_142151_curated_v1
--dataset.root=C:\Users\Admin\.cache\huggingface\lerobot\zhniu\smolvla_tape_clean_20261002_142151_curated_v1
```

建议评估时按 episode 留出验证数据。当前默认 split 仍是全部 88 条的 train；没有自动改变训练/验证分配。

脚本：`outputs/build_clean_dataset_20261002.py`。报告：`outputs/formal_audit_20261002/cleaning_result.json`、`clean_numeric_verification.json`。源数据校验摘要位于 `source_hashes_before_cleaning.json`。
