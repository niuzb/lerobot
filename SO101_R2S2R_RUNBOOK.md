# SO-101 Real-to-Sim-to-Real：胶带放入笔筒

入口：`so101_r2s2r.py`，也可运行 `python -m lerobot.sim2real.cli`。

严格标定流程保留原来的验收门槛。2026-10-18 用户明确选择跳过补标定并开始生成、训练，已新增独立的近似场景实验流程，保留 `calibration_passed=false`；物理成功和数据完整性检查仍执行。当前运行、GPU 和日志见 [SMOLVLA_R2S2R_20261018.md](SMOLVLA_R2S2R_20261018.md)。原始数据只读，每次使用新的输出目录；尚不能宣称抓取模型已经改善。

## 1. 当前数据和实测参数

Merlin 上已核实存在：

```text
/root/code/lerobot/datasets/smolvla_tape_clean_20261002_142151_curated_v1
```

88 条、27,632 帧、30 FPS；0–78 为训练，79–87 为验证。本地旧缓存目录目前不存在。`audit` 读取原数据并计算 meta/data/videos 的 SHA-256，不上传 Hub、不改数据。

初始配置：`configs/so101_r2s2r.json`；当前探索仿真的实测物体配置：`configs/so101_r2s2r_20261018_mesh_holder.json`，单位为米、千克。启动脚本会读取后者，再组合已保存的姿态与夹爪候选。

| 项目            |          数值 | 来源                                       |
| --------------- | ------------: | ------------------------------------------ |
| 胶带外径 / 内径 |    46 / 36 mm | 用户实测                                   |
| 胶带高度 / 质量 |   19 mm / 3 g | 用户实测                                   |
| 笔筒内径 / 高度 |    77 / 95 mm | 用户实测                                   |
| 笔筒底厚 / 质量 | 4 mm / 约 5 g | 用户实测 / 估计                            |
| 笔筒壁厚 / 结构 |   1 mm / 网状 | 用户 2026-10-18 补充；菱形外观参考既有实拍 |

物体在机器人基座坐标中的位置与接触参数仍使用探索假设；已采集的关节、夹爪间距和相机数据用于候选映射。用户已选择跳过进一步补测，按现有数据模拟，不能因此把历史验收结果改为通过。支撑平面边长 1 m、厚 20 mm 只是计算场景范围，不表示真实桌面尺寸。世界 z=0 为桌面。

笔筒是能自由移动和倾倒的刚体，不固定在桌上。胶带由凸楔段组成，中孔开放；当前笔筒为带真实网孔的菱形网壁、上下环形边框和实心底。网壁参与碰撞，未叠加隐藏实心圆柱壁，没有吸附、固定连接或脚本搬动物体。

## 2. 环境和数据审计

本地继续使用 `lerobot312`。MuJoCo 的正式运行使用已验证的远端 Linux / MuJoCo 3.3.7 / OSMesa；Windows DLL 问题不会阻止只读标定拍照。

新环境安装所需功能：

```powershell
uv pip install --python C:\Users\Admin\miniconda3\envs\lerobot312\python.exe -e ".[sim2real]"
```

硬件拍照另需既有 hardware 依赖；训练另需 training、smolvla 依赖。此命令不是要求覆盖当前已跑通环境。

远端执行：

```bash
cd /root/code/lerobot
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
PY=/root/miniconda3/envs/lerobot/bin/python
ASSET=/root/code/lerobot/outputs/so101_sim_assets_20261012/robotstudio_so101/so101.xml
$PY so101_r2s2r.py audit --config configs/so101_r2s2r.json \
  --asset "$ASSET" --output outputs/r2s2r_audit_NEW
```

`measurements_ready=false` 表示还不能建立经过标定的场景；不是原训练数据损坏。历史运行 `outputs/r2s2r_source_audit_20261016` 已确认源数据的数量和划分。

## 3. 打印棋盘格和只读采集

打印 `assets/calibration/so101_chessboard_A4.svg`，A4、实际大小、100%，关闭适应页面。模板标称参考线 90 mm、每格 18 mm，共 10×7 格、9×6 个内角点。计算必须使用纸上实测尺寸，不能仅凭打印设置采用标称值。当前纸张已实测横向 8 格=136 mm、纵向 5 格=85 mm，因此下面命令采用 `--square-mm 17`；换纸后需要重新测量。贴在平整硬板上或直接平铺固定，不弯曲，避免反光。

保持原有相机安装不变，分别让标定板覆盖两路视野的不同位置、距离和倾斜角。先采每相机至少 10 张清晰、完整的内参图。每次摆好后静止，再运行一次命令；不要让程序在相机预热时跟随移动的机械臂。

```powershell
cd C:\Users\Admin\code\lerobot
$Py = 'C:\Users\Admin\miniconda3\envs\lerobot312\python.exe'
& C:\Users\Admin\.local\bin\uv.exe run --no-project --python $Py --no-managed-python `
  python so101_r2s2r.py capture --split intrinsic --square-mm 17 --output outputs/calibration/view_001
```

每次改输出目录。`capture` 仅打开总线读取 COM5，并拍相机 0=front、2=side；不调用 `robot.connect()`、不配置电机、不发送动作，也不改变扭矩。每张照片附近读取关节，检查期间变化；输出原图、标有 0/8/45/53 角点的复核图和 `capture.json`。无棋盘格时也保存照片，但该图片不用于拟合。

带 `--board-transform` 的采集和 `assemble` 必须显式提供实测 `--square-mm`；程序记录本次方格尺寸，并拒绝将已记录为另一尺寸的采集混入汇总。只有照片的旧采集可以指定实测尺寸，但旧的米制世界点若缺少尺寸来源，需要重新生成，不能直接改标签复用。

外参需要**标定板在基座坐标系里的已知位置与方向**。把第 0 个检测内角点作为板坐标原点，第 8 个方向为 +x、第 45 个方向为 +y，检查角点编号图，避免将板旋转 180° 后仍使用原坐标。测量并保存 `T_world_board` 4×4 JSON 矩阵，然后增加：

```powershell
& $Py so101_r2s2r.py capture --split fit --square-mm 17 --board-transform outputs/board_pose_001.json `
  --output outputs/calibration/fit_001
```

每相机至少 3 个 fit 视图和另外 3 个 check 视图；check 视图不能用于拟合。重新安装相机必须生成新的场景配置。新拍摄的外参不能直接冒充 10 月 2 日录像时的外参；回放旧录像时需对齐其原安装，或单独维护旧录制场景配置。

## 4. 关节参考和相机拟合

`joint_map` 明确记录五个身体关节的 +/- 方向和弧度偏移，夹爪用 `[开度百分比, 模拟关节弧度]` 的单调折线，必须覆盖 0 和 100。不能把电机原始 homing_offset 直接当成 CAD 关节偏移。

用已知几何参考姿态将真实角度和模拟资产的关节角对应起来。至少 3 个 fit 参考姿态，且每个身体关节覆盖至少 10°；另留至少 3 个 check 姿态。check 记录还须测量对应夹爪接触点间距 `gripper_gap_m`。这是被动的几何参考测量，不会重新标定或改写电机 EEPROM。

观察格式见 `configs/so101_calibration_observations.example.json`。若已完成 capture，可将关节参考列表单独写入 `joint_references.json`，然后汇总：

```powershell
& $Py so101_r2s2r.py assemble --square-mm 17 --captures outputs/calibration `
  --joint-references outputs/joint_references.json --output outputs/calibration_assembled
```

将图片和汇总文件移到远端时，要同步修改汇总文件内的绝对图片路径，或在原机执行拟合。`intrinsic_images` 可留空以复用已经标定好的 K 和 distortion；不能在 K 未知时跳过内参。

```bash
$PY so101_r2s2r.py calibrate --config configs/so101_r2s2r.json --asset "$ASSET" \
  --observations outputs/calibration_assembled/observations.json --output outputs/r2s2r_calibrated_NEW
```

补齐物体尺寸、位置和映射后才执行。输出 `scene.calibrated.json` 和 `calibration_report.json`。验收需要独立检查视图 median≤8 px、p95≤20 px，参考姿态误差≤2°、夹爪接触点间距误差≤3 mm。失败退出码 2，不冒充通过。场景及资产内容改变会使旧验收报告失效。

## 5. 两种回放与物理验收

配置 `episode_setups`，每个源 episode 给出当时胶带中心和笔筒底部的位置，例如键 `"1"` 下的 `tape_position_m`、`holder_position_m`；不能假定所有示范初始摆放相同。`generation.template_episode` 只能选 0–78；`grasp_frame` 标注实际夹紧成功的源帧。

```bash
CFG=outputs/r2s2r_calibrated_NEW/scene.calibrated.json
$PY so101_r2s2r.py replay --config "$CFG" --episodes 0,1,2 \
  --replay-mode state --output outputs/r2s2r_state_NEW
$PY so101_r2s2r.py replay --config "$CFG" --episodes 0,1,2 \
  --replay-mode action --output outputs/r2s2r_action_NEW
```

state 模式只设置真实姿态用于视觉对齐，永远不作为物理通过证据。action 模式才发送原目标并推进动力学，比较下一时刻真实 state。源 action 是处理后的主臂目标，不能无证据宣称等于驱动实际下发值；回放不匹配时先排查单位、方向、延迟、限幅和接触参数。

动态验收至少覆盖 3 条训练轨迹，身体 RMSE≤5°、夹爪 RMSE≤10 百分点，且每条实际抓放完成。报告包含逐帧状态、原目标、物体位置、接触和视频。物体必须经历双夹爪接触、抬高至少 3 cm、水平搬运至少 3 cm、释放，并完全位于竖直笔筒内部、有物理支撑、低速度稳定 30 个控制周期，才判成功。

## 6. 20 条试验与 200 条正式仿真示范

```bash
CAL=outputs/r2s2r_calibrated_NEW/calibration_report.json
REPLAY=outputs/r2s2r_action_NEW/replay_report.json
$PY so101_r2s2r.py generate --config "$CFG" --calibration "$CAL" --replay-report "$REPLAY" \
  --count 20 --seed 1000 --output outputs/r2s2r_pilot_NEW
$PY so101_r2s2r.py generate --config "$CFG" --calibration "$CAL" --replay-report "$REPLAY" \
  --count 200 --seed 10000 --pilot outputs/r2s2r_pilot_NEW/generation_manifest.json \
  --output outputs/r2s2r_production_NEW
```

20/200 指通过物理判定的成功数，最多尝试目标数×10 次。失败 JSON 和视频也保存，但不导出为成功标签。IK 在 scratch state 中解算，每次根据随机化后的物体位置重新规划，不通过设置物体位置完成抓取。

默认胶带 x/y 各±2 cm、yaw±0.17 rad、小幅光照和摩擦变化。位置范围须经过可达性和真实失败分布审核；IK 失败保留失败证据。生产与试验种子不重叠。输出 `dataset/`（LeRobot）、`generation_manifest.json`、每次尝试视频与轨迹。数据严格使用动作前的 state/图像与接下来实际应用的目标配对。

## 7. 真实基线和混合训练

从同一固定基础模型快照独立启动两次训练；不继承旧任务模型。既有缓存快照：

```bash
BASE=/root/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/d9f33c94a60fb382c90dea2164c96845bd955e28
$PY so101_r2s2r.py train --config "$CFG" --base "$BASE" --variant baseline \
  --device cuda --output outputs/train/r2s2r_baseline_NEW
$PY so101_r2s2r.py train --config "$CFG" --base "$BASE" --variant mixed \
  --generated outputs/r2s2r_production_NEW --device cuda --output outputs/train/r2s2r_mixed_NEW
```

先查询实际 GPU worker，不能直接在没有 CUDA 的登录机运行正式命令。两者都是 20,000 次更新、batch 8、seed 1000；混合版每批恰好 4 real+4 sim。两者都只用 79 条真实训练示范重算归一化统计，以隔离扩增数据效果。每 5,000 步遍历全部 9 条验证示范、保存原生 checkpoint/处理器/优化器/RNG。`--smoke` 独立执行两步和首批验证，产物仅作软件验证，不能用于正式对照。

该独立训练入口暂不支持断点续训；不要用原生 `lerobot-train --resume` 续混合实验，避免丢失域采样规则。重新运行需使用新目录。正式训练和实机评测尚未进行，不能将测试产生的 fixture 或两步 checkpoint 当成候选模型。

## 8. 固定 30 组真机对照

```bash
$PY so101_r2s2r.py trials --config "$CFG" \
  --baseline /absolute/baseline/checkpoints/020000/pretrained_model \
  --mixed /absolute/mixed/checkpoints/020000/pretrained_model --output outputs/r2s2r_trials_NEW
```

该命令只生成 30 组相同摆放、交替先后顺序的计划和空结果表，不会驱动机械臂。每轮人工按计划摆放，分别通过已有 `smolvla_remote_rollout.py --checkpoint ... --execute --seconds ...` 执行，关联真实动作日志和视频后填写结果表。启动服务端时用新增的 `--checkpoint` 选择同一个候选路径，专用服务仍只接受该路径；不能只改客户端路径后假定服务已换模型。

```bash
$PY smolvla_cloud_server.py --checkpoint /absolute/candidate/checkpoints/020000/pretrained_model --port 50053
```

真机前可在独立种子上进行模型闭环仿真检查；同样要求标定和物理回放通过，结果只表示仿真表现：

```bash
$PY so101_r2s2r.py evaluate --config "$CFG" --calibration "$CAL" --replay-report "$REPLAY" \
  --checkpoint /absolute/candidate/checkpoints/020000/pretrained_model --device cuda \
  --seed 100000 --eval-episodes 30 --output outputs/r2s2r_policy_sim_NEW
```

```bash
$PY so101_r2s2r.py evaluate --config "$CFG" --plan outputs/r2s2r_trials_NEW/trial_plan.json \
  --results outputs/r2s2r_trials_NEW/results.completed.json --output outputs/r2s2r_comparison_NEW
```

拒绝缺项、重复项、非布尔人工判定、计划指纹不匹配。完整 60 条结果才比较抓取数、入筒数、耗时和故障数。初步接受条件为混合版至少多成功 6 次，且故障数不增加；这不是统计显著性保证。

## 9. 软件验证

```powershell
& $Py -m unittest discover -s tests/sim2real -p test_core.py -v
```

远端完整测试：

```bash
R2S2R_ASSET="$ASSET" MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
  $PY -m unittest discover -s tests/sim2real -v
```

覆盖单位转换、失败入筒判定、精确域采样、独立标定检查、相机投影与实际渲染、真实中孔碰撞、IK 不修改物体状态、LeRobot 视频和动作对齐。测试的合成数据只保存在临时目录，不能进入正式训练。

## 10. 2026-10-16 实施记录

- 本地 Ruff、锁文件一致性检查通过；远端完整 20 项测试通过。
- 源数据审计：88 条、27,632 帧，181 个源文件留存 SHA-256；首次审计已有的 info 和 data parquet 哈希未变。审计结果在远端 `outputs/r2s2r_final_audit_20261016/report.json`，本地副本在 `outputs/r2s2r_verification_20261016/source_audit.json`。
- 真实训练数据的 CPU 两步预检完成，远端目录 `outputs/r2s2r_baseline_smoke_20261016`。原生 checkpoint 含模型权重、配置、前后处理器及统计量、优化器和 RNG。重新加载后已输出有限的 `[1,6]` 动作，整个验证未发送机械臂动作。
- 该预检只验证训练和推理接口；首批验证 loss 1.0191 不代表抓取效果。混合训练、真实标定、成功仿真示范生成和真机对照都还没有执行。
- 当前没有已分配的 mlx GPU worker。正式训练需在标定和回放验收后再申请实际资源。
- 仍待补齐：笔筒壁厚；物体初始坐标；关节参考姿态和夹爪开度；双相机标定。棋盘格模板已生成，按第 3 节打印和采集。

## 11. 2026-10-17 相机内参检查

已保存当前相机的内参配置 `configs/so101_r2s2r_20261017_intrinsics.json`。分辨率 640×480，用户确认重打后的参考线为 90 mm，方格边长按 18 mm 使用。

| 相机       | 已采集 | 拟合 / 留出检查 / 排除 | 留出误差中位数 | 留出误差 95 分位 |
| ---------- | -----: | ---------------------: | -------------: | ---------------: |
| front 腕部 |     14 |             10 / 4 / 0 |       0.174 px |         0.380 px |
| side 外部  |     24 |             17 / 6 / 1 |       0.168 px |         0.408 px |

38 张原图及拍摄元数据均保留，SHA-256 核验未变。外部相机的 `live_1792211713981563100_side` 纸面可见弯曲，不满足平面标定板假设，已从拟合集排除，原图未删除；留出检查集保持不变。

新增视角后，外部相机在逐张移除拟合图的稳定性检查中，焦距变化范围低于 0.7%；腕部约 6.2%。两路均通过当前的初步内参检查。完整结果为 `outputs/calibration/20261017/intrinsics_review_003/intrinsics_report.json`，摘要为同目录 `review_summary.json`。

这些误差是镜头模型的像素重投影误差，不表示机械臂定位或抓取精度。当前只写入 K 和 distortion，`T_parent_cv`、关节映射和未测物体参数仍留空，完整场景验收没有通过。相机固定在后续使用位置后，继续采集相对于机器人基座的已知位置参考；旧数据录制时的相机外参仍需另外核对。

### 固定安装后的双相机参考

用户已确认两路相机固定在后续使用位置；确认记录为 `outputs/calibration/20261017/camera_mount_confirmation.json`。调整棋盘摆放后，`extrinsic_reference_004` 两路均检测到 54 个角点，保存原始图、角点图和关节反馈。两路顺序拍摄相隔约 4.92 秒，六维反馈一致，全程未发送运动命令。

同目录 `board_relative_pose.json` 保存相机相对于棋盘的初步位姿，腕部与外部的当前图像拟合 RMS 分别为 0.311 / 0.219 px。这是单组图像的拟合残差，不是独立外参验收，也不是相机相对于机器人基座或腕部安装体的最终标定。相机间变换仅对应本次机械臂姿态，并依赖拍摄期间棋盘静止。

`measurement_points.png` 标出棋盘角点 A=0、B=8，AB 设计值为 144 mm；P/Q 为外部图像左/右白色底座脚最前端贴桌面短边的中点。用户已确认棋盘直接铺在桌面，实测 AP=197、BP=140、AQ=234、BQ=145、PQ=58 mm。对应 JSON 中的 CAD 脚尖坐标只是待实物核对的候选，不能将图上示意像素当成实测坐标。

按 AB=144 mm 和前四个距离推算，PQ 为 50.47 mm，与直接测得的 58 mm 不一致；五个距离等权拟合的最大调整量约为 1.59 mm，不能据此断定哪一项单独错误。已请求实测 AB 以核对交点选择，并确认测量期间是否移动纸张、底座或相机。复拍 `extrinsic_reference_005` 两路仍识别到全部角点，但相对 004 的角点位移中位数为腕部 7.99 / 外部 4.65 px，原因尚未确定。这次复拍不是独立外参验收；`T_world_board` 和正式相机外参继续留空。原始测量和诊断分别保存在 004 目录的 `measurement_points.json` 与 `distance_consistency_diagnostic.json`。

随后用户实测 AB=136 mm。如果取点确为 8 格，则横向方格为 17 mm，而此前按 18 mm 计算；换用 AB=136 mm 后，前四距离推算的 PQ 为 53.75 mm，仍与 58 mm 有偏差。当前棋盘尺度已标记为待核对，保留先前 90 mm 参考线确认和所有原始报告，不覆盖历史证据。已请求重测当前纸上标注 90 mm 的参考线，以及 A 到角点 45（图中 C，跨 5 格）的长度，以区分尺寸缩放与取点差异。此前 `board_relative_pose.json` 中按 18 mm 得到的米制平移暂不能用于正式场景；说明见 `metric_scale_notice.json`。均匀缩放本身不要求重算像素内参，但横纵比例与当前纸张身份尚未确认，不能提前作此假设。

### 当前纸张按实测 17 mm 重算

用户进一步确认参考线为 86 mm、AC 为 85 mm。AB/8 与 AC/5 均为 17 mm，当前方格尺寸采用 17 mm；参考线按该比例预期 85 mm，与读数差 1 mm。原先的 90 mm 口头确认保留在 `board_measurement.json` 的历史字段。无需因均匀缩放重拍内参照片。

新的可加载配置为 `configs/so101_r2s2r_20261017_measured_board.json`：保留原 K/畸变，保存实测棋盘尺寸，正式外参仍为空。`metric_review_001/metric_report.json` 已按 17 mm 重算参考 003/004/005；004 中相机到棋盘中心距离为腕部约 0.345 m、外部约 0.405 m，较原 18 mm 假设缩小 17/18，像素拟合 RMS 仍为 0.311 / 0.219 px。原始图和旧报告未修改。

基于 CAD 两脚间距 55.56 mm 的粗定位候选，对 AP/BP/AQ/BQ 的拟合残差均小于 0.4 mm；实测 PQ=58 mm 与 CAD 仍差 2.44 mm。候选在底座左右示意点的投影偏差约 2.5 / 13.6 px；这些点是人工示意位置，并非独立验收真值。相机稳定性、照片与测量的对应关系、CAD 对应点及腕部安装变换尚待验证，因此候选矩阵仅存在诊断报告中，没有写入正式配置。

采集入口增加实测 `--square-mm`，移除固定 18 mm 的世界点换算。带世界变换的采集及汇总不提供尺寸时直接拒绝，汇总拒绝已记录的不同尺寸及来源不明的米制点。5 项无硬件回归测试通过，覆盖 136×85 mm 角点跨度、比例与投影关系、非法值、访问硬件前拦截缺失尺寸，以及不同尺寸混入检查；修改文件 Ruff 通过。内参分析脚本只处理 `split=intrinsic`，避免将后续外参图混入旧的拟合/留出划分。

用户确认拍照到量尺期间棋盘、底座及相机均未挪动。随后 `stability_001` 在同一次相机连接中连续采 5 组：腕部 4/5、外部 5/5 检出完整角点；有效帧相对首个有效帧的角点位移中位数均约 0.03 px，最大分别 0.41 / 0.23 px，全部关节反馈跨度为零。该短时静态检查通过，不构成外参验收，也不能解释先前跨次拍摄的变化。驱动对焦距、自动对焦、变焦等查询均返回 -1，不能据此判断这些设置是否启用。全程只读机械臂，没有发送动作或改变扭矩。当前串口枚举仅发现从臂 COM5 及两个蓝牙虚拟口，待确认是否可连接主臂后安排不同参考姿态采集。

### 双臂直接跟随与参考姿态采集

主臂现已连接 COM6，从臂为 COM5。用户手动调整两臂姿态后，只读检查记录在 `outputs/calibration/20261017/direct_follow_preflight_001.json`：两臂电机标定与各自本地文件一致，全部电机扭矩关闭；五个身体关节角度差最大约 9.80°，腕旋转差约 0.62°，夹爪开度差约 25.22 个百分点。本次检查没有发送目标、启用扭矩或改写标定。

按用户选择，新增 `so101_pose_capture.py` 和本地启动器。在仓库 PowerShell 中运行：

```powershell
.\start_so101_pose_capture.cmd
```

启动器通过 `uv run --no-project` 使用现有 `lerobot312` 环境，服务地址为 `http://127.0.0.1:8766/`。每次建立独立的 `outputs/calibration/YYYYMMDD/pose_capture_HHMMSS` 目录，棋盘实测方格使用 17 mm，不覆盖已有采集。启动初始只读；点击一次“开始跟随”后，身体关节直接使用主臂角度、夹爪直接使用主臂 0–100 开度，不加姿态偏移、不使用旧腕部补偿。目标受从臂已有标定范围限制，驱动目标变化率为 3°/秒（夹爪每秒 3 个百分点），30 Hz 控制。

点击“停止跟随”后从臂保持当前位置，停稳约 1 秒且两路均完整检测棋盘才允许保存。建议先采 8 个拟合姿态、3 个独立检查姿态；改变肩、肘、腕部角度，保持棋盘、基座和相机安装固定。重复姿态和外部棋盘明显移位会被拒绝。保存原始 PNG、检测角点、图像对应的最近关节反馈及时间差；世界坐标和独立关节真值仍为空，照片数量不代表外参或关节拟合通过。

页面关闭或失去心跳后停止跟随，不自动恢复。控制循环停顿超过 0.2 秒也停止跟随；反馈故障锁存，禁止新目标，仍允许操作者明确释放电机。停止跟随不等于断电，结束时先托住从臂，再点击“释放电机”和“关闭采集服务”。服务不会自动重新标定，也不会修改电机校准 EEPROM。

8 项无硬件测试通过，包含直接角度与夹爪跟随、超时停止、故障后的显式释放、先写当前目标再启用扭矩、启动默认只读、HTTP 来源检查、采集尺度与真值留空。Python Ruff 和页面 JavaScript 语法检查通过。网页接口测试使用模拟设备；真实双相机预览及实际运动尚未运行。自动审批拦截了后台启动服务的命令，仅返回 `blocked by policy`，因此需要用户运行上述本地启动器。

### 保存按钮和浏览器断连修复

用户随后运行了直接跟随采集。浏览器控制台确认：旧页面保存按钮的 `onclick="command(...)"` 被按钮自身的 `command` 属性遮蔽，抛出 `TypeError: command is not a function`。保存与关闭按钮改为显式调用 `window.runPoseCommand(...)`，释放逻辑也使用新函数名。页面 HTML 改为每次 GET 时读取，后续仅修改页面时刷新即可，不必为了前端改动重连机械臂。

修复期间已通过保存接口保存一组真实图像和关节反馈：`outputs/calibration/20261017/pose_capture_165319/pose_000_fit`，其原始文件继续保留。旧实例退出记录曾显示电机仍通电；在用户确认机械臂放稳后，显式释放并读取六个 `Torque_Enable` 全为 0，结果见该会话目录的 `release_verified.json`，没有发送位置目标。

用户给出的 `ConnectionAbortedError: [WinError 10053]` 位于 HTTP 响应写入时，是客户端断开连接的请求线程异常。新增处理只捕获 `ConnectionError` 家族（中止、重置、断管），结束对应连接；其他程序错误继续抛出。11 项无硬件测试通过，新增覆盖响应写入和请求读取阶段断连、无关错误不被吞掉；JavaScript 回归还模拟了按钮自身 `command` 属性，确认点击仍提交 `capture/fit`。测试结果在 `outputs/calibration/20261017/pose_capture_ui_regression.json`。启动后保留终端窗口，看到 `POSE_CAPTURE_READY` 表示进入服务循环，不需要等待命令提示符返回。

### 2026-10-17 保存条件反复切换

`pose_capture_170903` 中用户反馈保存后反复等待。检查时控制循环仍运行、`following=false`、`fault=null`；重复的目标记录是保持位置。旧日志仅记录成功的操作，未记录失败保存的原因；页面返回“请点击停止跟随、停稳 1 秒，并让两路完整看到棋盘格”，当前会话保存计数为 0。

只读抽样 30 次：关节变化全部为 0、两路每次均检测到棋盘，但保存就绪只有 16 次；front 图像年龄 0.336–1.070 秒、side 0.271–0.928 秒，超过后台的 0.8 秒时限时保存会被拒绝。证据见 `outputs/calibration/20261017/pose_capture_save_diagnostic.json`。串行角点检测与页面状态更新存在时差，原按钮随就绪状态开关，可能在点击前再次禁用。

前端现允许在停止跟随且两路棋盘完整时提交保存，仍由后台检查图像新鲜度和关节稳定性；仅对明确的未就绪 422 拒绝等待重试，总计最多 5 秒。保存期间禁止重复点击和开始跟随，显示等待、成功或失败结果。网络异常、超时、重复姿态及其他错误均不自动重试，避免重复写入。运动逻辑、标定范围及后台采集质量门槛保持原值。修改通过 Node 无硬件测试：`node tests/sim2real/test_pose_capture_ui.cjs`，覆盖时序竞争、有界等待、终止错误、重复点击和状态轮询。

刷新实际页面后，单次点击已保存 `outputs/calibration/20261017/pose_capture_170903/pose_000_fit`；日志包含成功的 `capture`，`samples.json` 计数为 1。两张 PNG 和关节元数据均存在，每路 54 角点；front/side 图像年龄约 0.569/0.364 秒，最近关节反馈时间差约 7.1/5.5 毫秒。保存成功截图为 `outputs/calibration/20261017/pose_capture_save_fixed.jpg`。本次未重启服务、释放电机或发起跟随。

### 2026-10-17 晚间 11 组姿态采集复核

用户重新采集后的独立会话为 `outputs/calibration/20261017/pose_capture_214526`，包含 8 个 fit、3 个 check、22 张 640×480 PNG。离线复核重新读取全部图片并检测角点，22/22 均识别到 54 个内角点；文件、六维关节状态及实测 17 mm 棋盘尺度完整，没有重复图像文件。全部原始文件保持不变，SHA-256 清单和逐图指标保存在 `outputs/calibration/20261017/pose_capture_214526_review/audit_report.json`，同目录 `audit.py` 为复核代码。

图像与最近关节反馈时间差绝对值中位数 12.92 ms、最大 23.12 ms，保存时图像年龄最大 0.712 秒。沿用既有相机内参，逐图在棋盘坐标中拟合姿态，其重投影误差 p95 的最大值为 0.646 px；该逐图拟合仅用于数据一致性诊断，不是独立机器人外参验收。外部相机用 8 个 fit 视图拟合固定棋盘相对位姿，在 3 个 check 视图上误差中位数 0.230 px、p95 0.998 px，同样不代表已建立机器人基座坐标。退出日志记录 `torque_left_enabled=false`、`fault=null`、清理无错误；本次复核没有连接或控制机械臂。

拟合组五个身体关节跨度依次为 49.23°、42.81°、30.95°、7.47°、32.26°。腕俯仰低于现有参考姿态验收的 10° 覆盖目标，需增加腕俯仰明显变化的 fit 姿态，不把 check 移入拟合。11 组夹爪均为 5.224%，无法拟合开度到实际间距的映射，还需不同开度的实测夹爪间距。独立关节几何参考、棋盘到基座的配准、腕部相机相对末端变换仍未验证；正式配置未被改为标定通过，尚未启动仿真数据生成或混合训练。

### 晚间腕俯仰补采及跨会话复核

新增会话 `pose_capture_220458` 有 3 个 fit 姿态，6/6 图片重新识别到完整棋盘；图像与最近关节反馈时间差最大 14.37 ms，保存图像年龄最大 0.720 秒。与前批电机标定完全一致，前批文件 SHA-256 未变。引用清单 `outputs/calibration/20261017/pose_capture_combined_review/reviewed_samples.json` 汇总 11 fit + 3 check，共 14 姿态、28 图片，原目录不移动、不改写，check 划分保留。

合并统计的腕俯仰范围为 54.86–98.99°，跨度 44.13°；五个身体关节的 fit 跨度均超过 10°。用户确认补拍期间棋盘、底座和外部相机均未移动或碰到；仍保留实际测到的跨批棋盘角点中位漂移 2.32–3.99 px，不把确认当作零误差。使用全部 11 fit 在棋盘坐标中拟合固定外部相机，在原 3 check 上误差中位数 0.722 px、p95 0.932 px；这只是板相对外部相机的一致性检查，不是基座/末端外参验收。最终复核为同目录 `reviewed_report.json`，早期诊断 `audit_report.json` 保留。

两次退出记录都显示电机已释放、无故障。夹爪仍仅覆盖约 3.13–5.22%，需要补测覆盖工作开度的实际夹爪间距；关节几何参考与完整外参拟合仍待完成，本次没有启动硬件或将仿真验收门槛标为通过。

### 2026-10-18 夹爪间距测量与显示截断核验

在只读会话 `pose_capture_111147` 采集了 9 个用户尺量点，完整保存在 `outputs/calibration/20261018/gripper_measurements_001/measurements.json`，每次修改前备份，原始读数未删除或平均。约 50% 的两次复测为 50.7%/38 mm、49.7%/36 mm；早先 49.3%/47 mm 暂列疑似异常。首次近闭合读数是 3.134%/0 mm，不能改写为 0% 端点。

本次用户报告显示 100%/79 mm。告知保持夹爪不动后，通过页面关闭只读采集服务，再只读连接 COM5：5 次原始夹爪编码器均为 2732，本地和电机标定范围均为 2030–2700，按原范围未截断换算为 104.776%。驱动将 RANGE_0_100 显示截断到 0–100，因此该 79 mm 不能作为精确的 100% 间距。原始读取发生在尺量回复之后，时间关联已注明；先前 100%/82 mm 的原始位置未知，不据此倒推。证据见 `gap_008_raw_encoder_check.json`、`endpoint_review.json`。

服务退出记录 `torque_left_enabled=false`、`sent_count=0`、无故障；只读复核前后六个电机 Torque_Enable 均为 0，串口已关闭。本次没有发送目标、改写标定或修改正式仿真配置。精确 100% 端点间距及开度到仿真关节角的映射仍未验收，先结束本轮手工量尺，保留全部数据用于后续几何拟合。

### 2026-10-18 离线夹爪与手眼候选拟合

用户确认所有尺量点都取最前端相对内侧的净间距，未加胶垫或配件。确认独立保存在 `gripper_measurements_001/contact_definition_confirmed.json`。新增 `gripper-fit` 离线入口，保留 fit/check 划分，自动隔离已标记异常点、未核验的截断端点；0 mm 闭合参考单独报告，不把接触状态强行当成唯一角度。

以 23.7%/16 mm 和 76%/56 mm 两个正间距拟合点建立刚性夹爪的线性角度候选：q=0.009834456655346525\*p-0.21788822275489023 rad。三个留出点绝对误差为 1.113、0.119、0.976 mm。候选端点为外推；其低端超过资产 -10° 下限，现用 tip1 球体间距在资产下限仍约 2.626 mm，且整个低端曲线有局部非单调。因此 `gripper_fit_001/gripper_report.json` 的 passed=false，未修改资产范围、碰撞或正式场景。间距一致不等于闭合接触模型已验证。

新增 `fit_pose_diagnostic.py` 离线脚本，用已采 11 fit + 3 check、17 mm 棋盘与固定内参联合拟合。批量正运动学先与 MuJoCo 8 个数值姿态核对（1e-12）；32 组方向候选（桌面模式另枚举两种板法向），仅用 fit 选择方向和参数。无桌面约束的早期 001 诊断出现板高约 239 mm 的非物理解，不能采用。002 加入板平面 z=0 的已知条件，选择 +++++ 方向；腕部 check 中位误差 2.053 px、p95 3.095 px、最大 3.729 px。报告、输入哈希和三张独立检查叠加图均保存在 `pose_mapping_diagnostic_002`。

第一轴与末轴零位和基座/相机变换相互耦合，脚本固定这两个零位为 0 只是数学坐标约定，不是真实标定；肩、腕候选范围仍超过资产范围。旧量尺参考照片 004 到后续 pose_000 的外部棋盘角点漂移中位约 25.86 px，因此旧 AP/BP/AQ/BQ 不直接套到这批位姿。漂移来源未判断。正式配置外参和偏移仍不写入，未启动动力学成功测试、生成或训练。汇总见 `offline_calibration_review.json`。

页面新增“读取夹爪原始开度”，只有只读且未持有扭矩时允许，读取 Present_Position 不截断并记录时间，不发送目标或改写标定。修改通过 13 个采集测试、3 个夹爪拟合测试和 Node 页面回归；拟合测试包含留出读数改变不能改变拟合结果、异常/截断隔离、候选超出资产范围不能被接受。真机页面需重新启动后验证。

复现夹爪候选（新输出目录；报告 passed=false 会返回非零退出码）：

```powershell
& C:\Users\Admin\.local\bin\uv.exe run --no-project --python C:\Users\Admin\miniconda3\envs\lerobot312\python.exe --no-managed-python --with mujoco==3.8.1 --with scipy==1.18.1 python so101_r2s2r.py gripper-fit --config configs/so101_r2s2r_20261017_measured_board.json --measurements outputs/calibration/20261018/gripper_measurements_001/measurements.json --output outputs/calibration/gripper_fit_NEW
```

用户重新启动只读会话 `20261018/pose_capture_120730` 后，已实际点击“读取夹爪原始开度”，成功返回 raw=2732、104.776%，标定范围仍为 2030–2700。启动预检两臂全部扭矩为 0，检查时 armed/torque_owned/following=false、sent_count=0、无故障。两路预览与只读读取证据保存在会话的 `initial_readonly_inspection`；JPEG 预览含检测叠加、非同步，不作为新的正式标定姿态。下一步请求用户手动轻合夹爪到最前端内侧刚接触，以核对闭合原始编码器和实物接触几何。

用户将最前端内侧轻合至刚接触后，读取 raw=2054、开度 3.58209%，双相机预览支持已接触。该 0 mm 是用户接触确认，不是新尺量读数。独立检查点 `gap_009` 已追加，旧测量快照 `measurements_before_gap_009.json` 的哈希与第一版拟合报告一致，原始读数全部保留。相对于首次闭合 3.13433%，差 3 个编码器计数（0.44776 个百分点）；未要求继续挤压至驱动 0%。照片和读数位于会话 `closed_contact_001`，sent_count 仍为 0。下一步只补测 99.0–99.8% 的非截断上端间距。

### 上端非截断测量与编码器约束对照

用户回复“99°间距76”，结合夹爪页面单位与现场读取解释为约 99%/76 mm；raw=2693，对应 98.95522388%，未截断。该点追加为 check `gap_010`，总计 11 个原始测量点，保留原回复、来源与原始值；两路预览、状态和读取证据位于 `pose_capture_120730/upper_gap_001`，仍只读、sent_count=0。

原先仅用两个正间距 fit 点得到的曲线在新 check 上预测 72.60895 mm，误差 3.39105 mm，超过 3 mm 初步目标；`gripper_fit_002` 明确保留失败。另一对照按代码中 STS3215 4096 计数/圈，以及电机与夹爪角度一比一的待验证假设，固定斜率为 0.010277671278833795 rad/百分点，只用原 fit 点估计零偏。四个 check 点的诊断最大误差降到 1.80159 mm，但方案比较由检查误差触发，不能把它再称为全新的独立验收。该候选也未写入正式配置。

原 tip1 球形测距模型在 -20.05° 到 5.73° 的扫描中最小仍为 2.20332 mm，不能复现真机的刚接触 0 mm；候选闭合角还超出原资产下限。另对视觉网格最远端 3 mm 内的顶点做诊断比较，两种测距量明显不同；顶点距离不是精确三角面间距或穿透判断，不据此修改碰撞半径。脚本与报告分别为 `compare_gripper_encoder_scale.py`、`inspect_gripper_tip_mesh.py` 和 `gripper_encoder_diagnostic_001`。第一版结果、原模型和正式场景均保持不变；两份新诊断目录均保存本轮不可变测量快照。下一步需把尺量接触面与资产几何正确对应，再做闭合接触验收。

### 2026-10-18 接触面与空夹动力学修正候选

新增 `gripper-contact` 入口及 `src/lerobot/sim2real/gripper_contact.py`。检查发现旧 gap 量只对比 tip1 两球表面，并非夹爪的整个接触面。原几何的 `fixed_jaw_sph_tip3` 与 `moving_jaw_box2` 在约 -11.43315° 首次接触；另外，两爪是父子刚体，需要显式接触配对才能越过 MuJoCo 默认父子碰撞过滤。官方依据：<https://mujoco.readthedocs.io/en/stable/computation/index.html#collision-detection>。

CAD 视觉网格末端 6/8/12 mm 区域的凸包诊断给出约 -11.414° 首次接触，支持原有碰撞形状的闭合位置，但截取区域会影响张开时的最近间距，不能把任意区域的最小距离直接当成尺量定义。3 mm 探索还出现距离为零、最近点线段却非零的数值不一致，该结果作废，不用于候选选择。生产帮助函数校验距离与线段长度一致；本次正式候选使用解析球/盒接触，不使用这些探索凸包。

可复现候选位于 `outputs/calibration/20261018/gripper_contact_alignment_002`。原始资产、视觉网格、碰撞形状、质量和惯量均未改动；独立 XML 副本增加 20 个尖端接触配对及两处测距 site。以已读取的 raw=2054 / 3.58209% 作为闭合参考，保留 STS3215 4096 计数/圈、一比一传动假设：q=0.010277671278833795\*p-0.2363616507914793 rad。副本的夹爪控制/关节范围相应成为约 -13.543° 到 45.344°，是电机标定范围的映射，不代表独立量到了两个机械极限。低于首次接触的闭合目标由接触阻挡；不会要求真机用手压到 0%。原 XML 的 -10° 下限本身不允许到达已核对的接触姿态。

新的测距量取首次接触时重合的两个材料点，以后随各自爪体移动。旧有效正间距点仅作回顾诊断，最大误差 1.59656 mm；早先 49.3%/47 mm 及未核验截断端点仍隔离，全部原始读数保留。材料点与用户尺子落点的对应还没有照片确认，不能据此宣称夹爪标定完成。`calibration.py` 对带双 site 的新资产使用这个测距量，对旧资产保留历史算法；只有一个 site 时拒绝处理。

空夹对照在独立内存模型中固定其他关节、关闭重力、300 Hz 物理步进、限力矩 0.1 Nm，放宽该测试的关节限位以排除“限位代替接触”的假象。初始设置后不重置 qpos。启用接触时闭合误差约 -0.00155°、2 个接触、峰值法向力约 0.955 N；去掉接触后继续越过约 1.146°，且无接触力；重新张开后两者均无接触。该测试证明接触实现起效，不是实物力学参数标定，也不是胶带抓取成功。正式场景验收仍为 false。

已冻结候选及输入快照，再请求新的约 35%、60%、85% 三组独立检查，每次只引导一组，先做 60%。需要同时保存实际读数、原始编码器和尺子落点照片，不向用户预告预测间距，不用这些检查点再拟合参数。初步目标为 3 组误差均不超过 3 mm，接触点定义一致。当前采集服务保持 armed=false、torque_owned=false、following=false、sent_count=0；本次代码工作没有操作真机电机。

复现命令（输出必须为新目录；整体 passed=false 会返回非零退出码）：

```powershell
& C:\Users\Admin\.local\bin\uv.exe run --offline --no-project --python C:\Users\Admin\miniconda3\envs\lerobot312\python.exe --no-managed-python --with mujoco==3.8.1 --with scipy==1.18.1 python so101_r2s2r.py gripper-contact --config configs/so101_r2s2r_20261017_measured_board.json --measurements outputs/calibration/20261018/gripper_measurements_001/measurements.json --output outputs/calibration/gripper_contact_NEW
```

3 项新增无硬件回归通过：真实表面闭合与旧球距差异、显式接触/无接触对照及检查值不能改变映射、错误原始闭合值拒绝；还校验原始资产与所有 geom 属性保持不变。另 16 项已有核心测试通过，合计 19 项。Python Ruff 检查/格式化及 git diff --check 通过。

### 用户选择停止补标定，直接运行现有数据的仿真

2026-10-18 用户明确指示：“就按现在给的数据来做模拟，跳过测试校准”。此前约 35%/60%/85% 的三组补测请求取消，后续不再以这些补测作为探索仿真的前置条件。历史报告保持原状态，实验单独记录 `calibration_checks_skipped=true` 和 `calibration_passed=false`，不把跳过解释为已通过。

新增 `so101_sim_preview.py` 与 `start_so101_sim_preview.cmd`，完全离线运行，不连接串口、相机或推理服务。采用已保存的夹爪接触候选、11 个 fit 姿态、身体偏移/腕部相机估计；外部相机位姿由同批已保存棋盘角点和现有板到基座候选组合，不要求用户重新采集。副本模型的肩/腕关节范围只扩至已有 fit 姿态所需范围；原资产和正式配置未覆盖。

胶带使用已测 46/36/19 mm、3 g，笔筒使用内径 77 mm、高 95 mm、底厚 4 mm、约 5 g，构造有孔胶带与开放笔筒。未测的笔筒壁厚暂设 2 mm，胶带中心暂设在已拟合棋盘中心，笔筒相对胶带在基座 XY 偏移 (+60,+120) mm；这些是配置中可修改的仿真假设，物体颜色、背景及摩擦/伺服参数仍为近似值。

运行：

```powershell
.\start_so101_sim_preview.cmd
```

每次新建 `outputs/sim_experiment_时间戳`，输出 `scene.experimental.json`、完整 `scene.xml`、`simulation_preview.mp4`、`trajectory.json` 和 `report.json`。独立实验资产 XML 与原 vendor 资产并列，名称含本次输出目录，复用源 meshes，不覆盖原文件。

预览使用 30 Hz 目标、300 Hz 物理步进、15 FPS 四面板视频（总览、外部、腕部、状态）。初始设置之后通过仿真执行器跟随 11 组保存姿态，并执行夹爪开合；姿态之间为平滑插值，不是原始任务示范动作流。该预览没有调用 SmolVLA，不能作为策略抓取成功率、真机标定验收或训练数据成功标签。

首轮 `outputs/sim_experiment_20261018_001` 已运行完成：556 个控制步、18.533 秒仿真，渲染耗时约 130.3 秒。夹爪自身接触出现于 30 个控制步；胶带接触为 0，未抬起或入筒，与本轮仅做姿态/开合预览的范围一致。浏览器视频为 `simulation_preview_h264.mp4`，H.264、1280×960、15 FPS，完整解码核对 278 帧。`gripper_contact.png` 保存闭合时的渲染帧；原始模型 SHA-256 未变，真机动作发送数为 0。

### 2026-10-18 按实测 1 mm 壁厚改成网状笔筒

用户补充筒壁厚 1 mm、笔筒为网状；不据“铜壁”一词推断铜材质。外观沿用已保存实拍中的黑色菱形网孔。新版配置为 `configs/so101_r2s2r_20261018_mesh_holder.json`；预览启动命令不变，旧配置与上一轮实心壁输出保留。

`lattice.py` 将两组反向绕筒的窄条离散为 2,688 个凸碰撞单元，渲染和碰撞共用相同几何；孔洞保持开放。筒壁径向厚 1 mm、内径 77 mm（名义外径 79 mm），保留高 95 mm、实心底厚 4 mm。网条宽 0.45 mm、每方向 48 条、绕向斜率 1、上下边框高 2 mm 均为可调整的近似参数，不是新增实测；网壁、上下边框、底分别分配总质量的 60%、20%、20%，总质量仍为估计的 5 g。模型为可移动和倾倒的刚体，不模拟金属网的弹性变形。

3 项针对网壁的无硬件检查通过：射线同时命中网条和穿过真实孔洞、筒口开放且总质量正确；胶带自由下落后由筒底稳定支撑；封孔或不合法网格参数被拒绝。自由下落是独立几何/支撑测试，不是机械臂抓取成功。代码通过 Ruff 检查和格式检查。

新版输出目录：`outputs/sim_experiment_20261018_002_mesh`，`holder_closeup.png` 为网状笔筒近景。保留此前选择：跳过进一步实物补标定，用已有数据探索，不标记标定通过。

网壁版预览已完成 556 个控制步、18.533 秒仿真，渲染耗时约 176.5 秒。`simulation_preview_h264.mp4` 完整解码核对为 H.264、1280×960、15 FPS、278 帧；原始机器人资产哈希未变，核对结果写入 `verification.json`。仍为保存姿态和夹爪开合预览，未调用策略；胶带接触为 0，不报告抓取或入筒成功。
