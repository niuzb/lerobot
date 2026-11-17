# SO-101：在 LeIsaac 中把胶带放进笔筒

代码位于 [`scripts/simulation/so101_tape_leisaac/`](../../scripts/simulation/so101_tape_leisaac/)。
这是依赖外部场景、机器人资产和模型缓存的实验工具集，不是安装仓库后即可直接启动的独立示例。
本文保留 2026 年 11 月实验记录，并补充各入口的输入与用途；历史记录不表示任务今天仍在运行，
也不构成启动新训练或保留计算资源的授权。2026-11-17 的仓库整理未重新验证远端运行状态或真机效果。

## 代码和运行环境

全部脚本保持在同一目录，使用脚本路径启动。`prepare_spatial.py` 会把整套代码复制到运行目录的
`code/`；其他编排器也依赖这样的完整快照，不能只复制单个入口文件。

| 用途                   | 入口与配套模块                                                                                                                                                                      |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 场景与物理采集         | `prepare_assets.py`、`collect.py`；`bootstrap.py`、`task.py`、`planner.py`、`common.py`                                                                                             |
| 相机与审核             | `camera_views.py`、`reference_views.py`、`view_audit.py`、`render.py`、`prepare_review.py`                                                                                          |
| 空间和纹理增强         | `prepare_spatial.py`、`run_spatial.py`；`spatial.py`、`texture_randomization.py`                                                                                                    |
| 数据导出与合并         | `export.py`、`merge_batch.py`                                                                                                                                                       |
| 100 条轨迹与监督训练   | `batch100.py`、`train_batch.py`                                                                                                                                                     |
| 真仿混合训练与离线评估 | `run_mixed.py`、`train_mixed.py`、`eval_real_loss.py`、`compare_mixed.py`；`mixed_data.py`                                                                                          |
| 在线 GRPO 与课程       | `run_grpo.py`、`grpo_probe.py`、`grpo_actor.py`、`grpo_env.py`；`grpo_objective.py`、`grpo_wire.py`、`grpo_curriculum.py`                                                           |
| 上游 SDE 实现          | [`flowvla_rl/UPSTREAM.md`](../../scripts/simulation/so101_tape_leisaac/flowvla_rl/UPSTREAM.md) 和随附 MIT [LICENSE](../../scripts/simulation/so101_tape_leisaac/flowvla_rl/LICENSE) |

使用两个独立环境：

- **仿真环境**：历史记录使用 Isaac Sim 5.1.0、Python 3.11 和 IsaacLab 2.3.0。
  附加依赖见 [`requirements-sim.txt`](../../scripts/simulation/so101_tape_leisaac/requirements-sim.txt)，
  该文件不包含 Isaac Sim、LeIsaac 源码或机器人资产，也不要把这些依赖直接装入仓库的主环境。
  `bootstrap.py` 从 `RUN/site-packages` 加载独立依赖，且需要 `--leisaac-root` 指向源码检出目录。
- **数据与训练环境**：需要当前仓库的 LeRobot、SmolVLA 相关依赖及本地模型缓存。
  导出、审核和混合数据处理使用 OpenCV、PyArrow、NumPy；审核视频还调用 PATH 中的 `ffmpeg`。
  训练和在线 actor 使用 CUDA。编排器使用 `fcntl`，混合训练编排器还读取 `/proc`，按 Linux 环境运行。

运行前准备以下输入，仓库不附带这些实验产物：

1. `RUN/scene.json`：包含 `asset` 指向的本地可信 MJCF 及其网格、`joint_map`、六维
   `initial_robot_state`、`table`、`tape`、`holder`、`generation` 和双相机 `cameras` 配置。
   相机参数包括父坐标系、内参 `K` 和外参 `T_parent_cv`；沿用原始来源记录及标定状态。
2. 独立的 Isaac 启动器、`RUN/site-packages`、LeIsaac 源码和 `--leisaac-assets` 指向的资产目录。
   `prepare_assets.py` 将场景中的 MJCF 转换到新的 `RUN/assets`，不会补齐缺失的源网格。
3. 训练所用的配置、数据集、缓存权重；需要时创建完整的 `RUN/code/` 快照。
   数据、USD/网格、视频、检查点、日志和运行 manifest 放在外部运行目录，不作为源代码提交。
4. `batch100.py` 额外要求已准备的 `request.json`（100 条独立轨迹请求和自动训练字段）、
   通过的 `preflight.json`、代码快照和训练配置。它不负责从空目录创建这些输入。

可参考 [Isaac Sim 环境记录](./ISAAC_SIM_MERLIN_SETUP_RECORD.md) 和
[SO-101 R2S2R 流程](./SO101_R2S2R_RUNBOOK.md) 查找原场景背景。

## 任务设计

这个任务复用 LeIsaac 捡橘子任务使用的 `SingleArmTaskEnvCfg`、单臂场景、
关节动作接口、双相机传感器和 `StateMachineBase` 执行流程。
新增任务名为 `LeIsaac-SO101-TapePenHolder-v0`。
上游任务见 [PickOrange 配置](https://github.com/LightwheelAI/leisaac/blob/main/source/leisaac/leisaac/tasks/pick_orange/pick_orange_env_cfg.py)。

胶带与笔筒使用已有实验的尺寸、质量和网壁结构。机器人由原 MJCF 通过官方导入器转换，
保留既有夹爪接触点、关节映射和相机父坐标系，避免直接换用另一个资产后改变数据含义。
导入结果采用单一机械臂根节点、固定基座，并保留真实碰撞体。

## 数据含义

- 状态和动作为六维：五个身体关节的度数，加夹爪开度百分比。
- 物理频率 300 Hz，控制和数据采样频率 30 Hz。
- `observation.images.front` 是腕部相机，`observation.images.side` 是外部相机；均为 640×480。
- 起始姿态来自真实训练 episode 1 的第 180 帧，不使用原验证集生成轨迹。
- 相机内参与畸变沿用之前的配置；原配置的 `calibration_passed=false` 保留。
- 原始真实数据和 MuJoCo 数据保持只读。新数据写入独立目录，不自动上传 Hub。

自动控制只发送关节目标，不在抓放动作中写入胶带、笔筒或机械臂的姿态。
成功检查要求双爪接触、抬高至少 3 cm、水平搬运至少 3 cm、释放、完整入筒、
由笔筒支撑、低速度且笔筒直立，并连续稳定至少 30 个控制周期。
失败尝试保留数值记录，不能导出为成功训练示范。

达到连续稳定 30 个控制周期后，成功终止该 episode。
图像导出单独回放 PhysX 实际记录的全部连杆、胶带和笔筒位姿，逐帧检查 USD 世界位姿和关节状态；
这种渲染回放不参与成功判定。两路视频还必须随动作发生实际画面变化，以排除相机缓存停留在首帧。
首次相机预热可以有空帧，正式逐帧录制不允许缺失相机帧。

本运行对接触采用零报告门槛，避免 IsaacLab 2.3 布尔启用接口将它写成 1 N，漏掉轻小物体的支撑力。
采用 32/16 次位置／速度求解迭代，并关闭物体休眠与场景稳定化，保持接触证据可观测。

## 分阶段运行

以下命令从仓库根目录运行。先把占位路径替换为已经准备好的环境和输入；`SIM` 是接受 Python
脚本及参数的独立 Isaac 启动器，`PYTHON_LEROBOT` 指向已安装当前仓库及训练依赖的 Python。
历史 Merlin 的启动器为 `/root/data/leisaac/runs/a30-probe-20261108-01/run_native.sh`，
运行目录为 `/root/data/leisaac/runs/tape-pilot-20261108-01`，仅作为来源记录。

```bash
CODE="$PWD/scripts/simulation/so101_tape_leisaac"
SIM=/absolute/path/to/run_native.sh
RUN=/absolute/path/to/prepared_run
LEISAAC=/absolute/path/to/leisaac
ASSETS=/absolute/path/to/leisaac_assets
PYTHON_LEROBOT=/absolute/path/to/lerobot_environment/bin/python
UV="$(command -v uv)"
```

独立运行时依赖记录在 `requirements-sim.txt`。先准备 `RUN/scene.json`，
其中保留原场景参数，并加入只读提取的 `initial_robot_state` 和 `migration` 来源记录。

准备资产（输出目录必须不存在）：

```bash
"$SIM" "$CODE/prepare_assets.py" --config "$RUN/scene.json" --output "$RUN/assets"
```

检查环境或生成物理记录：

```bash
"$SIM" "$CODE/collect.py" --run "$RUN" \
  --leisaac-root "$LEISAAC" --leisaac-assets "$ASSETS" \
  --output "$RUN/environment_check_NEW" --inspect-only

"$SIM" "$CODE/collect.py" --run "$RUN" \
  --leisaac-root "$LEISAAC" --leisaac-assets "$ASSETS" \
  --output "$RUN/physics_NEW" --count 5 --max-attempts 25
```

对成功物理记录生成完整双相机视频：

```bash
"$SIM" "$CODE/render.py" --run "$RUN" --physics "$RUN/physics_NEW" \
  --leisaac-root "$LEISAAC" --leisaac-assets "$ASSETS" --output "$RUN/render_NEW"
```

`--attempt N --stride 10` 可用于查看某次失败的低帧率诊断预览。
这类输出会标记为不可用于训练，导出器会拒绝它。

在已有 LeRobot 训练环境中导出 v3 数据：

```bash
uv run --no-project --offline --python "$PYTHON_LEROBOT" \
  python "$CODE/export.py" --run "$RUN" --physics "$RUN/physics_NEW" \
  --render "$RUN/render_NEW" --output "$RUN/dataset_NEW"
```

验收以实际 `generation_manifest.json`、`render_manifest.json` 和 `dataset_verification.json`
为准，不能以进程存在或目录已创建认定成功。导出器检查每路视频的实际帧数、FPS、图像尺寸，
并将全部导出状态和动作与物理记录逐行核对。

这个小批次用于检查流程和数据内容；不代表已经完成真实场景标定或真机效果评估。

## 2026-11-08 小批次训练记录

当时停止了小批次训练，等待查看 5 条视频，并保留了 A30 分配。
此前短跑文件作为运行记录保留，不能据此认定视频通过人工审核；恢复旧训练前需核验检查点完整性。
以下配置描述当时的试验，不代表当前资源或进程状态。

该次 Merlin 运行在 `RUN/smolvla_pilot_config.json` 中记录完整训练参数，输出为 `RUN/smolvla_train`。
从缓存的 `lerobot/smolvla_base` 微调动作专家与状态投影，保留两个图像输入；
4 条 episode 训练、1 条验证，batch 8、BF16、2,000 步，每 200 步保存检查点并计算验证损失。
归一化使用本批次的数据统计，相机字段沿用已有的 `front→camera1`、`side→camera2` 映射。

```bash
uv run --no-project --offline --python "$PYTHON_LEROBOT" \
  python -m lerobot.scripts.lerobot_train --config_path="$RUN/smolvla_pilot_config.json"
```

复用旧输出前先核查进程和检查点，避免重复启动。资源分配与释放应依据当前任务安排，不沿用历史状态。

## 双相机调整记录（v2，历史待审核状态）

`camera_views.py` 生成 `RUN/camera_views_v2.json`，保留原相机配置并记录本次调整。
腕部相机移到夹爪另一侧，朝向胶带操作区；外部相机重新对准工作区并放大，采用无畸变针孔视图。
这些是仿真机位调整，`calibration_passed=false`，未进行真机镜头或支架标定。

新 5 条成功动作来自 `RUN/physics_camera_v2`（seed 从 60100 开始），
对应视频为 `RUN/review_camera_v2`，与之前的动作和视频分别保存。
`view_audit.py` 检查全部物理帧：外部视角完整容纳胶带和笔筒，腕部视角完整容纳胶带，至少留 16 像素边距。
夹爪遮挡仍须通过实际渲染图像复查，不能仅凭投影范围认定通过。

```bash
"$SIM" "$CODE/render.py" --run "$RUN" --physics "$RUN/physics_camera_v2" \
  --leisaac-root "$LEISAAC" --leisaac-assets "$ASSETS" \
  --camera-config "$RUN/camera_views_v2.json" --output "$RUN/render_camera_v2_NEW"
```

`prepare_review.py` 将左右拼接视频转换为可在浏览器中播放的 H.264，完整解码验收后输出预览总图和 `review_manifest.json`。
当时只采集、渲染和准备审核视频，没有在这一阶段启动训练。

## 按真实照片调整朝向（v3）

当时使用的本机参考目录为 `outputs/smolvla_preflight_20261108_114746`：
`camera1.jpg` 实际是腕部，`camera2.jpg` 实际是外部。按画面内容确定角色，不能沿用其他拍摄目录的编号假设。
`reference_views.py` 记录参考图哈希，调整腕部镜头的真实旋转，使夹爪从画面下方向上伸出；不对捕获图像做镜像或倒置处理。
外部视角按参考图调整为机械臂从上方伸入、胶带在前、笔筒在右。

本次沿用 `physics_camera_v2` 中 attempt 9 的已验证抓放动作，仅重渲染一条视频用于相机对比。
相机配置为 `RUN/camera_views_v3_candidate.json`，完整渲染为 `RUN/reference_v3_render`，
播放版为 `RUN/reference_v3_review/episode_01_dual_camera.mp4`。
这是照片引导的朝向和构图调整，真实照片与模拟动作姿态并不完全相同；不视为完成精确相机标定。
当时训练保持停止；这里不推断当前训练或 A30 分配状态。

## 桌面高度和相机姿态增强

`prepare_spatial.py` 为不同桌面高度和相机安装位置生成独立场景；不会修改运行中的旧批次。
默认使用桌面相对原场景上移 2 cm、不变、下移 2 cm 三个条件。
桌面、机器人基座、胶带和笔筒一起移动，保持机器人固定安装及物体接触关系；
外部相机保持在世界坐标中，因此相机相对于工作台的视点会真实变化。
规划器把世界位置变回机器人基座坐标计算关节目标，成功判定仍使用实际 PhysX 接触。

默认按固定 seed 在每个条件中随机调整相机安装：腕部每轴 ±3 mm、每轴旋转 ±2°；
外部相机每轴 ±1 cm、每轴旋转 ±3°。这些安装扰动在一条 episode 中保持不变，
腕部相机的世界位姿则随真实机械臂连杆逐帧变化。内参不变，图像不做翻转或伪造视角变换。
全帧构图检查不通过时，只在相同范围内重采样相机机位，保留所有候选配置和检查报告。

当前世界坐标把原桌面定义为零点，因此 ±2 cm 是**相对基准桌面的高度变化**，
不代表测得桌面实际离地高度。

```bash
SOURCE=/absolute/path/to/prepared_source_run
OUTPUT=/absolute/path/to/new_spatial_run

uv run --no-project --offline --python "$PYTHON_LEROBOT" \
  python "$CODE/prepare_spatial.py" --source-run "$SOURCE" \
  --camera-config "$SOURCE/camera_views.json" --output "$OUTPUT" \
  --height-deltas=0.02,0,-0.02 --seed 71000

uv run --no-project --offline --python "$PYTHON_LEROBOT" \
  python "$OUTPUT/code/run_spatial.py" --root "$OUTPUT" \
  --sim "$SIM" --leisaac-root "$LEISAAC" --leisaac-assets "$ASSETS" \
  --python "$PYTHON_LEROBOT" --uv "$UV"
```

每个高度条件先生成一条物理成功演示，再逐帧渲染、导出并验证；不会启动策略训练或释放 GPU。
`status.json` 记录整体进度，各条件的 `dataset_verification.json` 和 `review/review_manifest.json`
必须通过才能计为完成。失败日志及物理记录保留，不在已有输出目录重复运行。

导出器用 `--spatial-features` 启用新版状态。旧模式仍保持六维状态／动作。

| 字段                            | 维度与含义                                               |
| ------------------------------- | -------------------------------------------------------- |
| `observation.state`             | 25：原六维关节状态 + 桌面高度 1 + 两路相机位置／旋转各 9 |
| `action`                        | 原六维关节目标，角度／夹爪百分比含义不变                 |
| `observation.table_height`      | 1，世界坐标中相对原桌面零点的高度，单位 m                |
| `observation.camera_pose.front` | 7，实际腕部相机在机器人基座系中的 XYZ + 四元数 wxyz      |
| `observation.camera_pose.side`  | 7，实际外部相机在机器人基座系中的 XYZ + 四元数 wxyz      |

状态中的旋转用相机到基座旋转矩阵的前两列编码；相机光学轴为右 X、下 Y、前 Z。
腕部相机位姿从记录的真实连杆位姿计算，不从目标动作估计。导出器逐行核对全部空间字段，
并验证实际读取出的状态批次为 `[2,25]`、动作批次为 `[2,6]`。
25 维状态可落在当前 SmolVLA 的 32 维状态上限内；训练时必须使用本批的新状态维度和统计，
不能直接沿用旧六维数据的归一化参数，也不能把两种状态格式直接混合。

`calibration_passed=false`、`human_review_pending=true` 保留。空间条件增强不等于完成真机标定，
也不代表已验证训练效果或真机迁移性能。

## 同一轨迹内逐帧改变纹理

`render.py --framewise-textures` 在每次正式图像采集前独立切换桌面、背景、胶带、笔筒和机器人
外观。每个组有独立随机序列，相邻帧的纹理编号不会相同；同一帧的两路相机共享同一组材质。
这与上一节的相机安装参数不同：安装参数在 episode 内固定，纹理在每个记录帧都改变。

默认每组 16 张、128×128 的可复现程序纹理，包括棋盘、木纹、编织、斑点、条纹和瓷砖图案。
使用真正的 RGB 贴图、UV 坐标和 USD 材质连接；没有下载外部图片，也不是只修改颜色编号。
桌面使用无碰撞的视觉顶面，背景使用工作区后方的无碰撞背景板。
所有修改只存在于渲染场景，不保存回原 USD 资产。

```bash
"$SIM" "$CODE/render.py" --run "$RUN" --physics "$RUN/physics" \
  --leisaac-root "$LEISAAC" --leisaac-assets "$ASSETS" \
  --camera-config "$RUN/camera_candidate_01.json" --output "$RUN/render_textures_NEW" \
  --framewise-textures --texture-seed 81000 --texture-count 16 --texture-subframes 4
```

在新的空间条件批次上，可给 `run_spatial.py` 加同名开关 `--framewise-textures`，其导出步骤
会自动保留纹理证据。`--max-frames 8` 只用于快速诊断；截断轨迹的渲染会标为不可导出训练数据。

纹理切换后使用多个渲染子帧，避免采集到尚未加载的新材质。
Isaac Sim 5.1 官方说明材质随机化时至少需要两个子帧，本示例默认四个；
见 [Replicator troubleshooting](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/replicator_tutorials/troubleshooting.html)。
USD 贴图连接采用 [UsdPreviewSurface / UsdUVTexture](https://openusd.org/release/spec_usdpreviewsurface.html)。

在正式录制前，先固定全部物体和相机位姿，只切换一个材质组，检查每组是否产生可见像素变化。
再固定全部纹理、改变物理回放帧，验证相机没有停留在旧画面。
另检查碰撞体、物理属性及物理材质关系指纹没有改变，原始物理 NPZ 的 SHA256 保持一致。

每个渲染 episode 的 `appearance.jsonl` 逐帧记录源帧索引、各组纹理编号和双相机捕获哈希；
`texture_bank/manifest.json` 记录每张 PNG 的图案和 SHA256。
导出器检查每一帧覆盖、相邻帧切换、双相机无重复冻结画面、全部贴图及记录哈希，
并复制到数据集 `meta/appearance/`。这些编号仅为审计元数据，不进入策略状态，
六维动作和已有空间状态含义保持不变。

因为每帧都会独立换纹理，30 FPS 的预览会有快速闪变；这不是视频丢帧或在 episode 边界才换图。

## 100 条独立轨迹与自动训练 SmolVLA（2026-11-09 实验配置）

该实验设计为按逐帧纹理方式生成 100 条轨迹，全部完成后自动开始训练。
`batch100.py` 要求运行清单中的明确请求字段及预检通过；本文不替新实验设置授权字段，
也不把真机标定标记改成通过。

历史实验目录为 `/root/data/leisaac/runs/tape-train100-20261109-01`，主入口是独立快照的 `code/batch100.py`。
它生成新的物理成功，不把旧轨迹重复渲染计作 100 条；物理 seed 与 NPZ 哈希均须唯一。
桌面 +2/0/-2 cm 轮换，相机安装独立扰动，五个纹理组在每个记录帧切换。
纹理库 seed 固定以便复现，各轨迹通过 `--texture-schedule-seed` 使用独立的逐帧序列。

流程按序执行：

1. 采集并验证一个独立物理成功，再做全帧构图检查、逐帧纹理渲染与 LeRobot 导出。
2. 普通抓取／构图失败保留证据但不计数；GPU、SDK 或数据完整性异常会明确停止，不掩盖为成功。
3. 收齐且验证恰好 100 条后，通过原生 `aggregate_datasets` 合并。视频按文件复制，不重新编码。
4. 合并时逐值核对全部状态、动作与空间字段，并对每条轨迹抽样解码比较原视频和合并视频。
5. 复制并重新索引逐帧纹理记录，去重存储纹理库，保存轨迹、物理 seed、纹理与相机来源映射。
6. 验证 100 条均独立且完整后，自动调用 `train_batch.py`，由原生 `lerobot_train` 管理训练、验证与检查点。

| 项目       | 历史实验配置                                                                        |
| ---------- | ----------------------------------------------------------------------------------- |
| 数据格式   | 25 维状态、六维动作、30 FPS、双相机 640×480                                         |
| 训练／验证 | 按完整轨迹划分 0–89 / 90–99，避免同一物理轨迹跨集合                                 |
| 高度分布   | 前 90 条中每档 30 条；验证 10 条为 4/3/3                                            |
| 模型来源   | 已缓存的固定 `lerobot/smolvla_base` 快照，重新微调，不续用旧六维 pilot 的优化器状态 |
| 训练       | batch 8、BF16、20,000 步、学习率 1e-4、warmup 1,000 步                              |
| 保存／验证 | 每 1,000 步保存并评估，验证使用全部 10 条轨迹                                       |
| 可训练部分 | 动作专家和状态投影；视觉编码器冻结                                                  |
| 归一化     | 只从训练 90 条计算数值统计，不沿用旧六维统计，也不使用验证轨迹拟合统计              |
| 外部服务   | Hub 上传和 W&B 关闭，使用本地缓存                                                   |

`preflight.json` 记录实际 batch 8 的前向／反向、有限损失／梯度和显存检查。
该预检没有执行 optimizer step，也没有保存更新后的权重，不算提前开始正式训练。
`aggregate_api_preflight.json` 验证无损合并接口；预检数据不计入这 100 条。

运行时读取 `status.json`、`ledger.json` 与对应阶段日志。
`accepted` 只在整条轨迹导出验证完成后增加；`training_started=false` 时不得宣称模型已在训练。
自动训练必须等待合并后的 `dataset_verification.json` 显示恰好 100 条且全部通过。

```bash
ROOT=/absolute/path/to/prepared_100_trajectory_run
uv run --no-project --offline --python "$PYTHON_LEROBOT" \
  python "$ROOT/code/batch100.py" --root "$ROOT" \
  --source-run "$SOURCE" --sim "$SIM" --python "$PYTHON_LEROBOT" --uv "$UV" \
  --leisaac-root "$LEISAAC" --leisaac-assets "$ASSETS"
```

上面的命令只适用于已准备好请求清单、预检结果和代码快照的运行目录。
恢复历史运行前先确认旧进程已结束，再加 `--resume`；
已验证轨迹通过 ledger 复用，未完成目录保留为 `_incomplete_*` 后再运行。
已有训练检查点时使用其原生配置恢复；没有有效检查点的训练输出会保留并要求检查。
每个阶段有超时和子进程组清理，异常会写入 `status.json`，而不会继续进入训练。

## 真仿混合训练与离线评估

`run_mixed.py` 编排固定的 real78 + sim100 实验：88 条真实轨迹中保留 10 条验证，
每个训练 batch 严格包含 4 条真实样本和 4 条仿真样本，使用单 GPU、batch 8、20,000 步。
`mixed_data.py` 只用训练数据计算按真实／仿真各占一半加权的数值统计；
`train_mixed.py` 在每次优化前检查数据来源及验证集隔离，并写入 `batch_balance.json`。
真实数据原有六维状态通过存档场景和运动学补齐空间字段，这些字段是估计值，不是当时记录的标定测量。

**这组入口保留实验专用条件。** `run_mixed.py` 固定了源运行目录、真实数据目录、场景文件、
基础模型快照、100 条／55,175 帧仿真数据和 3,104 帧真实验证数据的检查；其 `--root` 仅选择输出目录，
不会覆盖源输入。迁移到其他机器前，应逐项核对并调整代码中的路径及对应数据约束，然后创建完整
`MIXED/code/` 快照。不要为了运行任意数据而删除完整性校验。

已有匹配输入和代码快照时，入口为：

```bash
MIXED=/absolute/path/to/prepared_mixed_run
uv run --no-project --offline --python "$PYTHON_LEROBOT" \
  python "$MIXED/code/run_mixed.py" --root "$MIXED"
```

编排器先写 `request.json`、`train_config.json`，完成训练后分别对旧仿真模型和新混合模型做比较。
续跑使用 `--resume`；已有训练输出却没有有效检查点时会停止，保留现场。
底层 `train_mixed.py --root ... --config ... [--resume]` 依赖同一份 `request.json`，不负责准备输入。

`compare_mixed.py` 也读取该请求清单，在相同真实验证集上比较物理单位动作误差。
例如，对已经训练好的新模型单独生成报告：

```bash
CHECKPOINT="$MIXED/train/checkpoints/020000/pretrained_model"
uv run --no-project --offline --python "$PYTHON_LEROBOT" \
  python "$MIXED/code/compare_mixed.py" --root "$MIXED" \
  --checkpoint "$CHECKPOINT" --label new_mixed
```

旧模型使用 `--label old_sim_only` 及其检查点。最终查看 `comparison.json`、
`old_sim_only_physical_metrics.json`、`new_mixed_physical_metrics.json` 和 `batch_balance.json`。
归一化统计发生变化后，两个模型各自的原生验证损失不能直接作为同尺度改进量；
报告使用关节角度 MAE、夹爪百分比点 MAE 和共同训练尺度上的动作误差。
这些是离线监督评估，不是闭环抓放成功率。

`eval_real_loss.py` 则保留更早的固定仿真模型评估：要求指定的 88 条、六维状态真实数据，
并校验特定 step-20000 模型 SHA256；即使提供任意 `--checkpoint`，也会拒绝不匹配的模型。
该入口用于复核原实验，参数如下：

```bash
REAL_ROOT=/absolute/path/to/matching_real_dataset
REAL_REPO_ID=local/matching_real_dataset
SIM_CHECKPOINT=/absolute/path/to/matching_sim_step20000/pretrained_model
EVAL_OUTPUT=/absolute/path/to/new_real_loss_report
uv run --no-project --offline --python "$PYTHON_LEROBOT" \
  python "$CODE/eval_real_loss.py" --real-root "$REAL_ROOT" --real-repo-id "$REAL_REPO_ID" \
  --checkpoint "$SIM_CHECKPOINT" --scene "$RUN/scene.json" --output "$EVAL_OUTPUT"
```

## 在线 GRPO 和逆向课程

`run_grpo.py` 用 Unix socket 连接一个 `grpo_actor.py` 策略进程和一个 `grpo_env.py` 实时仿真进程。
每组 8 次 rollout，按组内二元物理奖励计算标准化 advantage；相同组使用相同初始条件，
成功判定继续要求实际接触、抓取、搬运、释放、支撑和连续稳定。
它使用连续 Flow-SDE，不能作为离散动作 token 论文的复现实验。
`flowvla_rl/` 的来源版本、数值路径与本地兼容改动见 [上游说明](../../scripts/simulation/so101_tape_leisaac/flowvla_rl/UPSTREAM.md)。

运行前需准备完整 `GRPO/code/`、`scene.json`、`assets/`、`site-packages/`，以及匹配的模型和处理器。
`run_grpo.py` 中的初始模型路径和 SHA256、Isaac 启动器、LeIsaac 源码与资产路径仍是历史机器值；
`grpo_env.py` 中的预检物理记录路径、`grpo_probe.py` 中的模型与数据路径也需逐项核对。
`--worker-id` 只记录现有计算资源标识，不分配新 GPU。`--root` 不会自动创建上述输入或改变这些路径。

完成输入适配后，运行入口为：

```bash
GRPO=/absolute/path/to/prepared_grpo_run
WORKER_ID=12345  # 替换为实际资源标识
uv run --no-project --offline --python "$PYTHON_LEROBOT" \
  python "$GRPO/code/run_grpo.py" --root "$GRPO" --worker-id "$WORKER_ID"
```

`grpo_probe.py --root ...` 用匹配数据上的单个动作块检查采样与重评分的一致性、有限动作和非零梯度；
它不替代实时仿真的成功率验证。协调器把现场写入 `request.json`、`coordinator_status.json`、
`status.json`、`actor.log`、`simulation.log` 和 `checkpoints/`。已有 `request.json` 的输出目录会被拒绝，
没有通用 `--resume` 开关。

`grpo_curriculum.py` 默认按 `release → transport → lift → approach → full` 从任务后缀逐步回退；
还支持显式配置 `open_hold`。每个 anchor 必须来自实际物理记录，并校验前缀中的抓取／抬升／搬运事实。
课程 JSON 需要 `source_physics` 及其 SHA256、`initial_tape_position`、`source_friction`、
`anchors`（每个阶段的 `tick`）和 `schedule`；仓库没有从空场景自动生成已审核课程的入口。
新 rollout 需要重新累计稳定周期，后缀阶段成功不能记为完整任务成功。

从已有 GRPO 检查点继续课程时，使用新输出目录，同时提供 `--parent-run` 和 `--curriculum`：

```bash
GRPO_NEXT=/absolute/path/to/prepared_continuation_run
PARENT=/absolute/path/to/parent_grpo_run
PARENT_CHECKPOINT="$PARENT/checkpoints/final"
CURRICULUM=/absolute/path/to/audited_curriculum.json
PARENT_SHA256=replace_with_verified_model_sha256
uv run --no-project --offline --python "$PYTHON_LEROBOT" \
  python "$GRPO_NEXT/code/run_grpo.py" --root "$GRPO_NEXT" --worker-id "$WORKER_ID" \
  --parent-run "$PARENT" --parent-checkpoint "$PARENT_CHECKPOINT" \
  --expected-parent-sha256 "$PARENT_SHA256" --curriculum "$CURRICULUM"
```

父检查点必须有 `pretrained_model/`、非空 `optimizer.pt` 和有效 `training_state.json`。
在同一 50-epoch 预算内续跑时，额外传 `--epoch-offset N --epochs M`，要求 `N + M = 50`，
且父状态 epoch 与 N 一致。需要恢复原课程进度时再加 `--restore-curriculum`，其课程文件哈希必须匹配父状态。
`curriculum_stalled` 和 `no_learning_signal` 是明确停止状态，不能记为完成训练。

## 本地检查

从仓库根目录运行六组无需 Isaac Sim 启动的测试：

```bash
uv run --no-sync pytest \
  tests/sim2real/test_isaac_batch.py tests/sim2real/test_isaac_curriculum.py \
  tests/sim2real/test_isaac_grpo.py tests/sim2real/test_isaac_mixed.py \
  tests/sim2real/test_isaac_spatial.py tests/sim2real/test_isaac_textures.py -q
```

这些测试检查独立数据校验、坐标／空间特征、纹理记录、采样比例、GRPO 目标和课程状态逻辑。
它们不启动 NVIDIA 仿真、下载模型或训练策略，不能代替 GPU 上的物理、相机和完整训练验收。
