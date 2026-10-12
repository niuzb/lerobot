# SO101 VLA-JEPA 训练准备（2026-10-11）

用户要求训练 LeRobot 支持的 JEPA 相关模型。选择 `lerobot/VLA-JEPA-Pretrain`，使用现有 SO101 双相机胶带数据微调，不从 π0.5 权重转换。

## 当前状态

### 2026-10-11 23:14 更新

**23:34 预检通过，23:34:53 已启动正式训练进程。** r3 预检退出码 0，4 步共 2 次优化器更新，eval_loss=1.8624（仅 8 样本）。action_loss 1.159–1.388、加权 wm_loss 0.156–0.159 均有限；Qwen、动作头、世界预测器梯度均已确认非零且有限；预测 shape=[2,7,6] 且有限。峰值 allocated=26.4565 GiB / reserved=26.8672 GiB。完整权重、优化器 checkpoint 已保存。
正式训练从适配后的原始基础权重重新开始，不接续预检；计划 batch=2、累积=8、40,000 小批次、有效 batch=16、5,000 优化器更新，每 5,000 小批次全验证/保存。

**23:38:33 正式训练已完成第 40 步，继续运行。** loss=1.422，action_loss=1.263，wm_loss=0.158，grad_norm=1.879，lr=1.1e-7（warmup）。首批初始化后约 1.3–1.4 小批次/秒；短时速率不作为保证完成时间。每 30 分钟自动检查正式日志与资源，正常推进不重复通知。

旧 worker `4394460` 在 06:32（北京时间）因缺少 diffusers 退出，未开始训练。
用户已授权补齐依赖、重新申请 GPU 并启动训练，也授权尝试多卡。
远端已安装 diffusers 0.39.0 / qwen-vl-utils 0.0.14，Qwen3VL、AutoVideoProcessor 和 VLAJEPAPolicy 导入通过。
同机双卡申请 `4396392` 因资源每台仅剩一张一直排队，已撤销。
当前 worker `4396405` 已获得单张 NVIDIA A100-SXM4-80GB；每次连接仍必须查询 `mlx worker list`。
权重已全部下载完成；处理器生成需要 CUDA，最后在 GPU worker 上完成。成功日志 `vla_jepa_assets_20261011.gpu.log`，退出码 `vla_jepa_assets_20261011.exitcode` 为 0。
首次准备因代理连接失败，重试统一设置大小写代理变量；HTTP 下载较慢，随后开启 Xet 并行准备三个固定版本模型。
新预检脚本 `vla_jepa_preflight_checked_20261011.py` 检查有限损失、Qwen/动作头/世界预测器非零梯度、有限 6 维预测和显存峰值。
启动脚本 `run_vla_jepa_checked_20261011.sh` 支持 torchrun 数据并行，正式训练保持有效批量 16、5,000 次优化器更新；单卡 40,000 小批次，双卡 20,000 小批次。
GPU 已通过 CUDA 11.8 / PyTorch 2.7.0 BF16 矩阵前向、反向及有限梯度验证。
`start_vla_jepa_pipeline_20261011.sh` 已在当前 GPU worker 启动，等待资产后执行预检；预检成功且生成审计 JSON / checkpoint 后自动进入正式训练，任何错误都会停止。
- 流水线日志 / 退出码：`vla_jepa_pipeline_20261011.log` / `.exitcode`。
- 当前预检日志 / 退出码：`vla_jepa_tape_20261011_preflight_r3.log` / `.exitcode`。
- 当前预检审计：`outputs/vla_jepa_launch_20261011/vla_jepa_tape_20261011_preflight_r3.audit.rank0.json`。
- 正式日志 / 退出码：`vla_jepa_tape_20261011_r1.log` / `.exitcode`。
- 正式输出：`outputs/train/vla_jepa_tape_20261011_r1`。
自动检查已改为监控当前 worker，旧日志不再作为当前状态。

预检前两次失败已修复：r1 为检查脚本缺少 multiprocessing 主入口保护；r2 为 ImagePrepProcessorStep 将 `observation.images.*_is_pad` 布尔掩码误作图像。图像处理修复已通过相关 13 项测试（含两种掩码维度的回归测试）。r3 预检使用 num_workers=0，正式训练仍为 2。
权重加载按计划重初始化 4 个 action/state 维度相关张量；源 checkpoint 还含有本实现未定义的 video_predictor extrinsics_encoder/state_encoder 的 4 个权重，加载时忽略，日志保留记录。

### 初始准备记录（历史）

已提交 A100-SXM-80GB / 8 CPU / 64 GiB 的 GPU 预检任务，worker `4394460`；记录时没有分配 podIP/SSH 端口，尚未开始训练、下载完整权重或进行 GPU 前后向验证。
申请由登录机 nohup 持有。资源分配后执行 `start_vla_jepa_20261011.sh`，依次检查 CUDA/BF16 和依赖、下载固定版本资产、准备数据适配、执行两小批次训练/验证/保存预检。预检结束后需检查结果再启动正式训练；不会自动启动机械臂或上传模型。
较早试分配的 jiuhuashan worker 实际显示 `NVIDIA_VISIBLE_DEVICES=none`，已释放；空闲占位 A100 申请已被本 GPU 预检任务替代。

## 配置

- 底座：Qwen3-VL-2B + V-JEPA2 + DiT 动作头。
- `freeze_qwen=false`、`enable_world_model=true`。本实现冻结 Qwen 会自动关闭世界模型损失，不能将冻结动作头训练称作完整 JEPA 微调。
- `causal_world_model_context=true`，避免预测器输入的编码特征看到未来帧；保留 8 帧视频窗口与世界模型损失权重 0.1。
- 双相机 `front`（腕部）/ `side`（外部），Qwen 图像预处理 224×224。
- SO101 action/state 均为 6 维，绝对关节位置；夹爪索引 5，物理值 0–100。
- 关闭源 checkpoint 的 `pre_snap_gripper_action` 和 `binarize_gripper_action`。
- 仅允许动作编码器、动作解码器和状态编码器的形状不匹配层重新初始化；其他权重沿用预训练模型。
- 复用 `/root/code/lerobot/datasets/pi05_tape_v2_20261006` 的独立元数据和训练集统计量；79 条训练、9 条验证，原数据不修改。
- 预检：batch 2，累积 2，2 小批次，4 帧验证，保存 checkpoint。必须检查 action_loss / wm_loss、梯度流、预测维度和显存后再长训。
- 候选正式预算：batch 2，累积 8，40,000 小批次，5,000 次参数更新，lr 1e-5；每 5,000 步全量验证并保存。最终预算与 batch 以 GPU 预检结果为准，尚无可靠耗时估计。

## 固定版本

- `lerobot/VLA-JEPA-Pretrain`: `e946c3e5b538d760f4b4ff239d1b1c12090c041d`
- `Qwen/Qwen3-VL-2B-Instruct`: `89644892e4d85e24eaac8bacfd4f463576704203`
- `facebook/vjepa2-vitl-fpc64-256`: `b3c1679b7c34d3255ef3547f27c7b226aefab26f`

元信息位于 `outputs/vla_jepa_launch_20261011/`，已同步远端。
准备脚本 `prepare_vla_jepa_20261011.py` 已通过语法编译；训练脚本 `train_vla_jepa_20261011.sh` 已通过 bash 语法检查；公开 checkpoint 配置可由当前 LeRobot 解析。上述检查不等于完成模型加载或训练。

## 远端检查位置

登录 `37944.niuzhibin.ws@merlin-ssh-proxy-cn.byted.org:22222`，共享目录 `/root/code/lerobot`。
- 分配日志：`vla_jepa_gpu_preflight_20261011.launch.log`
- 预检流水线日志：`vla_jepa_tape_20261011.log`
- 退出码：`vla_jepa_tape_20261011.exitcode`
- 准备成功标记：`outputs/vla_jepa_launch_20261011/prepared.json`
- 预检 checkpoint：`outputs/train/vla_jepa_tape_20261011_preflight/checkpoints/000002`

正式命令保存在训练脚本末尾，目前注释，需预检通过后再执行。
