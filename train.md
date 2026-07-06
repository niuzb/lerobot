# `lerobot_train.py` 中文理解指南

这份文档解释 `src/lerobot/scripts/lerobot_train.py`。它对应命令：

```bash
lerobot-train
```

这个文件不是具体模型实现，而是训练流程的编排入口。模型结构在 `src/lerobot/policies/*/modeling_*.py`，数据集结构在 `src/lerobot/datasets/lerobot_dataset.py`，数据预处理流水线在 `src/lerobot/processor/`。

## 核心职责

`lerobot_train.py` 主要做这些事：

1. 解析并校验训练配置 `TrainPipelineConfig`。
2. 初始化 `Accelerator`，支持 CPU、单 GPU、多 GPU、混合精度和分布式训练。
3. 创建训练数据集 `LeRobotDataset`。
4. 可选创建仿真评估环境。
5. 创建 policy 模型。
6. 创建 preprocessor 和 postprocessor。
7. 创建 optimizer 和 scheduler。
8. 可选从 checkpoint 恢复训练。
9. 运行离线训练循环。
10. 定期记录日志、保存 checkpoint、评估 policy。
11. 可选把模型和 processor 上传到 Hugging Face Hub。

## 文件入口关系

安装包以后，`pyproject.toml` 里有这行：

```toml
lerobot-train="lerobot.scripts.lerobot_train:main"
```

所以你运行：

```bash
lerobot-train --policy.type=act --dataset.repo_id=...
```

实际调用的是：

```python
main()
```

然后：

```python
main()
  -> register_third_party_plugins()
  -> train()
```

`train()` 被 `@parser.wrap()` 装饰。这个装饰器会自动解析 CLI 参数、读取配置文件，并构造 `TrainPipelineConfig` 后再传给真正的 `train(cfg)`。

## 两个主要函数

### `update_policy(...)`

这个函数只负责“一次参数更新”。

一次更新包含：

```text
batch
  -> policy.forward(batch)
  -> 得到 loss
  -> accelerator.backward(loss)
  -> 梯度裁剪
  -> optimizer.step()
  -> optimizer.zero_grad()
  -> lr_scheduler.step()
  -> 可选 policy.update()
  -> 写入 loss / grad_norm / lr / update_s
```

它不负责创建数据集、不负责保存 checkpoint、不负责评估。它只做训练循环里最小的一步。

### `train(cfg, accelerator=None)`

这是主函数，负责完整训练流程。

它的执行顺序可以理解为：

```text
cfg.validate()
  -> 创建 Accelerator
  -> 初始化日志/W&B/随机种子/设备
  -> make_dataset(cfg)
  -> 可选 make_env(...)
  -> make_policy(...)
  -> 可选 PEFT 包装
  -> make_pre_post_processors(...)
  -> make_optimizer_and_scheduler(...)
  -> 可选加载 RA-BC 权重
  -> 可选 resume optimizer/scheduler/step
  -> 创建 DataLoader
  -> accelerator.prepare(...)
  -> for step in range(...)
       -> batch = next(dataloader)
       -> batch = preprocessor(batch)
       -> update_policy(...)
       -> 可选 log
       -> 可选 save_checkpoint
       -> 可选 eval_policy_all
  -> 关闭 eval env
  -> 可选 push_to_hub
  -> accelerator.end_training()
```

## 关键对象说明

| 对象 | 来自哪里 | 作用 |
| --- | --- | --- |
| `cfg` | `TrainPipelineConfig` | 所有训练配置的总入口 |
| `accelerator` | `accelerate.Accelerator` | 管理设备、分布式、混合精度、DDP |
| `dataset` | `make_dataset(cfg)` | 离线训练数据 |
| `eval_env` | `make_env(cfg.env, ...)` | 训练中定期评估用的仿真环境 |
| `policy` | `make_policy(...)` | 具体策略模型，比如 ACT、Diffusion、SmolVLA |
| `preprocessor` | `make_pre_post_processors(...)` | 把 dataset batch 处理成 policy 输入 |
| `postprocessor` | `make_pre_post_processors(...)` | 把 policy 输出处理成 env/robot action |
| `optimizer` | `make_optimizer_and_scheduler(...)` | 参数优化器 |
| `lr_scheduler` | `make_optimizer_and_scheduler(...)` | 学习率调度器 |
| `train_tracker` | `MetricsTracker` | 记录 loss、lr、速度、进度等指标 |
| `wandb_logger` | `WandBLogger` | 可选 W&B 日志和 artifact 上传 |
| `rabc_weights` | `RABCWeights` | 可选 RA-BC 样本加权 |

## 为什么只让主进程做日志、保存和评估

`accelerate` 多卡训练时会启动多个进程。每个进程都参与训练计算，但不应该每个进程都写日志或保存 checkpoint。

所以代码里经常看到：

```python
is_main_process = accelerator.is_main_process
```

以及：

```python
if is_main_process:
    ...
```

主进程负责：

- 打印配置和训练日志；
- 初始化 W&B；
- 保存 checkpoint；
- 跑仿真评估；
- 上传 Hugging Face Hub。

其它进程负责计算，并通过：

```python
accelerator.wait_for_everyone()
```

和主进程同步。

## 数据加载为什么主进程先做

这段逻辑很重要：

```python
if is_main_process:
    dataset = make_dataset(cfg)

accelerator.wait_for_everyone()

if not is_main_process:
    dataset = make_dataset(cfg)
```

原因是 `make_dataset(cfg)` 可能会从 Hugging Face Hub 下载数据。如果多卡多个进程同时下载同一个数据集，可能出现竞态、文件锁或缓存冲突。

所以主进程先下载，其它进程等下载完成后再从本地缓存加载。

## Processor 在这里的作用

LeRobot 训练不是直接把 dataset batch 喂给模型，而是先走：

```python
batch = preprocessor(batch)
```

preprocessor 可能会做：

- feature 名称重命名；
- numpy/list 转 torch tensor；
- 图像增强；
- 图像/状态/action 归一化；
- 添加或整理 batch 维度；
- 移动到 CPU/GPU；
- 根据不同 policy 构造需要的字段。

评估或部署时，policy 输出也不是直接发给环境或机器人，而是通过 postprocessor 反处理。

这也是为什么保存 checkpoint 时要保存：

```python
preprocessor=preprocessor,
postprocessor=postprocessor,
```

因为没有 processor，模型权重本身是不完整的。

## 训练循环细读

主循环是：

```python
for _ in range(step, cfg.steps):
```

这里的 `step` 是已经完成的参数更新次数。`cfg.steps` 是目标总更新次数。

每次循环做：

1. `batch = next(dl_iter)`  
   从无限循环的 dataloader 中取 batch。

2. `batch = preprocessor(batch)`  
   把原始 batch 转成 policy 可以吃的格式。

3. `update_policy(...)`  
   做一次 forward/backward/optimizer step。

4. `step += 1`  
   因为这一步已经完成，所以更新 step 计数。

5. 判断是否到了日志、保存、评估频率：

```python
is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process
is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0
```

6. 如果需要，执行日志、checkpoint、eval。

## Checkpoint 保存了什么

保存函数是：

```python
save_checkpoint(
    checkpoint_dir=checkpoint_dir,
    step=step,
    cfg=cfg,
    policy=accelerator.unwrap_model(policy),
    optimizer=optimizer,
    scheduler=lr_scheduler,
    preprocessor=preprocessor,
    postprocessor=postprocessor,
)
```

这说明 checkpoint 不只是模型权重，还包括：

- 当前 step；
- 训练配置；
- policy；
- optimizer；
- scheduler；
- preprocessor；
- postprocessor。

这让后续 resume 和部署推理都有完整上下文。

## 评估逻辑

训练中评估只有在配置了 `cfg.env` 且 `cfg.eval_freq > 0` 时发生。

评估调用：

```python
eval_policy_all(...)
```

它会把当前 policy 放到仿真环境中 rollout，得到：

- 平均 reward；
- 成功率；
- 评估耗时；
- 每个 suite/task 的指标；
- 可选视频路径。

真实机器人评估通常不在这个训练脚本里做，而是用独立评估流程。

## RA-BC 是什么

这个文件里有可选的 RA-BC 加权逻辑：

```python
if cfg.use_rabc:
    rabc_weights = RABCWeights(...)
```

开启后，`update_policy` 会让 policy 返回 per-sample loss，然后按样本权重加权：

```text
weighted_loss = sum(weight_i * loss_i) / sum(weight_i)
```

这通常用于根据 SARM progress 等外部信号，让训练更关注某些样本或阶段。

## PEFT 是什么

如果配置里有：

```python
cfg.peft is not None
```

policy 会被：

```python
policy.wrap_with_peft(...)
```

包装。PEFT 常见形式是 LoRA/adapter。它通常只训练一小部分参数，适合微调大模型。

所以日志中：

```python
num_learnable_params
num_total_params
```

对 PEFT 很有用，因为可训练参数量会明显小于总参数量。

## 读这个文件时最该抓住的主线

如果你只想快速理解，不要陷入每个配置字段，先抓住这条线：

```text
dataset -> preprocessor -> policy.forward -> loss -> optimizer.step
                 |
                 +-> checkpoint 保存 processor

policy -> postprocessor -> env eval
```

也就是说：

- 训练时主要路径是 `dataset -> preprocessor -> policy -> loss`；
- 评估时主要路径是 `env observation -> env_preprocessor/preprocessor -> policy -> postprocessor/env_postprocessor -> env action`；
- 保存时必须保存 policy 和 processor。

## 推荐继续阅读

理解完这个文件后，建议按这个顺序读：

1. `src/lerobot/configs/train.py`  
   看 `TrainPipelineConfig` 到底有哪些字段。

2. `src/lerobot/policies/factory.py`  
   看 `make_policy` 和 `make_pre_post_processors` 如何根据 `policy.type` 分发。

3. `src/lerobot/policies/pretrained.py`  
   看所有 policy 的统一接口。

4. `src/lerobot/processor/pipeline.py`  
   看 preprocessor/postprocessor 的抽象。

5. `src/lerobot/datasets/lerobot_dataset.py`  
   看 batch 原始数据来自哪里。

6. 某个具体 policy，比如：
   - `src/lerobot/policies/act/modeling_act.py`
   - `src/lerobot/policies/diffusion/modeling_diffusion.py`
   - `src/lerobot/policies/smolvla/modeling_smolvla.py`

