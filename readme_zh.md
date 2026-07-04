<p align="center">
  <img alt="LeRobot, Hugging Face Robotics Library" src="./media/readme/lerobot-logo-thumbnail.png" width="100%">
</p>

<div align="center">

[![Tests](https://github.com/huggingface/lerobot/actions/workflows/nightly.yml/badge.svg?branch=main)](https://github.com/huggingface/lerobot/actions/workflows/nightly.yml?query=branch%3Amain)
[![Python versions](https://img.shields.io/pypi/pyversions/lerobot)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/huggingface/lerobot/blob/main/LICENSE)
[![Status](https://img.shields.io/pypi/status/lerobot)](https://pypi.org/project/lerobot/)
[![Version](https://img.shields.io/pypi/v/lerobot)](https://pypi.org/project/lerobot/)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v2.1-ff69b4.svg)](https://github.com/huggingface/lerobot/blob/main/CODE_OF_CONDUCT.md)
[![Discord](https://img.shields.io/badge/Discord-Join_Us-5865F2?style=flat&logo=discord&logoColor=white)](https://discord.gg/q8Dzzpym3f)

</div>

**LeRobot** 旨在为真实世界机器人提供基于 PyTorch 的模型、数据集和工具。它的目标是降低入门门槛，让每个人都能贡献并受益于共享数据集和预训练模型。

- 与硬件无关、Python 原生的接口，可在不同平台之间标准化控制，从低成本机械臂（SO-100）到人形机器人皆可覆盖。

- 标准化且可扩展的 LeRobotDataset 格式（Parquet + MP4 或图像），托管在 Hugging Face Hub 上，支持大规模机器人数据集的高效存储、流式读取和可视化。

- 最先进的策略模型，已被证明可以迁移到真实世界，并可用于训练和部署。

- 全面支持开源生态，助力物理 AI 民主化。

## 快速开始

LeRobot 可以直接从 PyPI 安装。

```bash
pip install lerobot
lerobot-info
```

> [!IMPORTANT]
> 如需详细安装指南，请参阅[安装文档](https://huggingface.co/docs/lerobot/installation)。

## 机器人与控制

<div align="center">
  <img src="./media/readme/robots_control_video.webp" width="640px" alt="Reachy 2 Demo">
</div>

LeRobot 提供统一的 `Robot` 类接口，将控制逻辑与具体硬件细节解耦。它支持广泛的机器人和遥操作设备。

```python
from lerobot.robots.myrobot import MyRobot

# Connect to a robot
robot = MyRobot(config=...)
robot.connect()

# Read observation and send action
obs = robot.get_observation()
action = model.select_action(obs)
robot.send_action(action)
```

**支持的硬件：** SO100、LeKiwi、Koch、HopeJR、OMX、EarthRover、Reachy2、Gamepads、Keyboards、Phones、OpenARM、Unitree G1。

虽然这些设备已经原生集成到 LeRobot 代码库中，但该库被设计为易于扩展。你可以轻松实现 Robot 接口，从而将 LeRobot 的数据采集、训练和可视化工具用于自己的自定义机器人。

如需详细硬件设置指南，请参阅[硬件文档](https://huggingface.co/docs/lerobot/integrate_hardware)。

## LeRobot 数据集

为了解决机器人领域的数据碎片化问题，我们使用 **LeRobotDataset** 格式。

- **结构：** 用于视觉数据的同步 MP4 视频（或图像），以及用于状态/动作数据的 Parquet 文件。
- **HF Hub 集成：** 在 [Hugging Face Hub](https://huggingface.co/lerobot) 上探索数千个机器人数据集。
- **工具：** 无缝删除 episode、按索引/比例切分、添加/移除特征，并合并多个数据集。

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Load a dataset from the Hub
dataset = LeRobotDataset("lerobot/aloha_mobile_cabinet")

# Access data (automatically handles video decoding)
episode_index=0
print(f"{dataset[episode_index]['action'].shape=}\n")
```

可在 [LeRobotDataset 文档](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)中了解更多信息。

## SoTA 模型

LeRobot 使用纯 PyTorch 实现了最先进的策略模型，覆盖模仿学习、强化学习和视觉-语言-动作（VLA）模型，未来还会加入更多模型。它还提供了用于监控和检查训练过程的工具。

<p align="center">
  <img alt="Gr00t Architecture" src="./media/readme/VLA_architecture.jpg" width="640px">
</p>

训练策略就像运行一个脚本配置一样简单：

```bash
lerobot-train \
  --policy=act \
  --dataset.repo_id=lerobot/aloha_mobile_cabinet
```

| 类别 | 模型 |
| --- | --- |
| **模仿学习** | [ACT](./docs/source/policy_act_README.md), [Diffusion](./docs/source/policy_diffusion_README.md), [VQ-BeT](./docs/source/policy_vqbet_README.md) |
| **强化学习** | [HIL-SERL](./docs/source/hilserl.mdx), [TDMPC](./docs/source/policy_tdmpc_README.md) & QC-FQL（即将推出） |
| **VLA 模型** | [Pi0Fast](./docs/source/pi0fast.mdx), [Pi0.5](./docs/source/pi05.mdx), [GR00T N1.5](./docs/source/policy_groot_README.md), [SmolVLA](./docs/source/policy_smolvla_README.md), [XVLA](./docs/source/xvla.mdx) |

与硬件类似，你也可以轻松实现自己的策略，并利用 LeRobot 的数据采集、训练和可视化工具，还可以将模型分享到 HF Hub。

如需详细策略设置指南，请参阅[策略文档](https://huggingface.co/docs/lerobot/bring_your_own_policies)。

## 推理与评估

使用统一的评估脚本，在仿真环境或真实硬件上评估你的策略。LeRobot 支持 **LIBERO**、**MetaWorld** 等标准基准，未来还会支持更多基准。

```bash
# Evaluate a policy on the LIBERO benchmark
lerobot-eval \
  --policy.path=lerobot/pi0_libero_finetuned \
  --env.type=libero \
  --env.task=libero_object \
  --eval.n_episodes=10
```

你可以按照 [EnvHub 文档](https://huggingface.co/docs/lerobot/envhub)学习如何实现自己的仿真环境或基准，并通过 HF Hub 进行分发。

## 资源

- **[文档](https://huggingface.co/docs/lerobot/index)：** 教程和 API 的完整指南。
- **[中文教程：LeRobot+SO-ARM101 中文教程-同济子豪兄](https://zihao-ai.feishu.cn/wiki/space/7589642043471924447)：** 关于组装、遥操作、数据集、训练和部署的详细文档。已由 Seeed Studio 和 5 位全球黑客松参与者验证。
- **[Discord](https://discord.gg/q8Dzzpym3f)：** 加入 `LeRobot` 服务器，与社区讨论交流。
- **[X](https://x.com/LeRobotHF)：** 在 X 上关注我们，获取最新进展。
- **[机器人学习教程](https://huggingface.co/spaces/lerobot/robot-learning-tutorial)：** 一门免费的实践课程，用 LeRobot 学习机器人学习。

## 引用

如果你在研究中使用 LeRobot，请引用：

```bibtex
@misc{cadene2024lerobot,
    author = {Cadene, Remi and Alibert, Simon and Soare, Alexander and Gallouedec, Quentin and Zouitine, Adil and Palma, Steven and Kooijmans, Pepijn and Aractingi, Michel and Shukor, Mustafa and Aubakirova, Dana and Russi, Martino and Capuano, Francesco and Pascal, Caroline and Choghari, Jade and Moss, Jess and Wolf, Thomas},
    title = {LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
    howpublished = "\url{https://github.com/huggingface/lerobot}",
    year = {2024}
}
```

## 贡献

我们欢迎社区中的每个人参与贡献！如需开始，请阅读我们的 [CONTRIBUTING.md](./CONTRIBUTING.md) 指南。无论你是添加新功能、改进文档，还是修复 bug，你的帮助和反馈都非常宝贵。我们对开源机器人技术的未来感到无比兴奋，也迫不及待想与你一起探索下一步。感谢你的支持！

<p align="center">
  <img alt="SO101 Video" src="./media/readme/so100_video.webp" width="640px">
</p>

<div align="center">
<sub>由 <a href="https://huggingface.co">Hugging Face</a> 的 <a href="https://huggingface.co/lerobot">LeRobot</a> 团队倾情打造</sub>
</div>
