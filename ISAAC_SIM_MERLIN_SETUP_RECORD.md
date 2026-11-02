# Isaac Sim Merlin 环境搭建记录

记录日期：2026-11-02。镜像完整性验证完成于北京时间 12:16。

目标是在 Merlin 上运行 LeIsaac / Isaac Sim，用 Windows 连接 SO-101 主臂控制仿真从臂，在 Windows 查看实时画面，并在 Merlin 保存 LeRobot 格式数据。

**当前已完成官方镜像下载和完整性校验、场景资产准备、遥操作及传输代码实现和部分独立测试；尚未启动真实仿真，也没有采集到真实仿真样本。**

GPU 状态是搭建过程中的检查快照，不是实时监控结果。继续操作前应重新查询节点状态。

## 1. 环境与职责

| 位置 | 用途 | 已确认的信息 |
| --- | --- | --- |
| 本机 Mac | 开发、部署管理、网络验证、保存本记录 | 本地 LeIsaac 仓库位于 `/Users/bytedance/code/ai/robot/leisaac` |
| Windows 电脑 | 连接 SO-101 主臂、运行串流客户端及备用预览 | 尚未完成该电脑上的串口、校准和遥操作验收 |
| Merlin CPU 开发机 | 保存代码、资产、镜像和缓存 | 开发机 `37944`，Debian 11，自身运行在受限容器中 |
| 原 GPU worker | 先前的硬件和网络检查 | `4461508`，2 张 A10、16 核 CPU、256 GiB 内存；后续已无法登录 |
| GPU 恢复申请 | 申请恢复原规格 | `4464933`；本次镜像准备开始时仍在排队，未确认分配成功 |

Merlin 登录入口：

```bash
ssh -p 22222 37944.niuzhibin.ws@merlin-ssh-proxy-cn.byted.org
```

登录后只读查询 GPU worker：

```bash
NO_COLOR=1 TERM=dumb mlx worker list
```

不要把历史 worker ID 或历史 IP 当作下一次运行的有效配置，也不要因本记录出现了恢复申请编号就认为 GPU 已经就绪。

## 2. 已完成与待完成

| 项目 | 状态 | 验证边界 |
| --- | --- | --- |
| LeIsaac 代码同步到 Merlin | 已完成 | 代码基线为 `24d3bcd`，另有本次未提交的部署及遥操作修改 |
| SO-101 / LiftCube 资产下载 | 已完成 | 资产路径及下载 SHA256 已记录在 Merlin |
| 官方 Isaac Sim 5.1.0 镜像下载 | 已完成 | `linux/amd64`，约 7.09 GiB |
| 镜像完整性校验 | 已完成 | 固定 manifest、17 个镜像层及配置 blob 全部通过 SHA256 校验 |
| 缓存及日志目录准备 | 已完成 | 新建运行目录按官方容器用户 `1234:1234` 设置所有者 |
| SSH 数据包转接 | 已通过独立测试 | Mac 到 Merlin 的真实网络回环，不是 WebRTC 视频验收 |
| 断线保护、TCP 转接、备用预览 | 已通过部分组件测试 | 包含 SDK 替身和合成图像测试，尚未与真实仿真集成验收 |
| Docker Engine / GPU 容器运行入口 | 未就绪 | 当前开发机不能直接按宿主机方式启动嵌套 Docker |
| IsaacLab / LeIsaac / LeRobot 仿真栈集成 | 待完成 | 不能把官方 Isaac Sim 镜像下载完成视为整套依赖已安装 |
| Vulkan / RTX 渲染及实际 WebRTC | 待验证 | 当前没有通过完整图形兼容检查，也没有真实串流画面 |
| Windows 主臂控制及数据采集 | 待验证 | 没有已验收的真实采集样本 |

本次没有升级宿主显卡驱动、替换系统 GLIBC、修改平台安全限制，或改动原有训练环境。准备镜像阶段没有新增 GPU 申请，也没有取消原排队申请。

## 3. 镜像与目标版本

### 3.1 已下载的官方镜像

| 属性 | 值 |
| --- | --- |
| 镜像名称 | `nvcr.io/nvidia/isaac-sim:5.1.0` |
| 平台 | `linux/amd64` |
| 基础层系统 | Ubuntu 24.04.2 LTS，通过基础层的 `os-release` 核对 |
| 镜像层数 | 17 |
| 归档字节数 | 7,617,878,016 |
| 完整性校验时间 | 2026-11-02 12:16:52，Asia/Shanghai |
| Docker 运行 / GPU 测试 | 均未完成 |

固定镜像 manifest：

```text
sha256:93b0f99635ab126fb5b33298d513c11520f119f0ee60ff8414ccef67ea977829
```

镜像配置 digest：

```text
sha256:781acaa42e829e01a43706ab79649b8e7494a368cecc90edfbfc71341e09d4cd
```

下载和校验遵循 [NVIDIA Isaac Sim 5.1 容器安装文档](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_container.html) 的镜像版本。由于当前开发机没有可用 Docker 服务，使用官方 go-containerregistry 的 `crane 0.22.1` 下载 Docker 镜像归档，而不是执行 `docker pull`。

`crane` 发行包经过发布方 SHA256 校验，仅放在专用工具目录，没有写入系统 PATH。镜像归档使用固定 digest 下载，随后逐层校验；没有把仅获取到镜像元数据当作下载成功。

### 3.2 LeIsaac 集成目标

以下是后续集成的目标组合，不是已运行环境清单：

| 组件 | 目标版本 |
| --- | --- |
| Isaac Sim | 5.1.0 |
| IsaacLab | 2.3.0 |
| 仿真 Python | 3.11 |
| PyTorch / CUDA wheel | 2.7.0 / CUDA 12.8 |
| torchvision | 0.22.0 |
| LeRobot | 0.4.2 |
| NumPy | 1.26.0 |

版本约束保存在 `scripts/deployment/requirements-sim.txt`。本次没有构建此前编写的自定义 `Dockerfile`，也没有完成这些依赖在官方镜像中的安装及联合验证。

## 4. 文件与目录

### 本机目录

| 内容 | 位置 |
| --- | --- |
| LeIsaac 仓库 | `/Users/bytedance/code/ai/robot/leisaac` |
| 本记录 | `/Users/bytedance/code/ai/robot/leisaac/ISAAC_SIM_MERLIN_SETUP_RECORD.md` |
| 详细部署操作手册 | `/Users/bytedance/code/ai/robot/leisaac/scripts/deployment/README.md` |
| 官方镜像准备与启动入口 | `/Users/bytedance/code/ai/robot/leisaac/scripts/deployment/official_container.py` |

### Merlin 持久化目录

| 内容 | 位置 |
| --- | --- |
| 代码工作副本 | `/root/code/leisaac-merlin` |
| 官方镜像归档 | `/root/data/leisaac/images/isaac-sim-5.1.0-amd64.tar` |
| 官方 manifest | `/root/data/leisaac/images/isaac-sim-5.1.0-amd64.manifest.json` |
| 完整性验证报告 | `/root/data/leisaac/images/isaac-sim-5.1.0-amd64.verified.json` |
| 下载层缓存 | `/root/data/leisaac/images/layer-cache/` |
| 下载工具 | `/root/data/leisaac/tools/crane-v0.22.1/crane` |
| Isaac Sim 运行缓存与日志 | `/root/data/leisaac/isaac-sim-cache/` |
| SO-101 模型 | `/root/data/leisaac/assets/robots/so101_follower.usd` |
| LiftCube 场景 | `/root/data/leisaac/assets/scenes/table_with_cube/scene.usd` |
| 资产下载摘要 | `/root/data/leisaac/assets/download-manifest.json` |
| 原环境预检结果 | `/root/data/leisaac/preflight-20261102.json` |

运行缓存包含 `cache/main`、`cache/computecache`、`logs`、`config`、`data`、`pkg`，对应官方容器挂载位置。只设置了新建缓存目录的所有者，没有递归修改项目或数据集权限。

更换 GPU 任务或开发机时，需要显式挂载或复制这些文件，不能假设新任务自动共享当前目录。镜像归档和层缓存会分别占用存储空间；`.partial` 文件不能作为完成产物使用。

## 5. 环境限制与 GPU 申请经过

### 5.1 当前开发机不能直接运行嵌套 Docker

CPU 开发机虽然以 root 身份运行，但没有宿主机的完整权限：

- 未发现 Docker daemon、Docker socket 或可用的其他容器运行时。
- 缺少 `CAP_SYS_ADMIN`。
- 单独创建 user namespace 的检查成功，但组合 user/mount/pid namespace 时，挂载 `/proc` 被拒绝。
- 没有为绕过限制修改权限、安全策略或系统服务。

因此，当前完成的是**镜像和启动材料准备**，不是 Docker Engine 已安装运行。拿到 GPU 后仍需满足以下一种条件：

1. Merlin 支持以指定官方镜像直接启动 GPU 任务或开发机；此时不需要在容器内部再运行 Docker。
2. 提供真正具备 Docker Engine 和 NVIDIA Container Toolkit 的 GPU 主机，再加载归档运行。

当前普通 worker 创建接口没有独立镜像选项。下载 Ubuntu 镜像不会把现有 Debian 工作台自动变成 Ubuntu，申请普通 GPU worker 也不会自动解除容器运行限制。

### 5.2 GPU 与驱动的已有证据

原 worker 的两张 A10 驱动均为 `535.261.03`，每张可见显存为 `23028 MiB`。此前 720p NVENC 编码冒烟测试通过。

A10 有 RT Core 和硬件编码能力，但这不等于完整 Isaac Sim 场景已验证可运行。缺少 NVIDIA Vulkan ICD 配置，也不能单独证明显卡没有图形能力。

Isaac Sim 5.1 文档列出的 Linux 驱动测试版本是 `580.65.06`。当前 `merlin.py` 采用这一保守门槛；它不是“535 必然不能运行”的证据。NVIDIA 还记录了 `535.256+` 在 Vulkan 下的版本误报问题。后续应以官方兼容检查和实际渲染结果判断，不能只靠版本号，也不应默认关闭检查。

### 5.3 GPU 节点变化

1. 最初确认 `4461508` 仍在运行，两张 A10 均空闲；当时没有执行释放操作。
2. 后续通过 SSH 和平台 API 发现 worker 数量为 0，原节点无法登录。消失原因未确认，不能断言是自动回收或人为释放。
3. 按恢复请求先尝试 Public Arnold 资源池，被 `candy-lq` 与 `gpuspot1-lq` 不支持跨集群操作的限制拒绝。
4. 改用原工作台资源池，申请 `4464933`，配置为 2 张 A10、16 核 CPU、256 GiB 内存。平台接受申请，但当时仍在等待调度。
5. 用户决定先准备镜像、后续再处理 GPU。镜像准备期间未新增 GPU 申请，也未更改已有排队申请。

继续申请资源前，先确认 `4464933` 是否已经就绪，避免重复申请。

## 6. 遥操作与画面回传方案

Windows 主臂输入通过独立 SSH 控制通道发送到 Merlin；仿真画面优先使用 WebRTC，经 SSH 回传；若原生串流不能满足要求，则使用仿真相机的 MJPEG 预览。

| 用途 | Windows 本地入口 | 容器转接入口 | 最终服务 |
| --- | --- | --- | --- |
| 主臂关节输入 | TCP 5556 | TCP 5557 | ZMQ SUB `127.0.0.1:5556` |
| WebRTC 信令 | TCP 49100 | TCP 49101 | Isaac Sim TCP 49100 |
| WebRTC UDP | UDP 47998 | TCP 47999 | Isaac Sim UDP 47998 |
| 备用相机预览 | HTTP `127.0.0.1:8080` | TCP 8080 | 带会话认证的 HTTP 服务 |

普通 SSH 转发只覆盖 TCP；UDP 视频通信需要双向封装并保留数据包边界。转接使用会话认证、有限队列和旧连接清理，控制与视频使用独立 SSH 连接。

Merlin 列表中的 SSH 入口地址与实际 GPU 容器地址不同，应用端口必须转发到后者。客户端读取当前运行状态取得容器地址，不应硬编码历史地址。

SSH 加密终止于登录节点，登录节点到 GPU 容器仍经过平台内网 TCP；没有宣称这一段也是端到端 SSH 加密，也没有新增公网端口或第三方视频中继。

### 已实现的保护与记录逻辑

- 保留六关节消息格式，增加发布端 `--connect` 和仿真端 `--remote_bind`。
- 首个有效输入到达前禁止开始运动；超过 500 ms 无有效输入时，使当前 episode 失效并暂停控制，恢复后需手动重新开始。
- 键盘和备用页面共用 `B` 开始、`R` 失败重置、`N` 成功保存的控制入口。
- 部署配置对齐物理 60 Hz、decimation 2、控制及记录 30 Hz、相机 30 Hz。
- LiftCube 的腕部相机作为显式选项恢复，不改变原任务默认设置。
- 数据来自仿真相机、实际关节状态和动作，不来自串流截图；默认不上传数据集。

这些逻辑已经实现，但仍需真实仿真与 Windows 主臂联合验收。

## 7. 已完成的验证

| 验证 | 结果与限制 |
| --- | --- |
| 新版转接代码的 Mac → SSH → Merlin 回环 | 10 次全部成功；64～8192 字节；中位 37.2 ms，最大 71.9 ms |
| 网络重连与认证 | 重连通过；错误令牌被拒绝；发现并修复了旧连接未及时释放导致重连受阻的问题 |
| ZMQ 接收与输入保护 | 真实 socket/接收线程检查通过；Isaac Device 基类使用替身，不能代替 SDK 集成测试 |
| 双相机 JPEG / MJPEG 输出 | 合成图像编码和 HTTP 输出通过，不是真实仿真相机画面 |
| 预览安全与命令队列 | 会话认证、跨站来源拒绝、过期命令丢弃检查通过 |
| 镜像校验器 | manifest、内容校验、篡改拒绝和不安全归档路径检查通过；真实镜像完整性校验通过 |
| 静态检查 | 对相关修改执行了 Python 编译、Black、Flake8 及补丁空白检查 |

**37.2 ms 是数据包往返时间，不是视频延迟；Mac 测试结果也不能直接代表 Windows。** 没有完成 ICE/DTLS/WebRTC 视频、连续 10 分钟运行或真实数据样本验收。

## 8. 后续操作顺序

### 8.1 在当前 CPU 开发机核对归档

```bash
cd /root/code/leisaac-merlin
python3 scripts/deployment/official_container.py verify
```

需要复用完整层缓存继续下载、校验及准备缓存目录时：

```bash
python3 scripts/deployment/official_container.py prepare
```

### 8.2 GPU 主机及容器运行入口就绪后

以下命令要求该主机实际具有可用 Docker Engine 和 NVIDIA Container Toolkit，并能访问镜像归档、项目和缓存目录：

```bash
cd /root/code/leisaac-merlin
python3 scripts/deployment/official_container.py load
```

阅读并接受 NVIDIA 许可后再执行：

```bash
export ACCEPT_EULA=Y
python3 scripts/deployment/official_container.py check-gpu
python3 scripts/deployment/official_container.py stream
```

启动入口不设置 `PRIVACY_CONSENT=Y`。停止该实例使用：

```bash
docker stop isaac-sim-leisaac
```

`official_container.py stream` 只启动官方 Isaac Sim，用于基础验证；它不会自动加载 LeIsaac 任务、主臂程序或整套 SSH 转接。完整遥操作入口是 `scripts/deployment/merlin.py`，须在依赖集成完成后使用，具体 Windows 和录制命令见部署操作手册。

不要同时运行两套占用相同串流端口的 Isaac Sim 实例。先完成官方镜像的基础检查，再停止该实例并进入 LeIsaac 集成阶段。

### 8.3 最终验收清单

- [ ] GPU 节点就绪，且具备自定义镜像或 Docker/NVIDIA 容器运行入口。
- [ ] 官方兼容检查通过，真实 Vulkan/RTX 渲染和编码可用。
- [ ] 固定版本的 IsaacLab、LeIsaac、LeRobot 完成集成，LiftCube 正常加载。
- [ ] Windows 显示真实画面连续 10 分钟；目标不低于 15 FPS、可见动作反馈 P95 不高于 300 ms。
- [ ] SO-101 主臂完成实际 COM 口识别、校准和逐关节小幅运动验证。
- [ ] 分别验证成功、失败及断线 episode，失效输入不能继续驱动或保存为成功数据。
- [ ] 生成至少一条成功样本，并通过 `validate_dataset.py` 实际解码及时间轴检查，再人工回放核对内容。
- [ ] 上述验收通过后再开始批量采集。

## 9. 排查经验与注意事项

- 镜像下载完成、Docker 加载成功、GPU 兼容检查通过、真实视频出现是四个不同阶段，不应相互替代。
- 当前阻碍不只是“缺少 GPU”。受限容器的运行权限、图形库注入和宿主驱动都需要分别确认。
- 535 驱动不是仅凭版本号即可否定；下载新版 Ubuntu 镜像也不会升级宿主驱动。
- 非交互且不读标准输入的 SSH 查询使用 `ssh -n` 可避免等待输入；通过管道或 here-document 传文件/脚本时不能使用 `-n`，否则内容会被丢弃。
- 部分系统 Python 缺少 HTTPS 支持；资产准备改用可用的 `curl`，没有为了下载而修改系统 Python。
- 新任务使用新 run 目录，避免覆盖已有数据；迁移节点时先确认持久化文件是否可见。
- 不把私钥、会话令牌或账号凭据写入记录、日志分享或仓库。本记录不含这些凭据。

## 10. 参考资料

- [NVIDIA Isaac Sim 5.1 容器安装](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_container.html)
- [Isaac Sim 5.1 系统要求](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)
- [Isaac Sim 5.1 Python 安装要求](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html)
- [Isaac Sim WebRTC 客户端](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/manual_livestream_clients.html)
- [Omniverse 驱动已知问题](https://docs.omniverse.nvidia.com/dev-guide/latest/common/technical-requirements.html#known-issues-and-limitations)
- [NVIDIA A10 规格](https://www.nvidia.com/en-us/data-center/products/a10-gpu/)
- [LeIsaac 安装说明](https://lightwheelai.github.io/leisaac/docs/getting_started/installation/)
- [LeIsaac 远程遥操作](https://lightwheelai.github.io/leisaac/docs/getting_started/teleoperation/)

本机详细操作手册：`/Users/bytedance/code/ai/robot/leisaac/scripts/deployment/README.md`。
