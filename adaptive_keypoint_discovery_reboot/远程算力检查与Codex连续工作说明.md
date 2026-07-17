# 远程算力检查与 Codex 连续工作说明

## 一、先确认 VS Code 终端确实在远程服务器

1. 在 VS Code 左下角确认显示 `SSH: 服务器名`，而不是本地窗口。
2. 选择“终端 → 新建终端”。
3. 在这个新终端运行：

```bash
hostname
nvidia-smi
```

`hostname` 用于防止误在本机执行；`nvidia-smi` 显示 GPU 型号、显存、驱动版本和当前占用。如果是 RTX 3090，通常会在型号中直接出现 `RTX 3090`，显存一栏约为 24 GB。

只看 `nvidia-smi` 还不够，因为它只能证明服务器有 NVIDIA 驱动。还需要确认“准备训练的 Python 环境”能调用 CUDA。在远程项目根目录运行：

```bash
python remote_gpu_check.py
```

脚本会输出：

- 当前主机名和 Python 路径；
- GPU 型号、显存、驱动和计算能力；
- PyTorch 版本及其 CUDA 构建版本；
- `torch.cuda.is_available()`、设备数量和每张卡的名称；
- 一句可直接阅读的结论。

补充检查命令：

```bash
which python
python -m pip show torch
nvcc --version
```

如果出现版本属性不存在，先检查拼写：正确形式是 `torch.__version__`，前后各有两个下划线。远端 `kf` 环境已从家目录和 `~/kp` 分别实测，二者都正确导入 `/media/neaucs2/evs/envs/kf/lib/python3.14/site-packages/torch/__init__.py`，不存在 `~/torch` 遮蔽。复核命令如下：

```bash
cd ~/kp
python -c "import torch; print('file=', torch.__file__); print('version=', torch.__version__); print('cuda=', torch.cuda.is_available()); print('cuda_build=', torch.version.cuda); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable')"
```

当前确认结果为 PyTorch `2.9.1+cu128`、CUDA 构建 `12.8`、`torch.cuda.is_available() == true`，GPU 为 NVIDIA GeForce RTX 3090。

`nvcc` 不存在不等于 PyTorch 不能训练；以 `torch.cuda.is_available()` 为最终环境判断。完整 CUDA Toolkit 只在编译自定义 CUDA 扩展时才是硬要求。

## 二、同一账号能否让当前工作“完全自动迁移”

不能把“登录同一账号”直接理解成以下内容会自动出现在服务器：

- 本机 `D:\kp` 中的代码、图像、实验输出；
- 本机 `C:\Users\F\.codex\memories` 中的本地记忆文件；
- 本次任务已经读取过哪些文件、运行过哪些命令的完整运行状态；
- 本地 Python 环境和已下载的模型权重。

截至本说明生成时，当前 Codex 可见的项目都标记为 `local`，尚未把 VS Code 的 SSH 服务器识别成可交接的 Codex 远程主机。因此目前不能承诺仅登录账号即可无缝接续。

Codex 若同时识别了本机和远程主机，并且两边存在匹配的项目工作区，任务交接功能可以减少重新解释的工作；但它仍不能替代项目文件和数据在远程服务器上的实际存在。

## 三、取消压缩包后的可靠做法

本项目不再制作“启动压缩包”。改用以下三个层次保持连续性：

1. `AGENTS.md`：写明不可偏离的研究边界，远程 Codex 打开项目后可立即读取。
2. `PROJECT_STATE.md`：记录当前数据版本、有效结论、失败版本、实验门槛和下一条命令。
3. 项目目录同步：用 Git、`scp/rsync`、共享存储或 VS Code 远程文件传输，把代码、清单和必要数据放到服务器。账号登录不能代替这一层。

建议把“代码/配置/清单”和“大体积图像/权重”分开：

- Git 管理代码、Markdown、CSV 清单和 YAML 配置；
- 图像与权重放服务器数据盘，并在配置中写绝对路径或环境变量；
- 不把 `outputs`、缓存、模型权重和第三方仓库重复提交到 Git。

如果暂时不用 Git，可以从本机 PowerShell 复制项目（把占位符换成实际值）：

```powershell
scp -r "D:\kp\adaptive_keypoint_discovery_reboot" user@server:/remote/project/path/
```

数据量较大时优先使用可断点续传的 `rsync`，或在服务器直接挂载/访问原始数据盘。

## 四、远程接续顺序

1. 在 VS Code Remote-SSH 中打开服务器上的项目根目录。
2. 先读 `AGENTS.md` 和 `PROJECT_STATE.md`。
3. 运行 `python remote_gpu_check.py`，保存输出。
4. 按 `PROJECT_STATE.md` 中的“下一步”安装环境并执行训练前冒烟测试。
5. 训练产生的新结论回写 `PROJECT_STATE.md`，不要只留在聊天中。

这样做无法做到“零文件迁移”，但能取消反复打包和重新口述，并使换设备后的 Codex 以同一套工程事实继续工作。
