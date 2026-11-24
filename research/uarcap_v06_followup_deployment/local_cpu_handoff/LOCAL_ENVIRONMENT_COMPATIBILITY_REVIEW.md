# UARCap v0.6 本地运行环境审查

审查以 BN 回执提交 `d184360cb1cda832dbaa2d6ed9f46fb4aaa186e6`、冻结补实验提交 `ee7d3b9876dea94a1ee9eeb2626b92102c85d458` 和推理源码归档 `52b5ed3a69e4b84a8e16ed747f141eefb5b1e2ec` 为准。Windows 阅读了八份回执及 `run_readonly_audit.py`。BN 版本来自其 2026-11-24 16:10 回执。原审计以 `-I` 查询版本元数据和模块路径，没有导入 PyTorch。此次没有重新运行该审计。

| 组件 | 归档 → BN 回执 | 静态结论 |
| --- | --- | --- |
| Python | 3.9.18 → 3.9.13 | 补丁版本差异已证实，未发现由此改变字幕结果的证据。实际代码阻塞来自整个 3.9 系列缺少 `zip(strict=True)`。冻结输入构造、独立 launcher 的视觉查找以及 B 的 Windows 评分路径均会用到。构建与评分留在 Python ≥3.10 的 Windows。BN 使用已有部署适配层的 `vision_py39` 明确校验长度，再调用普通 `zip`，保留严格配对语义。[Python 3.10 官方文档](https://docs.python.org/3.10/library/functions.html#zip) |
| PyTorch / CUDA | 2.1.0 → 2.1.2+cu121 | 确认版本差异。2.1.2 上游包含部署 hook 所用的 `register_forward_pre_hook(..., with_kwargs=True)`。归档同时记录 CUDA 11.3 Conda 工具包和 CUDA 12.1 Python 依赖，仅靠它无法确定历史原生库解析。BN 的 wheel 标签也无法证明实际驱动、动态库、attention/AMP/RNG 内核或 fp16 结果相同。未观察到版本改变输出的证据。[PyTorch 2.1.2 模块源码](https://github.com/pytorch/pytorch/blob/v2.1.2/torch/nn/modules/module.py#L1234) |
| torchvision 与图像栈 | 归档 torchvision 0.16.0、NumPy 1.24.3、OpenCV 4.7.0.72、Pillow 10.0.1、PyAV 14.0.1 | BN 的有效版本与原生库仍 `UNVERIFIED`。官方 cu121 wheel 将 torch 2.1.2 与 torchvision 0.16.2 配对，值得先核实现有配对。图像 TSV 使用 OpenCV 解码和后续图像转换，版本差异可能影响帧张量。PyAV 的在线解码绕开，但 `video_ops.py` 在加载时导入 `av`，缺包仍能阻断进入 TSV 路径。[PyTorch 历史版本配对](https://pytorch.org/get-started/previous-versions/#v212) |
| Transformers | 4.28.0 → 4.28.0 | 版本一致，BN 从 Display venv 解析包。冻结代码用本地 `LlamaTokenizer`（`use_fast=False`）和仓库内 LLaMA 类。需核实有效的 SentencePiece、tokenizers、HF Hub 依赖及本地缓存解析。上游 4.28.0 要求 tokenizers ≥0.11.1、<0.14，排除 0.11.3。版本号相同不能确定 wheel 内容或 prompt token 相同。[Transformers 4.28.0 依赖源码](https://github.com/huggingface/transformers/blob/v4.28.0/setup.py) |
| timm / 本地视觉缓存 | 0.6.13 → 0.6.13 | 固定版本支持 `timm.models.hub` 缓存 API。现有适配层检查 EVA/BLIP2 文件存在后才调用原缓存 API。部署前核对解析到的缓存根和文件可读性。冻结 video Q-Former 以 BERT config 构建网络，并加载已有 BLIP2/Video-LLaMA checkpoint。回执的 BERT 缺单独权重文件未构成已证实的缺口，实际离线初始化仍待验证。[timm 0.6.13 缓存源码](https://github.com/huggingface/pytorch-image-models/blob/v0.6.13/timm/models/hub.py) |

## venv 与 Conda 的边界

历史 runner 的解释器入口是 Display 的 `repro_evcap/venv/bin/python`，该链接指向 `/home/bn/anaconda3/envs/TTP/bin/python3.9`。`pyvenv.cfg` 开启 `include-system-site-packages`，因此 venv 与 TTP 的包共同参与解析。回执查到 Torch 从 TTP 而 Transformers 从 Display venv 加载。直接调用链接目标会改变 venv 识别语义。额外激活 Conda、用户 site、`PYTHONPATH`、`.pth` 与动态库路径都可能改变另一个 shell 的解析。当前 `-I` 结果只覆盖该次探测环境，不能充当历史进程或未来普通 launcher 的收据。

最低成本处理是保留已存在的解释器入口和缓存，在未来实际 launcher 的同一工作目录、相同 Python 启动标志与 shell 环境下，使用 `probe_runtime_metadata.py` 打印一份标准库收据。脚本记录解释器入口和真实路径、sys.path、有限环境变量、关键包的版本/定位及小型文件指纹。它没有模型或 GPU 依赖，已在 Windows 上运行。若拟使用隔离标志，先对比该模式与拟实际运行的模式。只在输出可写路径保存收据，并绑定至当前 `python_environment_sha256`。包元数据和源码入口哈希仍不能证明原生库已加载成功。

**已证实的影响**是直接用 BN 的 Python 3.9 执行冻结的 `zip(strict=True)` 路径会失败。**尚未证实的影响**是两个补丁版本、混合环境、tokenizer 或图像栈改变历史或新字幕结果。也缺少对 D1/D3/D7 的旧每进程依赖快照。由负责人确认接受现有运行环境差异后，才进入获准的真实 smoke。没有根据重装或更换现有环境。
