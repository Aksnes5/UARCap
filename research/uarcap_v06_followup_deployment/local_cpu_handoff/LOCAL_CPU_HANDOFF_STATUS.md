# 本地交接与 smoke 前剩余条件

| 交接状态 | 判定 | 依据 |
| --- | --- | --- |
| `LOCAL_CODE_READY` | **YES** | 冻结构建器复用，五个归档源核验，八份既有输入的本地复用包，冻结加新增 13 项 CPU 测试通过。 |
| `SERVER_INPUTS_VERIFIED` | **PARTIAL** | 复用 BN 对基础数据、权重、已有特征与顺序的资产核查。八份 prepared inputs 已在 Windows 核验，尚无 BN 暂存或目的端哈希收据。 |
| `GPU_SMOKE_READY` | **NO** | 欠 BN 输入落盘核验、可启动的精确环境收据与版本差异确认、image TSV 内容摘要及明确执行许可。 |
| `GPU_EXECUTION_APPROVED` | **NO** | 本轮未访问 BN、调用服务器 Agent、启动模型或执行 GPU smoke。 |

## 获准后最低成本补齐

1. 在可写项目盘的独立清洁路径中暂存 Windows 包（约 11.9 MB）。逐件核对收据中的 SHA，复用 BN 已有基本资产与 D 输入。Display 挂载保持只读。无需在 BN 重跑匹配、资产全审计或复制媒体。
2. 以历史 venv 解释器入口、实际 launch 的 shell 和 Python 参数生成小型环境元数据收据。核对 Torch/torchvision 原生包配对、图像和 tokenizer 栈及缓存路径，明确是否接受已记录的版本偏差。Vicuna/tokenizer 清单可复用 BN 模型回执中的文件证据，按 launcher 要求绑定 manifest 哈希。
3. 补齐当前 launcher 对 image TSV 的内容 SHA-256 要求。旧 BN 回执的 `VERIFIED` 来自大小、mtime、行/偏移和样本。它没有可填给 `frame_receipts.img.sha256` 的内容摘要。目标文件为 `/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/lj/datasets/MSRVTT-v2/test_16frm/frame_tsv/test_16frames_img_size224.img.tsv`，1,378,294,858 字节。先找可绑定到同一路径、大小和 mtime 的已有可信 SHA。没有时，获准后在低负载窗口于 BN 本地顺序读取一次，记录前后元数据，只返回 SHA 和收据。偏移文件的现有 SHA 不能代替它。保持现有内容校验门槛。
4. 明确批准具体 smoke 后才刷新 GPU 管理状态及原项目锁条件。按冻结顺序先 A 的真实 smoke，通过后才运行 A 全量。B smoke 和 B 全量仍受各自门槛约束。本地静态测试不验证真实模型 parity。

历史 D 臂强绑定缺失影响正式对照的复用。计划已保留对应新 arm，故无需为找齐历史收据推迟批准顺序。此次在 Windows 完成并停在交接边界。
