# BN 部署交接清单

传输提交 `bc737176ea700492402be63480a3c6e76f05be2e`，输入 ZIP `UARCap_V06_FOLLOWUP_BN_INPUTS.zip`，2,861,489 字节，SHA-256 `93289136546fc2d96d2f457cc9dec57c50fde1b2c517f735f23df0a1dfc63128`。使用同一提交上的 `GITHUB_INPUT_TRANSFER_MANIFEST.json` 与 `receive_bn_inputs.py`。 [固定输入包](https://github.com/Aksnes5/UARCap/blob/bc737176ea700492402be63480a3c6e76f05be2e/research/uarcap_v06_followup_deployment/local_cpu_handoff/transfer/UARCap_V06_FOLLOWUP_BN_INPUTS.zip)

| 阶段 | BN 或 Windows 需保留的证据 | 停止条件 |
| --- | --- | --- |
| BN 接收 | Git/HTTPS 取包、全包 SHA、八个成员的 SHA、2,990 / 2,690 / 2,618 队列顺序、独立路径、接收回执 | 下载、成员、顺序或哈希不匹配 |
| BN 只读增量核验 | 采用旧资产证据，补实际 venv 和库版本、checkpoint / Vicuna / 视觉缓存绑定、image TSV 内容 SHA、offset 哈希、GPU 管理状态、共享锁、可写盘空间 | 环境或可信摘要不完整，原资产不可读，使用路径与回执不符 |
| BN 生成配置 | 冻结源码、隔离根、run_id、独立 checkpoint 和输出槽、6 条 dry-run 的 argv、artifact SHA、设备 UUID、offline cache 和环境收据 | 配置用了假设路径或指向原输出目录 |
| A smoke | 单独授权，A0→A1→A7 各 8 条，Windows 验证真实零证据、帧/查询/module mode/RNG、dtype/mask 与日志 | 未授权、已有 GPU 作业、每 query gate 失败 |
| A full | 单独正式授权，A smoke gate，A0→A1→A7 三份完整 receipt、输出、trace、Windows A full gate | gate 不匹配、未授权或样本/引用不完整 |
| B smoke | A full gate、单独授权，B3→B7HQ→B7LQ 各 8 条，逐 query 固定视觉完全一致 | 任何 clip、tensor、权重、K/V、门控或视觉投影偏差 |
| B full | 单独正式授权，全部 B smoke 与 A full gate，三臂完整证据交 Windows | gate 不匹配、未授权或逐查询 parity 失败 |
| Windows 统计 | 仅评真实完整输出并沿用冻结 PTB CIDEr、source cluster、10,000 sign-flip/bootstrap、Holm 与固定比较 | 记录或冻结 cohort 不一致 |

独立提交包的公开许可只覆盖此包实际包含的字段和元数据。后续模型输出和日志另走已批准通道。任何运行中断保留已完成臂与失败 attempt 状态。不得覆盖或重新生成冻结材料。
