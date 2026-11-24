# Windows 最终交接状态

| 项 | 状态 |
| --- | --- |
| GitHub 分支 | `codex/uarcap-v06-local-cpu-handoff-20261124` |
| 原本地交接提交 | `56c908034c0125604e6089a0e6db8d0232c3ff10` 已推送并核对 |
| 固定 ZIP 与验证文件 | 输入传输提交 `bc737176ea700492402be63480a3c6e76f05be2e` |
| `TRANSFER_VERIFIED` | **YES**，远端完整 ZIP 及包内八份输入与 Windows 原始字节匹配 |
| `LOCAL_CODE_READY` | **YES**，现有输入已通过接收检查，无须重建 |
| `SERVER_INPUTS_VERIFIED` | **PARTIAL**，基本资产沿用 BN 原回执，BN 尚需实际下载、解包和验证 |
| `GPU_SMOKE_READY` | **NO**，现场先完成接收与剩余只读核查，形成正式部署 JSON |
| `GPU_EXECUTION_APPROVED` | **NO** |
| 历史结果严格复用 | **0**，六组冻结矩阵继续有效 |

Windows 工作已完成，可把 `BN_CODEX_RECEIVE_AND_EXECUTE_PROMPT.md` 交给 BN 对方完成前置核查。当前还缺 BN 的目的端传输收据、运行环境和版本偏差确认、image TSV 可信内容 SHA-256、checkpoint/缓存与隔离输出目录绑定，以及当时的磁盘和 GPU 锁状态。执行前先通过这些门槛，再申请 A 的实际 smoke 授权。A 正式运行、B smoke 和 B 正式运行各有单独授权及先决条件。

输入包 **2,861,489 字节**，SHA-256 `93289136546fc2d96d2f457cc9dec57c50fde1b2c517f735f23df0a1dfc63128`。Windows 本轮仅检查、封装和同步既有输入及交接文档。未连接 BN，未启动模型、CUDA、GPU smoke 或统计运行，论文、冻结协议和数据内容保留。 [固定输入包](https://github.com/Aksnes5/UARCap/blob/bc737176ea700492402be63480a3c6e76f05be2e/research/uarcap_v06_followup_deployment/local_cpu_handoff/transfer/UARCap_V06_FOLLOWUP_BN_INPUTS.zip)
