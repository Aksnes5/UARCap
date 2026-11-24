# A0 / B 输入构建与本地回放

**本轮已交付 Windows 本地输入包。** 已找到上轮 CPU 回放的八份产物。本轮通过冻结指纹重新核验内容，逐文件复制并对比字节哈希。没有重新枚举 B 的 full tail assignment。包在 `D:\UARCap\uarcap_v06_prepared_inputs_local`，`PREPARED_INPUT_REPLAY_RECEIPT.json` 记录每个文件的原始与规范化 SHA-256。同一收据已纳入本分支。其余大体积归档和模型文件不进入本次提交。

## 数据来源

从当前 Windows checkout 本地 Git 对象读取以下五个固定 blob，使用 `git --no-lazy-fetch`，总计 18,820,077 字节。BN 回执对应文件的 SHA-256 与冻结值一致。父目录为 BN `/home/bn/videoufo_research/`。

| 相对路径 | 字节 |
| --- | ---: |
| `external/external_test_top20.json` | 13,124,819 |
| `external/external_test_d7_cap.json` | 3,311,809 |
| `external/external_test_d7_sims.json` | 2,156,249 |
| `manifests/msrvtt_test2990.tsv` | 119,600 |
| `manifests/msrvtt_untouched2690.tsv` | 107,600 |

构建只使用候选及名单。无需访问帧 TSV、视频、参考描述、预测、特征张量或模型权重，也无需在 BN 重新生成任何数据。

## 冻结构造

- A0 每个 query 的 CAP 为 `[[],[],[]]`、SIM 为 `[[],[]]`，覆盖全部 2,990 条，保留运行时原有融合接口。
- B 由 `prepare_inputs.py` 原实现构造。用 D 的 caption / clip list / raw SIM 配准前三条。按冻结 X-CLIP bands、来源多样性、词数、head overlap、tail redundancy 及固定排序规则匹配尾部。每条附加文本的视觉关联为空，三个 B arm 共用前三条的 SIM 和同一视觉合同。HQ/LQ 指 X-CLIP relevance band。
- 完整运行顺序 2,990 条。A 正式 2,690 条。B 固定合格 2,618 条（97.32%），规范化哈希 `450770ffd92520a8b474186c75c9e08cf7fbe3d11949f227a987e793cef03972`。72 条未达匹配条件的评测查询与 300 条 pilot 在 B 都保留 Text3。
- B 三组共享 7,035 个唯一视觉 ID，Windows 本地逐查询核实它们是 A7 的 15,213 个 ID 的子集。BN 对 A7 文件存在性的回执由此覆盖 B 的路径集合。它未做每个文件的可读性/解序列化验证。既有 launcher 将逐臂检查所需路径并记录实际加载。

## 使用与测试

```text
python -B research/uarcap_v06_followup_deployment/local_cpu_handoff/build_prepared_inputs.py --repo . --out <new-local-dir> --reuse-prepared <previous-local-prepared-dir>
```

需要从头恢复缺失本地输入时，在 Windows 的 Python ≥3.10 下省略 `--reuse-prepared`。脚本调用原冻结构建函数而非另一套匹配实现，验证完整统计摘要和八个对象的冻结指纹后才写输出目录。收据最后写入，作为完成标记。两种模式都拒绝覆盖现有目录。传输时保留 JSON 原始字节及收据，从而让未来预测收据可以用同一 raw SHA 核对。

本轮通过冻结测试 8 项、新增测试 5 项，包含拒绝错误源哈希、错误输出哈希和虽具有同一规范化对象哈希却改变 query 顺序的文件。八份输入合计 11,326,098 字节，逐臂视觉 ID 清单合计 551,501 字节。未在 BN 暂存。待获准部署时，只需上传这个本地包与小型收据到项目可写隔离目录。
