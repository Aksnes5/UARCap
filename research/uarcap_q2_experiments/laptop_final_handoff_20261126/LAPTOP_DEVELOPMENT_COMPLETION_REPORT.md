# UARCap Q2 V2 Windows 交付完成报告

- **交付分支**: `antigravity/uarcap-q2-laptop-final-development-v2-20261126`
- **V1 起点提交**: `1693dbb65ed43d962bcf8915fa390d46d8d57761`
- **BN 交接基线**: `2027e4267dce105d47a60c8f9eedcef5c9099bda`
- **冻结 v0.8 论文基线**: `6421b6b9acd395420ce29dd630b8f1c5b15f84ec`
- **当前状态**: `LAPTOP_ENGINEERING_COMPLETE_BN_ACCEPTANCE_PENDING`

V2 使用独立分支、独立部署目录 (`/home/bn/uarcap_q2_runtime/deploy/laptop-20261126-v2`) 与独立输出目录 (`/home/bn/uarcap_q2_runtime/outputs/laptop-20261126-v2`)。旧代码、历史字幕、原 v0.8 论文与冻结主检验均完整保留。

## 1. V2 本地工程闭环摘要

| 本地交付模块 | 验证结果与关键产物 |
| --- | --- |
| Codex 中断恢复与缺陷修复 | 39 个未提交文件已独立只读备份至 `D:\Antigravity\uarcap_codex_v2_recovery_20261126\`；修复 `scripts/e3_reserve_pool.py` 缺失 `import re`、`scripts/e2_metric_bridge.py` `Spice_fn()` 调用、`deployment/q2_task_entry.py` 锁打开模式及 12 项队列指纹漂移。详见 [恢复审计报告](CODEX_INTERRUPTED_WORK_RECOVERY_REPORT.md)。 |
| V2 111 任务队列与全量审计 | `REMAINING_GPU_TASK_QUEUE.json`（111 任务：60 GPU 包装 + 51 CPU 包装；52 `videollama2` / 56 `vllm` / 3 `repro_evcap`）与 `EXPECTED_OUTPUT_ARTIFACTS.json` 一一对齐；`audit_q2_queue.py` 通过 29 项真实 CLI `--help` 检查、DAG 无环与零指纹漂移验收（0 错误）。 |
| P0-1 E3 候选储备与失败隔离 | 6 类源级排除、独立种子 20261126、同配双端筛查、预生成审批上限、150 门槛不可放宽；不足时明确输出 `E3_INSUFFICIENT_VALID_CANDIDATES`（exit 3）且不阻断 A3/A5 与其他模块。历史无效 Neu 保留在历史归档中。 |
| P0-2 统一盲法人工门户 | `q2_human_portal.py` + `web_app.py` + `launch_annotation_ui.py` 统一加载 60 视频概率包、Pilot A2、Pilot A3（排除旧无效 Neu）、E3/E3-Bridge 全视频与 16 帧真值及 E3 表达盲审；强制回环隔离访问令牌、双人独立签收与独立仲裁回灌。 |
| P0-3 已知答案评价器压力测试 | `evaluator_stress.py` 提供 196 项受控文本病例（14 类：182 NLI + 14 原子拆并），复用生产提示与解析器，CPU 夹具全链路通过；真实 Gemma/Llama 推理待 BN 执行。 |
| P0-4 实际视频近重复签名与敏感性 | 真实归档 Top-7 核验为 600 查询 / 4,200 配对 / 2,622 去重代表 clip；`generate_visual_signatures.py` 支持不批量解压单件 zip/tar 读取、安全路径校验与 16 帧 dHash/pHash/center-crop pHash 缓存；`q2_visual_risk_sensitivity.py` 执行预定义父聚类敏感性排除。 |
| P1-1 Qwen 统一长度控制 | 归档 120×3 条短字幕实测 IQR 为 **13–15 词**，三臂共享统一提示目标且不做后处理截断；如检测到 V1 详细提示已启动或存在产物则硬阻断替换。 |
| P1-2 & P1-3 Qwen 模态因子与帧数×预算交互 | 固定 40 查询 `T0F0/T1F0/T0F1/T1F1` 因子（16 查询帧 + F1 加 8 Top-1 外部帧，带 vLLM processor token 消费硬验证）；150 查询 A7 + 嵌套前 60 查询 A0 的 16/32/64 帧 DiD 交互；64 帧失败与 16/32 帧严格隔离。 |
| P1-4 补充模块闭环与统一总汇报 | 自然溯源、Top-7 相关性（4,200 字幕 + 420 Llama 抽检）、MSR-VTT 历史反向桥接（60 视频 180 字幕）全部接入单/双评价器独立报告、矢量/位图绘图及 `q2_release_report.py` 归档清单。 |
| Windows CPU 全量测试 | `python -m pytest -q -rs`：**113 passed, 0 failed, 0 skipped (37.77s)**；AST 未绑定符号 **0**；`compileall` 退出码 **0**；7 项真实归档重放与 7 类失败隔离场景全部通过。 |

## 2. 仍待 BN 实机验收与真实人工核验的事项

当前 BN GPU/vLLM/Video-LLaMA 实机运行、可用视频媒体、评价器真实标签和全部双人人工门禁均处于 `PENDING`。部署前需操作者在 BN 侧提供/确认：VideoUFO 媒体索引或位置清单 (`videoufo_clip_locations.jsonl`)、VidHalluc 调试排除清单 (`vidhalluc_debug_source_ids.txt`) 及绑定本次提交完整 SHA 的真实授权文件。若 BN 上 V1 已有执行历史，须先对齐现有回执与产物，不得覆盖或丢弃任何已完成结果。Windows 轻薄本未执行任何 BN GPU 任务，亦未伪造任何人工标注。

## 3. 核心交付文件索引

- [恢复审计报告](CODEX_INTERRUPTED_WORK_RECOVERY_REPORT.md) · [V2 完成度矩阵](V2_TASK_COMPLETION_MATRIX.json) · [端到端 CPU 测试报告](ALL_PIPELINES_END_TO_END_CPU_TEST_REPORT.md) · [CPU 测试回执](CPU_TEST_RECEIPT.json)
- [BN 部署与执行提示词](BN_DEPLOYMENT_AND_EXECUTION_PROMPT.md) · [BN 最终验收清单](BN_FINAL_DEPLOYMENT_CHECKLIST.md) · [BN 资源估算](BN_RESOURCE_ESTIMATE.md)
- [V2 完整任务队列 (111 项)](REMAINING_GPU_TASK_QUEUE.json) · [预期输出契约 (111 项)](EXPECTED_OUTPUT_ARTIFACTS.json) · [剩余科学门禁](REMAINING_SCIENTIFIC_GATES.json)
- [E3 储备协议](E3_RESERVE_POOL_PROTOCOL.md) · [评价器压力测试规范](EVALUATOR_STRESS_TEST_SPEC.md) · [视觉签名手册](VISUAL_SIGNATURE_GENERATION_RUNBOOK.md) · [人工核验手册](HUMAN_VALIDATION_RUNBOOK.md)
- [论文最终结构骨架](FINAL_PAPER_MANUSCRIPT_SCAFFOLD.md) · [论文证据链](Q2_PAPER_EVIDENCE_CHAIN.md) · [审稿人攻击矩阵](Q2_REVIEWER_ATTACK_MATRIX.md)
