# 全管线 Windows 端到端 CPU 测试与重放报告（V2）

- **测试日期**: 2026-11-26
- **平台与运行时**: Windows, Python 3.11.9, pytest 9.1.1, numpy 2.4.6, scipy 1.17.1, matplotlib 3.11.2
- **代码分支**: `antigravity/uarcap-q2-laptop-final-development-v2-20261126`
- **V2 队列版本与 SHA-256**: `laptop-20261126-v2` / `ef7c275a54e154d00fa65325108fa5fb0eed3dda6ac47a7d8e8d878aa45ee576`

> **严格边界声明**：本报告严格区分**真实归档样本重放**、**CPU 确定性夹具测试**、**Mock/Stub 测试**与**待 BN 实机验证项**。Windows CPU 测试全部通过**不等于** BN 真实 GPU/vLLM/Video-LLaMA 推理验收通过，亦**不等于**任何真实人工标注或确认性统计门禁已通过。

## 1. 全量自动化测试与静态审计结果

| 检查项 | 命令 | 结果 | 说明 |
| --- | --- | --- | --- |
| 全量 Pytest 测试套件 | `python -m pytest -q -rs` | **113 passed, 0 failed, 0 skipped (37.77s)** | 含原有 93 项 + V2 闭环 14 项 (`test_15_v2_closures.py`) + V2 最终审计与失败隔离 6 项 (`test_16_v2_final_audit.py`) |
| AST 未绑定符号扫描 | 静态扫描全部 `.py` | **0 未绑定符号** | 修复了接管审查发现的 `scripts/e3_reserve_pool.py` 缺失 `import re` 缺陷 |
| 字节码编译检查 | `python -m compileall -q scripts uarcap_q2 deployment bn_carebench_pilot_a01/overnight_20261126/tasks` | **退出码 0** | 全模块语法与导入结构完整 |
| V2 队列全量审计 | `python scripts/audit_q2_queue.py --queue laptop_final_handoff_20261126/REMAINING_GPU_TASK_QUEUE.json --out <audit.json>` | **退出码 0，0 错误** | 111 任务（60 GPU 包装 / 51 CPU 包装；52 `videollama2` / 56 `vllm` / 3 `repro_evcap`），29 项真实 `argparse --help` 检查全部通过，0 指纹漂移，0 循环依赖，0 输出路径冲突 |

## 2. 四类验证层级明细（严格区分真实归档重放 / CPU 夹具 / Mock / 待 BN 实机）

### A. 真实归档数据 CPU 重放（无 Mock，直接读取已提交或本地真实归档文件）

| 管线 / 模块 | 实际输入 | 验证结果 | 备注 |
| --- | --- | --- | --- |
| P1-1 / P1-2 / P1-3 敏感性设计生成 (`q2_controlled_sensitivity.py plan`) | 归档 `LENGTH_CONTROL_PLAN.json` + `VISUAL_SAMPLING_PLAN.json` + 真实 360 条归档 Qwen 短字幕 (`formal/qwen/predictions.jsonl`) | **PASS**（0.48s） | 实测短字幕四分位目标词数范围 **`[13, 15]`**（`n=360`），120 长度查询 / 40 Qwen 模态查询 / 60 嵌套帧数查询确定性冻结 |
| P0-4 真实 Top-7 视觉签名分母核对 (`generate_visual_signatures.py`) | 归档 `formal/formal_retrieval.jsonl`（600 查询 × Top-7） | **PASS**（0.08s） | 实测 **600** 查询、**4,200** 配对、**2,622** 个去重 Top-7 代表 clip |
| P0-4 来源重叠真实核验 (`source_and_visual_risk.py sources`) | 本地官方 `clips_full.tsv` + `E3_PILOT_EXCLUSION.json` + `E3_FORMAL_APPROVED.json` + `msrvtt_mapped_source_ids.txt` | **PASS**（7.88s） | VideoUFO 源 **345,437**，旧正式 E3 源 **159**，识别并排除 **2** 条历史重叠，CaReBench 诚实标记 `NOT_JOINABLE` |
| 历史指标桥接与预算汇总（既有重放产物） | `formal/` 真实 3,600 条预测与评价 | **PASS** | 归档 7,200 行修复指标表（R-Full / R-Short）与 0/1/7 真实预算报告均保持完整 |
| 统一总汇总快照 (`q2_release_report.py`) | 111 任务 V2 队列 + 归档证据 | **PASS**（1.11s） | 输出 `FINAL_EXPERIMENT_MATRIX.json`、`ARCHIVE_MANIFEST.json`、`MANUSCRIPT_EVIDENCE.md`，全部确认性门禁严格保持 `false` |
| 确认性统计硬门禁 (`run_final_analysis.py --confirmatory`) | 真实仓库状态（无人类标签） | **PASS（按设计硬拒）** | 退出码 **3**，拒绝在无人工校准收据下执行确认性统计 |
| V2 队列与产物契约一致性 (`audit_q2_queue.py` + `EXPECTED_OUTPUT_ARTIFACTS.json`) | 真实 V2 队列与全部本地脚本 | **PASS**（13.87s） | 111 任务全部对齐，29 项真实 CLI `--help` 核验通过，`errors == []` |

### B. CPU 确定性合成夹具测试（真实算法/真实视频编码/真实 HTTP 服务，小规模合成输入）

| 管线 / 模块 | 测试覆盖 | 结果 |
| --- | --- | --- |
| P0-1 E3 合法储备池与续筛 (`e3_reserve_pool.py`) | 6 类来源排除（Pilot 30 / Formal 180 / VideoUFO / Historical / Debug / Duplicate）、未审批硬拒、不可降 150 门槛、配置一致校验、`continue_screen()` 候选不足隔断（exit 3）与查询 ID 正则防注入 | **PASS** |
| P0-2 统一盲法人工门户与双人回环 (`q2_human_portal.py` + `web_app.py` + `launch_annotation_ui.py`) | FULL_VIDEO / GENERATOR_FRAMES / CAPTION_EXPRESSION 作用域隔离、真实本地 HTTP 服务回环访问令牌鉴权（拒绝缺失/错误 Cookie 403）、`Range` 206 视频定位、同人冒充双审拒绝、完整导入与仲裁 | **PASS** |
| P0-4 真实视频解码、压缩成员单件读取与近重复检测 (`generate_visual_signatures.py` + `source_and_visual_risk.py` + `q2_visual_risk_sensitivity.py`) | `av` 生成真实 MP4 与 `.zip` 成员单件读取、目录穿越 (`../escape.mp4`) 拒绝、16 帧 dHash/pHash/center-crop pHash 缓存复用、时序偏移近重复、UNVERIFIED 隔离与父视频聚类敏感性排除 | **PASS** |
| P1-3 帧数 × 预算交互 DiD (`q2_controlled_sensitivity.py`) | 16/32/64 帧 A7−A0 双重差分、密帧输入漂移硬拒、单评价器 `llama_n_contra = UNAVAILABLE` 诚实保留 | **PASS** |
| P1-4 溯源/相关性/历史模块报告与释放矩阵 (`q2_module_report.py` + `q2_release_report.py`) | 单评价器与双评价器独立输出、部分/失败产物独立记录 | **PASS** |

### C. Mock / Stub 边界测试（替代不可在轻薄本加载的大模型或 Java SPICE）

| 管线 / 模块 | Stub 范围 | 验证的真实生产逻辑 | 结果 |
| --- | --- | --- | --- |
| P0-3 已知答案评价器压力测试 (`evaluator_stress.py`) | 用确定性 `CPUTestFixture` 模拟模型文本输出 | 196 条病例、真实生产 `NLI_SYS` / `EXTRACT_SYS` 提示、`parse_and_validate_verdicts` / `parse_extracted_claims`、断点续跑幂等性、解析失败计数、共享错误与混淆矩阵 | **PASS** |
| P1-2 Qwen 外部视觉帧模态因子 (`qwen3_vllm_adapter.py`) | 用 `SimpleNamespace` 模拟 vLLM tokenizer/output | 16 查询帧 + 8 Top-1 外部帧消息角色边界、`<|vision_start|>`/`<|image_pad|>` token 消费硬验证（伪造缺失块时报错） | **PASS** |
| SPICE 异常隔离 (`e2_metric_bridge.py`) | 注入 `pycocoevalcap` 模拟计分器与异常 `Spice_fn` | 验证 SPICE 报错时 BLEU-1/BLEU-4/METEOR/ROUGE-L/CIDEr 仍正常产出且记录 `SPICE_status: BLOCKED: ...`；SPICE 正常返回字典时正确解析 `All.f` | **PASS** |

## 3. 7 类失败隔离场景专项验证结果（`tests/test_16_v2_final_audit.py`）

| # | 失败场景 | 验证结果 |
| --- | --- | --- |
| 1 | E3 正式候选不足 150 且无储备审批 | `e3_reserve_continue` 退出码 3，写入 `E3_INSUFFICIENT_VALID_CANDIDATES`；A3/A5 预算、压力测试、视觉签名与总快照均无 E3 依赖，完全不受阻断 |
| 2 | 仅单模型完成（Video-LLaMA 或 Qwen） | `budget_metrics_{m}` 与 `budget_report_{m}` 仅依赖该模型自身 Gemma 评价，可独立产出 |
| 3 | 单评价器（Gemma）成功、第二评价器（Llama）失败/缺失 | E3 产出 `PARTIAL_JUDGE_COVERAGE`（绝不伪装为 `DUAL_JUDGE_COMPLETE`）；可控敏感性与模块报告产出 `GEMMA_ONLY` 且 `llama_n_contra = UNAVAILABLE` |
| 4 | SPICE 跳过或报错 | 队列默认 `--skip-spice`，且未跳过时异常被捕获写入 `SPICE_status`，其余 4 项传统指标正常完成 |
| 5 | 64 帧 Qwen 失败（如 OOM） | `report_visual32_gemma`、`report_visual32_dual`、`report_length_*`、`report_factorial_*`、`budget_report` 与 `final_snapshot` 的传递依赖均不包含任何 `visual64` 任务 |
| 6 | 人工核验未完成 | 探索性总汇总与报告正常产出并诚实标记 `MACHINE_ASSISTED_UNVALIDATED` / `POST_HOC_MACHINE_ASSISTED` |
| 7 | 无人工收据请求确认性统计 | `run_final_analysis.py --confirmatory` 硬拒（退出码 3）；`FINAL_EXPERIMENT_MATRIX.json` 全部科学门禁保持 `false` |

## 4. 明确标注：仍待 BN 实机验证与真实人工验收的事项

以下事项在 Windows 轻薄本**无法也未被声称已完成**，必须在 BN 服务器实机执行并按回执验收：

1. **BN 真实环境预检**：`install_q2_v2_bn.sh <SHA> --dry-run` 验证 `/home/bn`、`/media/bn` 模型目录、VideoUFO 媒体定位文件 (`videoufo_clip_locations.jsonl`)、`vidhalluc_debug_source_ids.txt` 与三个 Python 环境。
2. **真实 GPU 模型推理与试跑**：Video-LLaMA EVCap 冻结模型、Qwen3.8-27B-FP8（含 16/32/64 帧与 16+8 外部帧实机显存/processor 验证）、Gemma 3 27B 与 Llama 3.3 70B 真实评价器推理及压力测试。
3. **真实视频媒体解码与视觉签名**：600 CaReBench 视频 + 2,622 VideoUFO Top-7 真实代表 clip 的实际读取与签名生成。
4. **真实双人独立人工盲法核验与仲裁**：Pilot A2、Pilot A3、60 视频内部概率样本、E3/E3-Bridge 全视频与 16 帧真值、E3 表达抽检及独立签收。
