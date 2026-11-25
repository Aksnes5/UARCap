# UARCap Q2 完整离线工程实现报告（OFFLINE IMPLEMENTATION REPORT）

> **最终工程状态**：`OFFLINE_IMPLEMENTATION_READY_AWAITING_BN_ACCESS`  
> **基准科学协议**：`761dea2c51ab07c083b88805168ad1757db53d67`（`antigravity/uarcap-q2-pilot-a-protocol-v1.2-final`）  
> **工作模式**：Windows 轻薄本纯 CPU 离线完整工程实现、单元测试、端到端模拟、故障注入与 BN 一键部署准备；**未连接 BN、未启动 GPU、未下载完整视频/权重、未修改任何冻结历史结果**。

---

## 1. 已完整实现的工程模块总览

| 模块目录 | 核心实现文件 | 覆盖研究范围与关键工程保障 |
|---|---|---|
| `uarcap_q2/common/` | `io_utils.py`, `schemas.py`, `gates.py`, `gpu_lock.py` | 严格分离 `PilotStage` vs `ExpArm`；原子写入与 SHA-256；`FORMAL_B` 默认硬阻断授权门控；Pilot-to-Formal 排除防泄漏；协同共享 GPU 锁（复用 `/home/bn/videoufo_research/gpu.lock`，绝不杀异项目进程）；Token/凭证与受限资产扫描器 |
| `uarcap_q2/datasets/` | `dream1k.py`, `vidhalluc.py`, `video_decoder.py`, `asset_transfer.py` | DREAM-1K 5 域规范化、`UNKNOWN_OPAQUE` 严禁伪造 source ID、事前 `R_core_ref` 提取、固定种子分层抽样与永久排除清单；VidHalluc 11 位 YouTube ID 精确解析、同源 1 片去重、默认 `CANDIDATE_PENDING_VERIFICATION`；PyAV 16 帧均匀采样 + 历史重复首帧 bug 主动拦截 + EVCap `img.tsv/lineidx/yaml` 打包 + dHash；Route 1 断点下载与 Route 2 分卷校验重组 |
| `uarcap_q2/retrieval/` | `dedup.py`, `reranker.py`, `arm_builder.py` | VidHalluc 精确 YouTube ID 零重叠审计；DREAM-1K 两阶段保守近重复筛查（X-CLIP+dHash+词面 Jaccard，绝不宣称精确源级零重叠）；忠实复现冻结 `D` 共识重排（`.strip()` 分组 + 全字幕平均成对 ROUGE-L）；`ARM_A0` 严格空证据、`A1⊂A3⊂A7` 前缀一致与 E3 `BASE_A0`/`BASE_B3` 视觉锁定 |
| `uarcap_q2/interventions/` | `vidhalluc_e3.py`, `e3_bridge.py` | E3 `Sup/Con/Neu` 最小编辑构造、双 tokenizer 词长均衡校验、5 项人工适用性审核状态机（未经审批拒绝入池）；DREAM-1K `E3-Bridge` 同队列衔接（$<25$ 干净对自动禁用并输出主张收缩指令） |
| `uarcap_q2/generators/` | `base.py`, `videollama_adapter.py`, `modern_vlm_adapter.py`, `mock_generator.py` | 复用冻结 BN Video-LLaMA（锁定检查点 SHA-256 `a8b5af79...` 与 `BASE_ARGS`）；隔离现代 VLM 适配器（首选 `Qwen2.5-VL-7B-Instruct`，离线路径+revision+16 帧+8–18 词短字幕提示+显存遥测）；CPU Mock 与故障注入钩子；逐样本真实 `InputTrace` 审计 |
| `uarcap_q2/evaluators/` | `backends.py`, `reference_fact_eval.py`, `e3_semantic_eval.py`, `calibration.py` | 统一裁判后端（Local Transformers / OpenAI-compat / Mock）；Layer 3 UARCap 三值参考代理（9 类非法输出全部拦截，绝不默认判对/错）；原子命题漏抽错误审计；E3 双盲单目标独立二元评价器（允许 `(1,1)` 共存、防循环评价）；3×4 交叉矩阵、`A7` 差分偏置审计与 `PASS/FAIL/INCONCLUSIVE` 三态裁决 |
| `uarcap_q2/annotation/` | `package_builder.py`, `web_app.py`, `adjudicator.py` | 严格盲法任务打包（防泄密递归断言）；零依赖本地 HTML5 网页标注工具（Windows/Ubuntu 通用：静音视频播放+16 帧预览+Phase 1 核心事件锁定+一键四分类+增删改+盲态同视频规范命题复用+E3 双目标独立勾选+原子自动保存/续标/计时+CSV/JSONL 导出）；双人 $\kappa$/AC1 与双人合计总人时核算 |
| `uarcap_q2/statistics/` | `e1_historical.py`, `paired_stats.py`, `two_phase_correction.py`, `power_planner.py` | E1 回顾性跨参考选择与来源聚类自助法；视频/来源聚类配对置换检验、Holm–Bonferroni、精确 McNemar 与 E3 四象限归类、跨模型 DiD；两阶段 Horvitz–Thompson IPW 误差校正（含校准-验证隔离检验）；试点方差定容与 RTX A6000 吞吐外推 |
| `uarcap_q2/runner/` | `orchestrator.py`, `report_generator.py` | 统一多阶段运行器（单/批/臂/模型/限量/断点跳过/原子写/OOM 安全停机/失败收据）+ 自动生成 `PILOT_A1_EXECUTION_REPORT.md`、`PILOT_A1_INTEGRITY_REPORT.md`、`PILOT_A1_GO_NO_GO.md` |
| `configs/`, `deployment/`, `scripts/` | 4 份隔离环境配置 + 4 份阶段 YAML + 4 个 CLI + 8 个 Linux `.sh` 一键脚本 + 跨平台标注 UI 启动器 | 完整实现 `bootstrap_bn.sh`、`run_preflight.sh`、`run_pilot_a01.sh`、`run_pilot_a2.sh`、`run_pilot_a3.sh`、`run_formal_b.sh`、`status.sh`、`resume.sh`，全部通过 Unix LF 与 `bash -n` 语法校验 |

## 2. 仍严格等待试点结果冻结的科学参数

为严守科研纪律，以下参数已在 `configs/formal_b_candidate.yaml` 中参数化并锁定为 `PENDING_PILOT_FREEZE`，绝不提前伪造冻结：
- DREAM-1K 正式样本量（`600–800`，由 Pilot A2 实测方差定容）；
- `E2-Val` 独立验证集规模（基线 `50` 或自适应 `80` IPW，由 A2 实测错误率决定）；
- 唯一跨家族本地文本裁判与冻结提示词 SHA-256（由 A2 校准决定）；
- 唯一现代开源 VLM 与最终短字幕实测校验（由 A3 timed smoke 决定）；
- VidHalluc 正式规模（`150–180`）与 `E3-Bridge` 是否启用（$\ge 25$ 干净对门槛）；
- 两阶段 IPW 误差校正作为主分析还是敏感性分析（试点后裁决）。
