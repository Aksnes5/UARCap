# UARCap Q2 离线工程最终交付与真实执行链整合报告

- **当前工程状态**：**`OFFLINE_CODE_COMPLETE_AWAITING_BN_DEPLOYMENT`**
- **交付分支**：`antigravity/uarcap-q2-final-offline-handoff`
- **继承基线**：完整保留 `a00f8f27` 与 `bde76688` 及所有历史冻结提交、v0.8 论文与科研协议。
- **本轮验证范围**：Windows 轻薄本离线静态/CPU 单元/端到端/真实入口契约/历史黄金回归测试（**59 passed, 0 failed, 0 skipped**）；未连接 BN、未执行 GPU 推理、未下载大模型权重或完整视频压缩包。

## 本轮最终收尾补齐内容摘要

1. **模型选型修正与既有 BN 资产复用（取消默认下载 `Qwen2.5-VL-7B-Instruct`）**
   - 新增 [`uarcap_q2/generators/qwen3_vllm_adapter.py`](uarcap_q2/generators/qwen3_vllm_adapter.py) 与 [`uarcap_q2/generators/qwen3_vllm_worker.py`](uarcap_q2/generators/qwen3_vllm_worker.py)：面向既有 **`Qwen/Qwen3.8-27B-FP8`**（`/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Jnh_rag/models/Qwen3.8-27B-FP8`，备用既有 `Qwen/Qwen3.5-27B`）的 vLLM 0.29.0 多模态视频推理适配器，支持 16 帧真实输入、短字幕对齐、检索文本注入（`PROMPT_TEXT_ONLY`）、显式关闭/控制 thinking（默认 `enable_thinking=False` + `<think>` 剥离审计）、本地分片完整性检查、离线 `vllm_offline` / 本地 `vllm_openai_http` / 跨环境 `vllm_subprocess` 三种执行模式及完整 `InputTrace`。
   - 跨家族文本裁判默认复用既有 BN 模型（首选 **`Mistral Small 3.2 24B`**，备用 `Gemma 3 27B`、`Llama 3.3 70B W4A16`、`gpt-oss-20b`、`Phi-4 14B`、`Llama 3.1 8B`），新增 `LocalVLLMJudgeBackend` 与 `vllm_subprocess` 跨环境调用入口。
   - 同步更新 `configs/bn_paths.example.yaml`、`configs/pilot_a23_config.yaml`、`configs/formal_b_candidate.yaml`、`DATA_AND_MODEL_ASSET_MANIFEST.json`、`BN_DEPLOYMENT_MANIFEST.json` 与 `deployment/run_stage_cli.py`。
2. **A3 人工评价完整闭环（4.1）**
   - `run_pilot_a3_pipeline()` 生成双模型 E3 字幕并完成盲法评测后，自动导出 E3 匿名双目标标注包与现代 VLM DREAM-1K 匿名标注包，并**强制停于 `HUMAN_VALIDATION_PENDING`（退出码 10）**，绝不因生成成功而提前通过。
   - 重入 A3 时自动复用既有评测与盲法映射，调用 `audit_e3_evaluator_agreement()` 与 `audit_cross_model_dream1k_bias()` 完成独立 `Y_true` / `Y_conflict` 一致性、`Con vs Neu` 差分偏置与跨模型 DREAM-1K 偏置核验；`gates.py` 同步要求 Formal B 校验 A3 `e3_human_validation_completed == True`。
3. **A2 人工核验严格对账与盲法包复用（4.2）**
   - 废除 `zip(auto_r["claims"], h_labels)` 位置截断对齐，改为 `reconcile_a2_human_and_auto()` 基于稳定 `claim_id` 与 `package_sha256` 严格对账，拒绝缺失/重复 `claim_id` 与版本失配，原生支持遗漏命题补录（`added_claims`）与命题拆分/合并修订（`claim_revisions`）。
   - 重入 A2 导入人工真值时自动校验 `a1_predictions_sha256` 并复用既有 `a2_ref_fact_eval.jsonl` 与盲法映射，不重跑裁判、不破坏人工对应关系。
4. **正式方案 B 统计全量接通与无零值吞错（4.3）**
   - 彻底消除 `or 0.0` 解析失败转零问题，所有 E2/E3 统计均保留缺失状态并输出逐臂缺失率与 `parse_failed_ids`。
   - 完整接入：E2 `A1-A0` / `A7-A1` / `A7-A0`（`n_contra_ref`、`n_entail_ref`、`r_core_ref`、`p_unment_ref`、`n_unment_ref` + Holm）、E2 跨模型 DiD、E2 独立人工验证集盲包与两阶段 IPW 误差校正、E3 主比较与全部 5 组次要干预比较及跨模型 DiD、MSR-VTT 历史输出桥接，以及 `E3-Bridge` 达标时的**真实双模型四臂执行与 McNemar 统计**。
5. **正式样本隔离与分层概率抽样（4.4）**
   - 新增 [`uarcap_q2/datasets/formal_sampling.py`](uarcap_q2/datasets/formal_sampling.py)：实现参数化分层概率抽样、独立 E2 验证集抽样与纳入概率记录、Pilot 永久排除、DREAM-1K 疑似近重复排除、VidHalluc 同来源隔离与 Pilot 候选来源排除，以及带 SHA-256 的正式分析清单冻结。
6. **跨环境模型调用接口、共享 GPU 锁与显存释放（4.5）**
   - 新增 [`uarcap_q2/common/gpu_runtime.py`](uarcap_q2/common/gpu_runtime.py)、[`uarcap_q2/retrieval/xclip_worker.py`](uarcap_q2/retrieval/xclip_worker.py)（`SubprocessVideoUFOXCLIPRetriever`）、[`uarcap_q2/generators/qwen3_vllm_worker.py`](uarcap_q2/generators/qwen3_vllm_worker.py)、[`uarcap_q2/generators/modern_vlm_worker.py`](uarcap_q2/generators/modern_vlm_worker.py)（`SubprocessModernVLMAdapter`）与 [`uarcap_q2/evaluators/judge_worker.py`](uarcap_q2/evaluators/judge_worker.py)（`SubprocessTransformersJudgeBackend`），统一封装共享 GPU 锁、进程间 JSON/JSONL/PNG 交换与任务间 `release_gpu_memory()` 显存释放。
7. **一键自动化数据准备与双人人工核验闭环（4.6 + 五）**
   - 新增 [`uarcap_q2/datasets/data_preparer.py`](uarcap_q2/datasets/data_preparer.py) 与 `deployment/stage_assets.py prepare|check-ready`，串联文件识别/受控下载/分卷导入、校验、防 zip-slip 安全解压、目录规范化与元数据-视频对应校验。
   - 完善本地网页标注工具（新增命题 Split/Merge 与 `Export JSON`）并新增一键双人仲裁工具 [`scripts/adjudicate_annotations.py`](scripts/adjudicate_annotations.py)（支持 E2 与 E3 全流程，零手工改 JSON）。
