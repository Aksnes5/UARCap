# 真实入口契约与离线收尾测试结果报告（REAL ENTRYPOINT CONTRACT TEST RESULTS）

- **执行命令**：`pytest -q research/uarcap_q2_experiments/tests`
- **总计结果**：**`59 passed, 0 failed, 0 skipped`**
  - 既有 43 项单元/端到端/故障注入/`bash -n` 测试（`test_01`–`test_09`）
  - 既有 9 项真实入口契约与历史黄金回归测试（`test_10_real_entrypoint_contracts.py`，已同步升级适配 A2 稳定 `claim_id` 与 A3 人工核验停机节点）
  - 本轮新增 **7 项离线收尾回归测试**（`test_11_final_handoff_closures.py`）
- **明确边界**：以上测试在 Windows CPU 轻薄本完成，验证离线工程逻辑、数据契约、跨环境进程接口与统计计算正确性；**绝不等同于 BN RTX A6000 真实 GPU 环境运行验证**。

## 本轮新增 7 项回归测试明细（`tests/test_11_final_handoff_closures.py`）

| # | 测试用例 | 覆盖任务 | 验证内容 | 结果 |
|---:|---|---|---|---|
| 1 | `test_qwen3_8_27b_fp8_vllm_multimodal_adapter_thinking_control_and_trace` | 模型选型修正 | `Qwen3.8-27B-FP8` / `Qwen3.5-27B` 的 `vllm_offline` + `vllm_openai_http` 适配、16 帧强校验、本地分片完整检查、关闭 thinking 与 `<think>` 剥离审计、`PROMPT_TEXT_ONLY` 零视觉泄漏与 E3 四臂视觉一致性 | **PASSED** |
| 2 | `test_a2_strict_claim_id_reconciliation_omissions_split_merge_and_bundle_reuse` | 4.2 | A2 重入零重跑裁判并复用既有盲法映射；拒绝缺失/重复 `claim_id` 与错误 `package_sha256`；准确统计遗漏命题补录（`added_claims`）与拆分修订（`claim_revisions`） | **PASSED** |
| 3 | `test_a3_e3_differential_bias_detection_and_cross_model_dream1k_audit` | 4.1 | `audit_e3_evaluator_agreement` 准确检出 `Con vs Neu` 差分偏置并判 `FAIL`；完整 A3 流程停于退出码 10 并在导入 E3 与跨模型 DREAM-1K 人工真值后产出完整校验报告 | **PASSED** |
| 4 | `test_formal_b_sampling_full_stats_no_zero_coercion_e3_bridge_and_msrvtt` | 4.3, 4.4 | 分层概率抽样、近重复排除、Pilot 与 VidHalluc 同来源隔离；解析失败保留缺失不转 `0.0`；完整计算 `A1-A0`/`A7-A1`/Core/Unmentioned/两阶段 IPW/E3 次要比较/真实 E3-Bridge 执行与 MSR-VTT 历史桥接 | **PASSED** |
| 5 | `test_isolated_xclip_and_judge_subprocess_interfaces_and_gpu_lock` | 4.5 | `SubprocessVideoUFOXCLIPRetriever` 批量预取、`SubprocessTransformersJudgeBackend`、`LocalVLLMJudgeBackend`、共享 GPU 锁与跨进程 JSON/JSONL 交换 | **PASSED** |
| 6 | `test_one_click_data_prepare_safe_extract_normalize_and_zipslip_block` | 4.6 | `stage_assets.py prepare` & `check-ready` 一键串联分卷导入、防 zip-slip 解压、嵌套目录规范化与元数据-视频对应校验，并拦截恶意 `../` 压缩包 | **PASSED** |
| 7 | `test_end_to_end_web_ui_export_adjudication_cli_for_both_e2_and_e3` | 五 | 本地网页标注器 `AnnotationStore` JSON 导出 → `scripts/adjudicate_annotations.py`（E2 与 E3）分歧提取与仲裁 → 直接回流 A2/A3 流水线 | **PASSED** |
