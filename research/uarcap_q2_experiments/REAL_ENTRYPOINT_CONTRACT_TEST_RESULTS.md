# 真实入口契约测试结果报告（REAL ENTRYPOINT CONTRACT TEST RESULTS）

- **执行命令**：`pytest -v research/uarcap_q2_experiments/tests`
- **总计结果**：**`52 passed, 0 failed, 0 skipped`**（含原 43 项单元测试/端到端/故障注入/`bash -n` 测试 + 本轮新增 **9 项真实入口契约与冻结黄金回归测试** `tests/test_10_real_entrypoint_contracts.py`）
- **重要边界**：本轮 9 项契约测试通过真实 `run_stage_cli.dispatch_stage`、真实 `VideoUFOXCLIPRetriever`、真实 `FrozenVideoLLaMAAdapter.generate_batch`、真实 `Qwen25VLAdapter` 与真实状态机运行，底层硬件调用使用可注入 CPU fake backend（fake X-CLIP / fake EVCap subprocess / fake processor & model / fake judge），**属于 CPU 工程契约验证，绝不等同于 BN RTX A6000 真实 GPU 验证**。

## 本轮新增 9 项真实入口契约测试明细

| # | 测试用例（`tests/test_10_real_entrypoint_contracts.py`） | 覆盖 P0 | 验证内容 | 实测结果 |
|---:|---|---|---|---|
| 1 | `test_p02_golden_historical_d_rerank_exact_match` | P0-2 | 以冻结提交 `52b5ed3a` 的 3 条真实历史 MSR-VTT 查询（`video7010/7011/7012`）验证 D 共识重排的 top-7 顺序、ROUGE-L 分数（`<1e-12`）、代表 clip 与相似度完全一致 | **PASSED** |
| 2 | `test_p02_real_videoufo_xclip_retriever_contract` | P0-2 | 真实 `VideoUFO.csv` + `clips_full.tsv` + `fp16.npy` mmap + 16 中心帧预处理 + fake X-CLIP `VideoSpecificPrompt` 分块检索 + `.pt` 特征校验 + `retrieval_manifest.json` 落盘 | **PASSED** |
| 3 | `test_p01_p03_real_pilot_a1_batch` | P0-1, P0-3 | 验证真实 CLI 缺失资产返回退出码 2（绝不打印 `READY_FOR_BN_REAL_ADAPTER`）；再经真实 `PILOT_A1` 分发调用 `VideoUFOXCLIPRetriever` + `FrozenVideoLLaMAAdapter.generate_batch`（验证 18 列 TSV、绑定 `eval->train` 检查点槽位、每臂仅 1 次 subprocess 加载 7B、前向 hook trace、`REAL_RUNTIME_PASS`） | **PASSED** |
| 4 | `test_p04_modern_vlm_real_frames_processor_telemetry_and_e3_parity` | P0-4 | 经真实 `Qwen25VLAdapter` + fake processor/model 验证：缺帧报错拒绝；16 张解码 PIL 帧真实送达 processor；捕获 `pixel_values_videos` 形状/哈希与 `input_ids` 长度/哈希；`PROMPT_TEXT_ONLY` 不伪造视觉证据；E3 四臂视觉张量哈希完全一致 | **PASSED** |
| 5 | `test_p05_p07_mock_cannot_unlock_real_stages_and_a2_a3_formal_b_chain` | P0-5, P0-7 | 验证 `MOCK_PASS` 无法解锁真实 A2；`stage_assets.py download` 强制阻断未授权 VidHalluc 下载；真实 A2 读取 A1 输出并停在退出码 10（`HUMAN_VALIDATION_PENDING`）；A3 拒绝未校准 A2；A3 经人工批准候选执行双模型四臂 + Blind E3 评测；Formal B 校验全部真实前置收据与冻结哈希后跑通并输出统计 | **PASSED** |
| 6 | `test_p06_trace_failure_never_commits_and_resume_rejects_truncation_and_config_drift` | P0-6 | Trace 校验失败时 0 行写入 `predictions.jsonl`/`input_traces.jsonl`（仅写 `failures.jsonl`）；更换 checkpoint 续跑抛出 `ResumeIntegrityError`；截断 JSONL 续跑抛出 `ResumeIntegrityError` | **PASSED** |
| 7 | `test_p07_e1_cli_entrypoint` | P0-7 | 真实 `run_stage_cli.py --stage E1` 读取历史 CIDEr JSON 并落盘 `e1_report.json` | **PASSED** |
| 8 | `test_p06_full_run_resume_metrics_cover_earlier_empty_output` | P0-6 | 第 1 次运行有空输出、第 2 次续跑新增正常输出后，最终 `empty_output_rate` 覆盖全量运行（`0.50` → `FAILED`），绝不只算新增切片 | **PASSED** |
| 9 | `test_p05_pilot_a2_human_annotation_return_and_a3_unverified_e3_pause` | P0-5, P0-7 | 验证 A2 接收人工真值回流完成校准并标记 `human_calibration_completed=True`；验证 A3 未提供人工批准候选时安全停于退出码 10（`HUMAN_VALIDATION_PENDING`）而非运行未核验干扰项 | **PASSED** |
