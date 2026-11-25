# UARCap Q2 离线真实执行链整改报告（OFFLINE REAL EXECUTION INTEGRATION REPORT）

> **最终状态**：`OFFLINE_REAL_EXECUTION_IMPLEMENTED_AWAITING_BN_RUNTIME_VERIFICATION`  
> **整改分支**：`antigravity/uarcap-q2-real-execution-integration`（基于 `a00f8f274fa54b47dd765bb92bd8fb8e7f704fcc` 独立切出，保留全部历史提交与冻结资产）  
> **重要声明**：全部 7 项 P0（P0-1 至 P0-7）真实执行链缺陷均已在 Windows CPU 离线环境下彻底修复，并通过真实入口契约测试与冻结历史黄金回归测试；但**所有涉及 RTX A6000 48GB GPU 推理、百万级 VideoUFO 索引加载、真实模型吞吐与人工标注的事项均仍待 BN 服务器恢复后现场实测**，绝不将 CPU 契约测试等同于真实 GPU 验证。

---

## 一、本轮整改的 7 项核心实质成果

1. **P0-1（删除真实 CLI 假成功占位）**：彻底移除 `deployment/run_stage_cli.py` 中 `if not args.cpu_mock: print("READY_FOR_BN_REAL_ADAPTER"); return 0` 占位路径，改为真实分发至 `PILOT_A0`、`PILOT_A1`、`PILOT_A2`、`PILOT_A3`、`FORMAL_B` 及 `E1`，缺失资产返回退出码 `2`（`FAILED`）、授权不足返回退出码 `3`（`BLOCKED`）、等待人工核验返回退出码 `10`（`HUMAN_VALIDATION_PENDING`），彻底杜绝未产生结果却静默退出 0 的假成功。
2. **P0-2（接通真实 VideoUFO X-CLIP 检索器 + 历史黄金回归测试）**：
   - 从冻结历史提交 `52b5ed3a`（`scripts/extract_query_features_v2.py` 与 `scripts/make_external_retrieval_test.py`）1:1 恢复真实实现至 `uarcap_q2/retrieval/videoufo_retriever.py`（`VideoUFOXCLIPRetriever` + `BNXCLIPBackend`），涵盖 16 中心帧预处理、VideoUFO.csv + `clips_full.tsv` strip-only 分组与字典序代表 clip 排序、`videoufo_brief_text_fp16.npy` mmap 分块（100,000）查询条件化 `VideoSpecificPrompt` 打分、top-20 候选提取与代表 `.pt` 特征校验；
   - 修复了 `reranker.py` 与冻结历史脚本的两处隐蔽偏差：`rouge_l_f1` 改为精确 `a.lower().split()`、D 排序平局打破改为严格按原始检索位次 `(-q[i], i)`；并在 50 条历史冻结 MSR-VTT 查询（及内置 3 条永久回归 fixture `fixtures/golden_historical_d_rerank_sample.json`）上验证排序、ROUGE-L 分数（误差 `< 1e-12`）、代表 clip 与相似度权重 **100% 精确一致**。
3. **P0-3（Video-LLaMA 批量执行、18 列 TSV 修复与真实检查点槽位绑定）**：
   - 核查冻结 `vision_language_tsv.py` 第 311 行（`len(row) >= decoder_num_frames + 2`，帧从 `row[2:]` 开始），修复此前 17 列 TSV 潜在退化为单图的隐患，改为精确 18 列（`qid \t 16 \t f0 ... f15`）；
   - 为 `GeneratorAdapter` 升级一等 `generate_batch(...)`，并在 `FrozenVideoLLaMAAdapter` 实现真实批量执行（每臂仅加载一次 7B 模型，彻底消除 `NotImplementedError`）；
   - 依据冻结 `eval_evcap` 的 `outpath.replace('eval','train')` 加载规则，强制绑定并校验 `out_rel.replace('eval','train')/FROZEN_CHECKPOINT_NAME` 指向规范检查点且 SHA-256 匹配，同时通过只读 wrapper 挂载前向 hook 采集真实 `evcap_forward_trace_{arm}.jsonl`。
4. **P0-4（现代 VLM 实际视频帧传递与真实 Processor 观测 Trace）**：
   - 修复 `orchestrator.py` 将真实解码的 16 张 `PIL.Image` 帧经 `BatchItem.frame_images` 传入 `Qwen25VLAdapter`；缺帧立即抛出 `TraceVerificationError`，绝不再静默传 `videos=None`；
   - 显式区分 `PROMPT_TEXT_ONLY`（`visual_evidence_ids=[]`）与旧模型 `BIMODAL_CROSS_GATING`，实际应用 RNG 种子、校验本地 `model_revision.json`，并从 processor 实际接收帧与返回张量提取 `actual_frames_sha256`、`pixel_values_shape`/`sha256`、`input_ids_len`/`sha256` 写入 `InputTrace`。
5. **P0-5（统一状态机与 Mock-vs-Real 彻底隔离）**：
   - 引入 7 态 `StageRunStatus`（`MOCK_PASS`、`REAL_RUNTIME_PASS`、`HUMAN_VALIDATION_PENDING`、`INCONCLUSIVE`、`FAILED`、`BLOCKED`、`SCIENTIFIC_PROTOCOL_PENDING`）；
   - `MOCK_PASS` **永久无法解锁**任何真实后续阶段或数据下载；A2 需 A1 `REAL_RUNTIME_PASS`，A3 需 A2 `REAL_RUNTIME_PASS` 且 `human_calibration_completed=True`；`FORMAL_B` 强制校验 3 份真实前置收据与 4 项冻结文件 SHA-256 绑定，且 `run_formal_b.sh` 完整透传全部前置参数至内部 CLI；`stage_assets.py download` 真正执行 `stage_unlock` 授权校验。
6. **P0-6（Trace-Before-Commit 与配置绑定断点续跑）**：
   - 仅当 `verify_actual_input_trace` 通过后才允许成对原子提交 `predictions.jsonl` + `input_traces.jsonl`（附 `run_config_fingerprint` 与 `trace_sha256`）；Trace 校验失败绝不写入成功记录；
   - 续跑重新校验每行完整性（截断、未成对、配置/模型指纹变更一律抛出 `ResumeIntegrityError`），精确比对预期 `query × arm × model` 集合（报告缺失/重复/多余），并在**完整全量运行记录**上计算最终 `empty_output_rate` 等指标。
7. **P0-7（A2 / A3 / Formal B / E1 / E3-Bridge 全链路串联）**：新增 `uarcap_q2/runner/stage_pipelines.py`，打通 A1→A2（含人工等待停顿与回流校准）→A3（含未审核 VidHalluc 停顿与双模型四臂评测）→Formal B（E2+E3+可选 E3-Bridge+最终统计）及独立 E1 命令行入口。
