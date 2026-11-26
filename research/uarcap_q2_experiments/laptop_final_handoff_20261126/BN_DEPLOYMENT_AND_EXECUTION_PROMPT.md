# BN 服务器 V2 下载、预检、部署与执行指令

> **运行界面依据**：严格继承 `handoff_20261126/BN_RUNTIME_INTERFACE.md` 中声明的 BN 真实路径、Python 环境（`videollama2` / `.venv_rag_vllm` / `repro_evcap`）、模型目录、`/home/bn/videoufo_research/gpu.lock` 协作锁（`GPU-66d83fcc-2695-33a7-33aa-af30b7e3d26d`）与 `/home/bn/uarcap_q2_runtime/overnight/20261126/` 调度器接口。
> **严禁**：在无操作者显式审批收据前启动任何 GPU 推理；覆盖/删除已完成正式产物；并行启动第二个调度器；终止非本队列进程（如外部常驻 PID）。

## 1. 拉取独立 V2 发布目录并核对完整 40 位 Commit SHA

将 `expected` 设为本次交付经远端核对一致的完整 40 位 Git Commit SHA（见最终交付报告）。

```bash
expected='<交付完整40位SHA>'
branch=antigravity/uarcap-q2-laptop-final-development-v2-20261126
release=/home/bn/uarcap_q2_runtime/deploy/laptop-20261126-v2
git clone --filter=blob:none --no-checkout --single-branch --branch "$branch" ssh://git@ssh.github.com:443/Aksnes5/UARCap.git "$release" &&
git -C "$release" sparse-checkout set research/uarcap_q2_experiments research/uarcap_v06_followup_deployment/bn_v06_full_ab manifests reports/v07_final paper &&
git -C "$release" checkout --detach "$expected" &&
test "$(git -C "$release" rev-parse HEAD)" = "$expected"
```

## 2. 只读环境预检、CPU 契约测试与调度器合并干跑（零 GPU 推理）

```bash
bash "$release/research/uarcap_q2_experiments/deployment/install_q2_v2_bn.sh" "$expected" --dry-run
```

该命令依次执行：
1. `deployment/q2_v2_bn_preflight.py`：只读检查 3 个 Python 环境、Qwen3.8-27B-FP8 / Gemma 3 27B / Llama 3.3 70B / Vicuna-7B / EVCap checkpoint、既有正式产物、`videoufo_clip_locations.jsonl` 与 `vidhalluc_debug_source_ids.txt`（缺失项按模块报告，不阻断无关模块）。
2. 在 BN `videollama2` CPU 环境下运行全量 `pytest` 与 `audit_q2_queue.py`（核验 111 任务、0 指纹漂移）。
3. `deployment/q2_laptop_queue.py --dry-run`：检查 `/home/bn/uarcap_q2_runtime/overnight/20261126/task_queue.json` 与 `task_state.json`，验证无重复调度器进程，确认未启动的 V1 任务可安全退役，保护所有已完成或运行中的既有产物与任务。

## 3. 可选前置输入准备（按需，仅 CPU）

- **VideoUFO 媒体定位索引（P0-4 & P1-2）**：若 `/home/bn/uarcap_q2_runtime/data/source_witness/videoufo_clip_locations.jsonl` 尚未生成，使用 `scripts/generate_visual_signatures.py index --inventory <declared_shards.json> --clips /home/bn/videoufo_research/manifests/clips_full.tsv --retrieval /home/bn/uarcap_q2_runtime/outputs/formal_candidate_unvalidated/formal_retrieval.jsonl --out /home/bn/uarcap_q2_runtime/data/source_witness/videoufo_clip_locations.jsonl` 生成，**严禁批量解压全量归档**。
- **E3 调试源排除清单（P0-1 储备池）**：确认 `/home/bn/uarcap_q2_runtime/data/source_witness/vidhalluc_debug_source_ids.txt` 包含历史调试用 YouTube ID（每行 11 字符 ID）。
- **E3 储备池追加授权（仅在原始 180 筛查 `< 150` 时使用）**：在阅读 `E3_RESERVE_POOL.json` 与 `ORIGINAL_SCREEN_FREEZE.json` 后，若决定启用合法储备续筛，于生成前写入绑定 `pool_sha256`、`policy` 与 `max_additional_candidates` 的 `/home/bn/uarcap_q2_runtime/deploy/laptop-v2-e3-reserve-approval.json`（详见 `E3_RESERVE_POOL_PROTOCOL.md`）。若不提供，系统自动输出 `E3_INSUFFICIENT_VALID_CANDIDATES` 且不阻断 A3/A5 及其他任务。

## 4. 审批激活 V2 队列（111 项，协作 GPU 锁 + 60s 心跳 + 30 分钟巡检）

确认干跑摘要无误后，创建绑定本次完整 SHA 的操作者审批文件 `/home/bn/uarcap_q2_runtime/deploy/laptop-v2-gpu-approval.json`（含 `authorized: true`、`commit: "<expected>"`、`approved_by: "<operator>"`），然后执行：

```bash
bash "$release/research/uarcap_q2_experiments/deployment/install_q2_v2_bn.sh" "$expected" --install /home/bn/uarcap_q2_runtime/deploy/laptop-v2-gpu-approval.json
```

队列将按优先级与协作锁自动执行以下全部模块（输出隔离至 `/home/bn/uarcap_q2_runtime/outputs/laptop-20261126-v2/`，零改写旧产物）：
1. **敏感性设计校验 (`control_design`)**：确认 V1 详细长度未启动后，锁定 13–15 词统一长度目标、40 查询 Qwen 模态子集与 60 查询嵌套帧数子集。
2. **E3 V2 完整重筛 + 条件储备续筛 + 双模型生成与双评价器分析**（含单/双评价器独立汇总）。
3. **Qwen 交叉评价器 1 条缺失键修补 (`qwen_cross_single`)**（复用 5,098 条哈希验证标签）。
4. **A3/A5 全预算扩展 (`budget_*`)**：Video-LLaMA 与 Qwen 各 600×2 试跑→全量生成→Gemma 评价→Llama 交叉评价→单模型/联合传统指标与预算报告（默认 `--skip-spice`，SPICE 隔离不影响其他指标）。
5. **已知答案评价器压力测试 (`stress_gemma` / `stress_llama` / `stress_report`)**：196 项受控文本 NLI 与原子拆并病例。
6. **E3-Bridge V2 (`bridge_*`)**：60 CaReBench 自然冲突候选 V2 筛查、≥30 门槛冻结、双模型生成与双评价器分析。
7. **Qwen 统一目标长度 (`length_*`)、外部视觉帧 2×2 因子 (`qwen_factorial_*`)、EVCap 模态 (`modality_*`)、32/64 帧 A7+嵌套 A0 交互 (`visual32_*` / `visual64_*`)**（64 帧失败严格与 16/32 帧隔离）。
8. **真实视频近重复视觉签名 (`visual_signature_generation` → `visual_risk` → `visual_risk_sensitivity_*`)**（600 查询 + 2,622 去重 Top-7 clip）。
9. **自然溯源 (`fact_provenance*`)、Top-7 语义相关性 (`retrieval_relevance*`)、MSR-VTT 历史反向桥接 (`historical_*`)**（全含 Gemma 主评 + 60 查询 Llama 可靠性抽检 + 矢量/位图输出）。
10. **统一释放总汇报 (`final_snapshot`)**：独立收集全部通过与部分产物，输出 `FINAL_EXPERIMENT_MATRIX.json`、`ARCHIVE_MANIFEST.json` 与 `MANUSCRIPT_EVIDENCE.md`。

## 5. 独立人工盲法验证与确认性门禁（人工阶段，严禁跳过）

按 `HUMAN_VALIDATION_RUNBOOK.md` 运行 `scripts/q2_human_portal.py` 构建双盲包，并为两位评审员分别在隔离端口 + 私有回环访问令牌下启动 `scripts/launch_annotation_ui.py`。导出双份独立签收与独立仲裁结果后，执行 `q2_human_portal.py import`、`build_reference_endpoint_import.py` 与 `run_final_analysis.py`。
- **探索性产出**（`MACHINE_ASSISTED_UNVALIDATED` / `POST_HOC_MACHINE_ASSISTED`）在 BN 运行完成后即可审阅；
- **确认性主统计**（`run_final_analysis.py --confirmatory`）在全部人工校准/核验与 `FORMAL_B_AUTHORIZATION.json` 到齐前持续保持硬门禁阻断（退出码 3）。

## 6. 安全停机与回滚

- 停机：创建 `/home/bn/uarcap_q2_runtime/overnight/20261126/STOP_REQUESTED`（当前任务平滑完成后停止后续调度）。
- 回滚：按 `BN_DEPLOYMENT_ROLLBACK.md` 使用 `--rollback` 撤回未运行的 `q2l_v2_*` 任务并保留所有备份与产物。

[操作验收清单](BN_FINAL_DEPLOYMENT_CHECKLIST.md) · [资源估算](BN_RESOURCE_ESTIMATE.md) · [V2 任务队列 (111 项)](REMAINING_GPU_TASK_QUEUE.json)
