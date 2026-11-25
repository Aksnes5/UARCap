# UARCap Q2 科研协议局部修订记录：E2 主数据集正式切换为 CaReBench（DREAM-1K 降为备用）

> **修订日期**：2026-11-25｜**适用分支**：`antigravity/uarcap-q2-carebench-final-handoff`  
> **修订原则**：仅执行数据集替换所必需的局部协议修订。完整保留既有三大研究问题（RQ1/RQ2/RQ3）、检索预算设计（A0/A1/A7）、VidHalluc E3 四臂受控干预（Base/Sup/Con/Neu）、三值参考事实评价与统计检验框架、以及全部历史资产（MSR-VTT、MSVD、VideoUFO 与冻结 v0.8 论文）。

---

## 一、局部修订对照清单

| 协议条目 | 原协议（v1.2-final） | 本轮正式修订（CaReBench 切换版） | 对应工程实现 |
|---|---|---|---|
| **1. E2 主数据集与来源** | `DREAM-1K`（`omni-research/DREAM-1K`，gated，5 域，1,000 片段） | **`CaReBench`**（`MCG-NJU/CaReBench`，`gated=False`，`MIT`，`3.65 GiB`，源自 FineAction 4 大类 / 82 子类，1,000 片段 / 913 个不同 FineAction 父视频 ID）；`DREAM-1K` 保留为 `--e2-dataset dream1k` 备用回退 | `uarcap_q2/datasets/carebench.py`、`deployment/run_stage_cli.py --e2-dataset {carebench,dream1k}` |
| **2. 父视频级抽样策略** | DREAM-1K 5 域 × 每域 6 条 = 30 条 Pilot | **先按 FineAction 父视频 ID (`v_XXXXXXXX`) 确定性去重（1,000 → 913）**，再按 4 大类比例分层抽样 + 子类轮转分散抽取 30 条 Pilot（固定随机种子 `20261125`）；同一父视频的所有片段严禁同时进入 Pilot、调试集、独立验证集与正式主分析集 | `deduplicate_carebench_by_parent_video()`、`sample_carebench_stratified_pilot()`、`write_carebench_pilot_exclusion_manifest()` |
| **3. 样本量规划范围** | 正式 E2 规划 600–800（验证集 50–80） | 正式 E2 暂定规划调整为 **600–750**（独立评价器验证集仍暂定 **50–80**）。913 父视频扣除 30 Pilot 与 50–80 验证集后剩余 833–803，再扣解码与疑似近重复后才确定最终样本量，**不提前承诺达到 750**，且绝不依据显著性选样 | `uarcap_q2/datasets/formal_sampling.py::build_formal_e2_manifest()`、`power_planner.py::plan_formal_e2_sample_size()` |
| **4. 人工参考事实来源** | DREAM-1K 人工事件描述 | 以 CaReBench 双人独立标注 + 专家复核合并的 **`caption`、`spatial_caption`、`temporal_caption`** 为主要人工参考事实；公开元数据中的 **`events` 与 `objects` 系 GPT-4 自动提取，明确标记为 `GPT4_EXTRACTED_NOT_INDEPENDENT_HUMAN_GOLD`，绝不视作独立人工认证事实** | `load_carebench_records()` 显式区分 `human_reference_fields` 与 `llm_extracted_facts_provenance` |
| **5. 核心事件候选来源** | DREAM-1K 预模型提取核心事件 | 采用 CaReBench 开篇单句总览（~94% 样本匹配总览模板，均长 14.5 词）作为 **未验证的核心事件候选（`UNVERIFIED_CANDIDATE_REQUIRES_PILOT_A2_AUDIT`）**，不自动认证其覆盖全部核心事实，其最终可靠性由 Pilot A2 人工抽检核验 | `extract_opening_overview_candidate()`、`core_event_candidate_status` |
| **6. 16 帧可见性核验** | 16 帧均匀采样静音输入，人工标注区分 `Video-Unverifiable` | 保持相同 16 帧 / 224×224 静音输入协议；因 CaReBench 空间描述细节更密（均长 14.35s，短于 MSR-VTT 15.01s），Pilot A2 人工校准界面同步核验 16 帧采样条带，严格区分“低分辨率/帧间不可见（`Video-Unverifiable`）”与“明确视觉冲突（`Video-Contradicted`）” | 复用既有 `video_decoder.py` 与盲法网页标注 UI |
| **7. 三值指标与静态/动态辅助诊断** | 主终点：每字幕 `n_contra_ref`、`r_core_ref`、预设配对比较 | **主终点与正式假设检验族 100% 保持不变**（不因 CaReBench 分离时空描述而扩增正式假设检验）。明确区分**参考支持率（`p_entail_ref`，对应 AutoDQ/CapST Precision 定义）**、**明确参考冲突率（`p_contra_ref` / `n_contra_ref`）**与**真实视频事实错误率**；新增基于主张文本显式词法/谓词规则的辅助诊断四分类（`static_entity_attribute` / `dynamic_action_temporal` / `mixed` / `unclassified`），**严禁简单依据参考句来自 `spatial_caption` 还是 `temporal_caption` 归类** | `reference_fact_eval.py::classify_claim_fact_dimension()`、`formal_b_statistics.json::e2_auxiliary_static_dynamic_diagnostics` |
| **8. 来源独立性残余限制与 E3 推论边界** | DREAM-1K 匿名 ID 残余重叠；VidHalluc 精确去重 | (a) 明确声明 CaReBench 的 913 个不同 FineAction 父视频 ID **不等同于** 913 个已证明独立的原始 YouTube 来源，须保留检索后高相似近重复筛查与残余不确定性披露；(b) 明确 `VidHalluc` 的 YouTube ID 过滤仅证明**已知 ID 层面无交集**（不排除二次转载）；(c) 明确 **VidHalluc E3 的受控因果结论主要限于动作事实**，不可自动推广至 CaReBench 中的全部静态对象与属性错误（E3-Bridge 仍作为同库可选机制验证，由试点门控决定是否启用） | `dedup.py::screen_e2_near_duplicates()`、`e3_bridge.py`、`formal_b_statistics.json::e3_causal_scope_notice` |

---

## 二、决策支撑文档索引

本修订依据本轮全部 8 份调研与定向核验文档做出（均已同步提交至 `research/q2_dataset_selection_review/`）：
- `GLOBAL_VIDEO_DATASET_LITERATURE_SURVEY.md`
- `DATASET_ANNOTATION_PROVENANCE_AUDIT.md`
- `DREAM1K_VS_CAREBENCH_DEEP_COMPARISON.md`
- `DATASET_SOURCE_OVERLAP_AND_MODEL_FIT.md`
- `DATASET_SCIENTIFIC_SELECTION_MATRIX.json`
- `Q2_MINIMAL_EXPERIMENT_DESIGN_OPTIONS.md`
- `Q2_FINAL_DATASET_SELECTION_DISCUSSION.md`
- `Q2_DATASET_FINAL_TARGETED_REVIEW.md`
