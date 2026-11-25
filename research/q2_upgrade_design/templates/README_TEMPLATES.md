# 空白人工标注记录模板说明（templates/）

> **状态**：`DESIGN_REFINED_AWAITING_FEASIBILITY_AUTHORIZATION`（空白结构化 CSV 模板，可直接用 Excel / WPS / Google Sheets 打开使用，**无需开发任何标注软件**；暂不含任何真实视频记录）

| 模板文件 | 对应层级与阶段 | 使用时机 | 关键字段说明 |
|---|---|---|---|
| `TEMPLATE_01_CORE_FACTS_L2.csv` | **L2 核心事实预提取**（Step 1） | **仅看静音视频、未见任何字幕前填写** | `watched_muted_full`（是否完整静音观看）、`core_fact_id`（`CF1`–`CF3`）、`core_fact_category`（四类之一）、`core_fact_canonical_text` |
| `TEMPLATE_02_ATOMS_TRUTH_L1.csv` | **L1 原子真值 + L2 覆盖分**（Step 2–3） | 共享盲法命题表生成后，双人独立填写 | `atom_source_mode`（`PURE_HUMAN` 或 `AI_DRAFT_AUDITED`）、`l1_truth_label_SCU`（`S`/`C`/`U`）、`uncertainty_reason_code`、`l2_core_fact_1/2/3_score`（`0`/`0.5`/`1.0`） |
| `TEMPLATE_03_TRANSITIONS_L3.csv` | **L3 精细转移与事后证据对齐**（Step 4） | **仅在 L1 真值彻底锁死后，于先验随机 L3 子集填写** | `base_state_SCU0` / `target_state_SCU0`（`S`/`C`/`U`/`0` 表示 `Ø`）、`alignment_rule_applied`（`EXACT_SYN` / `SAME_ANCHOR_SLOT` / `UNALIGNED`）、`transition_category_1to8`、`posthoc_evidence_aligned_01`（最后一步开放检索文本后填） |
| `TEMPLATE_04_E3_TARGET_L4.csv` | **L4 E3 靶事实五态判定**（Step 5 & E3） | 隐藏 `Base/Sup/Con/Neu` 臂名下双人独立填写 | `expresses_f_plus_10U`（`1`/`0`/`U`）、`expresses_f_minus_10U`（`1`/`0`/`U`）、`l4_five_state_code`（`NEITHER` / `ONLY_POS` / `ONLY_NEG` / `BOTH` / `AMBIGUOUS`） |
| `TEMPLATE_05_TIME_LOG_M0SMALL.csv` | **M0-small 秒表吞吐与规则缺口日志** | Step 1–5 全程同步记录 | `elapsed_minutes`（精确到 0.5 分钟）、`submode_pure_vs_aidraft`、`rule_gap_comment` |

> **编码提示**：全部模板为 UTF-8（含 BOM 兼容建议在打开时选 UTF-8），正式标注前由协调员为 Annotator A 与 Annotator B 分别复制独立文件（如 `_AnnA.csv` / `_AnnB.csv`）并锁死匿名 ID。
