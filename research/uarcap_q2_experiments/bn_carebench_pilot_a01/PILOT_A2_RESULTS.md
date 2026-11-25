# Pilot A2 — CaReBench 自动事实评价与人工标注包

**状态：`PILOT_A2_HUMAN_VALIDATION_PENDING`**（预期人类参与门禁，不是运行失败）
**日期：** 2026-11-26 · **运行：** `pilot_a2_carebench_eval_v1` · **自动审计：14/14 通过**

> ⚠️ **所有自动事实标签均未经真实人工校准。** 在两位标注员独立完成、仲裁并回灌之前，
> 不得把它们表述为准确率、幻觉率、显著性结论，也不得据此选择正式样本或修改科学假设。
> `UNMENTIONED` 不等于事实错误。

## 1. 范围与授权

- 授权回执：`/home/bn/uarcap_q2_runtime/authorizations/pilot_a2_auth.json`
  （`approved=true`，`approved` 范围仅 `PILOT_A2`；未授权 A3 / Formal B）。
- 未重新运行 A1、未重新下载数据、未修改冻结历史资产。
- 读取并校验的 A1 输入（全部与归档 SHA-256 一致）：

| 文件 | SHA-256 |
| --- | --- |
| `predictions.jsonl` | `f958c87c83d31b7f28468e6637c22a3f45df338f36fc94f36c317f034a119428` |
| `input_traces.jsonl` | `cbf789fe4261f0431d4949c398eba424036c8c2d01400210112ef8dca2bbd133` |
| `retrieval_candidates.jsonl` | `14f29dde5ae9f5484b5ae404946b42a695e468eaf24821df34d32aa0dc86c2dd` |
| `run_receipt.json` | `e792d35ba5b9d3e8a45d5b7278a6d3542961426797a8be906f3e19b0b5703ae3` |

- 冻结抽样清单内部 SHA-256：`26fbf6ab…bd21`；A1 状态 `REAL_RUNTIME_PASS`。
- 核心事件锁（在**任何评价调用之前**生成）：
  `a2_core_event_lock.json`，`lock_sha256=6692d1dd20aa09f16b4815892c56547d72ead07b920790cee6d47bfacb94c552`。

## 2. 评价器选择（与 Mistral 兼容性说明）

配置首选 `mistralai/Mistral-Small-3.2-24B-Instruct-2506`。实际核查：

- 配置中的非 FP8 目录不存在；现有资产是
  `/media/.../xyl/Jnh_rag/models/Mistral-Small-3.2-24B-Instruct-2506-FP8`
  （Mistral 原生格式：`consolidated.safetensors` + `params.json`，FP8）。
- 共享 vLLM 0.29.0 环境搭配 transformers 5.17.0 时，vLLM 的 Pixtral 实现无法导入
  （`PixtralRotaryEmbedding`、`position_ids_in_meshgrid` 已被移除）；只读兼容补丁不足，
  且任务禁止下载新的大型评价模型、也不修改共享环境。
- 因此使用已有的 **priority-2 跨家族备选**：
  `RedHatAI/gemma-3-27b-it-FP8-dynamic`（Gemma 3 27B Instruct，路径
  `/media/.../xyl/Jnh_rag/models_campaign/M04_gemma3_27b`），judge family `gemma`。
  生成器为 Video-LLaMA/Vicuna，家族不同。
- 运行配置：vLLM 0.29.0、transformers 5.17.0、torch 2.13.0+cu130、temperature 0.0、
  `max_new_tokens=768`、`max_model_len=4096`、`gpu_memory_utilization=0.85`、
  `language_model_only=True`、原生采样器（`VLLM_USE_FLASHINFER_SAMPLER=0`，共享环境缺 `ninja`）。
- 峰值显存 44.37 GB（A6000 48 GB）；评价结束后 GPU 已释放（0 MiB / 0%）。

## 3. 自动评价协议

- 90 条字幕全部评价；同一 `(query_id, caption)` 完全相同者只评价一次：
  **50 个唯一字幕输入、40 次复用**；每条预测仍保留独立行、独立臂映射与统计权重。
- 三轮批量调用（每批 16 条）：原子事实抽取 → 对完整人工参考（`caption` + `spatial_caption` +
  `temporal_caption`）做三值 NLI → 对预锁定核心事件做覆盖判定。
- 严格解析：标签别名无损归一化（如 `ENTAILMENT`→`entailed`）；命题文本按输入顺序对齐；
  不接受语义改写；`null` 证据视为缺失；格式错误重试（最多 2 次），不静默默认。
- 事实维度：对象/属性/动作/关系/混合/无法分类（规则化辅助诊断，不新增正式检验）。
- CaReBench 的 GPT 抽取 `events/objects` **未**作为 Gold。

**本轮结果：90/90 解析成功，0 重试，0 解析失败，150 次原始调用，126.3 s。**
（前序格式调试运行保留了 `superseded_*` 目录，未混入最终结果。）

## 4. 自动描述性诊断（未校准）

| 臂 | 字幕 | 解析成功 | 原子事实 | Entailed | Contradicted | Unmentioned | 支持率 | 冲突率 | 未提及率 | 核心事件覆盖 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ARM_A0 | 30 | 30 | 85 | 66 | 7 | 12 | 0.7765 | 0.0824 | 0.1412 | 0.6667 |
| ARM_A1 | 30 | 30 | 80 | 64 | 7 | 9 | 0.8000 | 0.0875 | 0.1125 | 0.7000 |
| ARM_A7 | 30 | 30 | 83 | 63 | 7 | 13 | 0.7590 | 0.0843 | 0.1566 | 0.6667 |

- 自动评价失败率：**0 / 90 = 0.0%**。
- 事实变化（描述性，非显著性）：
  - A0→A1：新增 11，删除 16，替换对 3。
  - A1→A7：新增 27，删除 24，替换对 7。
  - A0→A7：新增 29，删除 31，替换对 9。
- 细粒度维度（总计 248 条事实）：对象 100、属性 7、动作 12、关系 1、混合 99、无法分类 29。
- 典型变化示例（完整列表见 `PILOT_A2_AUTO_DIAGNOSTICS.md`）：
  - 新增：`v_00010336_1` A1 “One musician is playing a cello.”（entailed）。
  - 删除：`v_00007888_2` A0 “The hurdles are on a track.”（unmentioned）→ A1 未复现。
  - 替换：`v_00002283_1` “A woman is riding a scooter.”（A1 unmentioned）→
    “A girl is riding a scooter.”（A7 entailed）。
- 重点视频：`v_00002093_8`、`v_00008320_0`、`v_00010336_1`、`v_00013412_0`、`v_00015614_0`
  的逐臂字幕、命题、标签与核心覆盖见诊断报告。

**解释边界：** 这些是自动代理指标，只能用于 Pilot A2 的校准与失败模式检查；
不得写成正式效果结论。

## 5. 盲法人工标注包

- `/home/bn/uarcap_q2_runtime/outputs/pilot_a2/pilot_a2_carebench_eval_v1/blind_annotation_package/`
  - `blind_tasks_e2_A1.json` / `blind_tasks_e2_A2.json`：各 30 视频任务、90 条匿名卡、
    每任务 16 帧实际模型输入帧；包内无臂名、模型名、检索证据、自动标签或参考描述。
  - `blind_key_e2_DO_NOT_OPEN.json`：**不推送、不展示给标注员**；解盲/仲裁专用。
  - `frames/`：30×16 张 JPEG（8.4 MB），仅本地保留，不推送。
- 包 SHA-256：`f55acbaa6f4baf88bfb884a82c78017a5ccb54dfceac11ff5eb88f598a6ac571`。
- 工作量（去除同视频重复后）：

| 项目 | 数值 |
| --- | --- |
| 视频任务 | 30 |
| 盲法字幕卡 | 90 |
| 同视频唯一字幕 | 50（40 张可复用） |
| 原子命题总数 / 同视频唯一 | 248 / 119 |
| 可盲法复用的标签数 | 129 |
| 预计每位标注员耗时 | **81–143 分钟** |

## 6. 独立审计

`AUDIT_PILOT_A2_AUTO.json` / `.md`：14/14 通过，包括：

- 90 行完整、全部 OK、同输入字幕的标签/证据完全一致且每臂 claim ID 唯一；
- 完整人工参考字段一致（`caption`+`spatial`+`temporal`）；
- 核心事件 verdict 数与锁定事件数一致，锁时间早于评价；
- 两个盲法包 30 任务/90 卡/16 帧，无臂泄露；盲法键覆盖 2×90 条件与全部 249→248 命题映射；
- 150 条原始调用全部保存；A1 输入 SHA-256 全部一致；
- 状态严格为 `PILOT_A2_HUMAN_VALIDATION_PENDING`。

## 7. 下一步（人工）

1. 按 [`PILOT_A2_ANNOTATION_GUIDE.md`](PILOT_A2_ANNOTATION_GUIDE.md) 启动两个标注服务，
   两位标注员独立完成 30 个视频任务。
2. 导出两份 JSON，运行 `scripts/adjudicate_annotations.py --task e2`，完成第三方仲裁。
3. 以 `--human-annotations` 回灌 `run_pilot_a2.sh`；只有在那之后才允许讨论 A2 校准结论。
4. **不得**自动启动 Pilot A3 或 Formal B。

## 8. 产物索引（本地完整保存）

| 文件 | 字节 | SHA-256 |
| --- | --- | --- |
| `predictions`（A1，复用） | 200,978 | `f958c87c…a119428` |
| `a2_ref_fact_eval.jsonl` | 283,532 | `63747be652051a65ac9513abf5b8a1d39a2fcc738c83dacb167cdf3917594968` |
| `a2_judge_raw_calls.jsonl` | 304,611 | `db837335c12de570a858a2934e8e8352f359097cfc5370d7de619e5d3cd9ff90` |
| `a2_judge_run_receipt.json` | 1,517 | `918239ed2ad2ef29a3e680f211accd61e0743ed6f16a253b5f36458bc7dc8505` |
| `a2_core_event_lock.json` | 116,784 | `2deb0464f47bfa70066d8fe5626c969743c97b46e6d2ec6a67b26135e58aa90e` |
| `a2_input_verification.json` | 2,460 | `56ee5cd9b27eef6ed3a6b5e942b788d91418c29b7d2e0eca8ecbf1844b133b5b` |
| `a2_bundle_manifest.json` | 1,527 | `bb007b931cb136546422635fcc631dbc5e5dc404ed40dfec25d689765a691764` |
| `calibration_receipt.json` | 1,307 | `031da7e070695e6a5260c4a411668a4f0a8ec20f7cda756528e867e51e160e75` |
| `PILOT_A2_AUTO_DIAGNOSTICS.md` | 10,148 | `4e9b2608d5cd3ed2ef7710a08e829154b9a521a3dda860a4766cae73a4a04936` |
| `blind_tasks_e2_A1.json` | 141,118 | `74e164d367051c857372d870c0c4b25bfe75ae6252977e746db64018ab4e6e0d` |
| `blind_tasks_e2_A2.json` | 141,118 | `0b2e7f18d236a3a48ef6e0587eb2a79bbf9dec22d3b8006f21c183f641cf8370` |

**受保护、不推送：** `blind_key_e2_DO_NOT_OPEN.json`、`frames/`、`annotation_saves/`、
任何未完成的人工标注答案。
