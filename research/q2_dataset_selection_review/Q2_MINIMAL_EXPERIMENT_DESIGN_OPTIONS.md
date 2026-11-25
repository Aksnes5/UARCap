# UARCap Q2 精炼实验组合方案比较与人工成本实证估算（Q2 MINIMAL EXPERIMENT DESIGN OPTIONS）

> **核心原则**：完整保留 MSR-VTT、MSVD 与百万级 VideoUFO 历史成果，绝不通过堆砌数据集来“凑工作量”，只寻找 **科学论证最有力、人工成本最可控、工程风险最低、故事最集中** 的二区论文最小闭环。

---

## 一、三种精炼实验组合方案全面对比

| 评估维度 | **方案一（维持原方案）：DREAM-1K + VidHalluc** | **方案二（推荐首选）：CaReBench (913 独立父视频) + VidHalluc** | **方案三（轻量多维替代）：CaReBench + VidHal（真实自然子集）** |
|---|---|---|---|
| **历史资产保留** | 100% 保留 MSR-VTT + MSVD + VideoUFO | 100% 保留 MSR-VTT + MSVD + VideoUFO | 100% 保留 MSR-VTT + MSVD + VideoUFO |
| **完整回答 RQ1/RQ2/RQ3** | 是（开放域 5 类视频 + 动作受控干预） | **是，且 RQ1 静态+动态拆分归因更强**（细粒度对象+动作自然检索 + 动作受控干预） | 是（可覆盖对象/数量/颜色/动作多维干预，但 E3 人工筛选成本更高） |
| **区分错误事实 vs 未提及（核心痛点）** | 中等（59 词事件描述容易漏静态对象 → `Unmentioned` 灰区偏高，A2 校准风险较高） | **强**（228 词时空分离双人+专家参考，对象密度高 73%，显著压低静态幻觉 `Unmentioned` 漏检） | **强**（E2 同方案二；E3 需人工剔除 GPT-4o 瑕疵项） |
| **来源泄漏与 YouTube ID 去重** | E2 不可查重（连父视频聚类都未知）；E3 **100% 精确 YouTube ID 零重叠** | E2 父视频可精确去重（1,000→913），YouTube ID 仍需高相似筛查；E3 **100% 精确 YouTube ID 零重叠** | E2 同方案二；E3 **丧失精确 YouTube ID 查重能力**（重大科学降级！） |
| **16 帧旧模型适配性** | 优（8.9s 均值，短字幕粒度最贴近） | 良–优（14.35s 短于 MSR-VTT 15.01s；配合开篇单句总览作 `R_core` 与三值精度指标无粒度惩罚） | 中（VidHal 含 261 条 CLEVRER 合成球体与 OCR 字母序题，须大量过滤） |
| **数据获取与许可阻力** | 中（DREAM-1K `gated: auto`，需用户先在 HF 网页登录授权并 Windows 下载中转） | **极低**（CaReBench **`gated=False`, MIT, 3.66 GiB`**，BN 镜像或 Windows 均可立即获取；VidHalluc 亦 `gated=False`） | 极低（总视频体积最小：3.66 + 5.08 = 8.74 GiB，全开放许可） |
| **论文叙事聚焦度（SCI Q2）** | 清晰（开放域自然检索 + 动作冲突干预） | **最清晰且有额外亮点**（MSR-VTT/MSVD 开放域历史结果 + CaReBench 细粒度静态/动态事实拆解 + VidHalluc 精确零重叠动作因果干预） | 略弱（E3 源头混杂 4 个子库且为 GPT-4o 合成字幕，易遭审稿人质疑真值公信力） |
| **既有工程复用与改动量** | **100% 零改动直接运行** | **复用 ~95% 既有代码**，仅新增约 150 行 `carebench.py` 加载器 + 配置路径切换 + 回归测试（约半天离线工作） | 需重写 E2 与 E3 两套加载器（约 1 天） |
| **单卡 RTX A6000 48GB 适配** | 轻松适配（估算 ≤ 35–45 GPU 小时） | 轻松适配（估算 ≤ 35–45 GPU 小时） | 轻松适配（估算 ≤ 30–40 GPU 小时） |

### 为什么不采用“单数据集同时做 E2 + E3”？
经全面核查全球 22 个基准，**目前不存在任何一个公开视频数据集**同时拥有：(1) 高密度人工撰写自然视频参考描述，**以及** (2) 独立人工验证的互斥最小编辑正负冲突对。强行在 CaReBench 或 DREAM-1K 上自行编写 180 套互斥正负干扰项会大幅增加人工构造与防泄露审校成本；而若只用 VidHalluc 则无法回答自然检索（A0/A1/A7）下的开放事实覆盖与错误变化。因此，**“1 个 E2 主数据集 + 1 个 E3 受控干预数据集（+ 可选 E3-Bridge 自然冲突子集）”** 是科学上不可再压缩的最简组合。

---

## 二、人工标注成本分项实证估算（明确区分“已有参考描述”与“核验我们新模型输出”）

**关键澄清**：任何外部数据集（无论 DREAM-1K 还是 CaReBench）**都不可能替我们认证我们自己新生成的 A0/A1/A7 字幕的对错**。数据集能节省的是：**参考事实准备、核心事件定义、以及因参考不完整而被迫扩大的校验/二阶段校正规模**。

下表按 **双人独立标注 + 联合仲裁总人时（2-annotator total person-hours）** 进行透明区间估算（基于单条 10–15s 短视频 3 条短字幕平均看片+核验 2.5–3.5 分钟/人，联合仲裁按双倍工时计）：

| 人工工作项 | 方案一：DREAM-1K + VidHalluc | 方案二：CaReBench + VidHalluc | 方案三：CaReBench + VidHal | 压缩机制与不确定性说明 |
|---|---|---|---|---|
| 1. 参考事实准备人时 | 0 人时（复用人工参考） | 0 人时（复用双人+专家人工参考） | 0 人时 | 两者均直接复用 |
| 2. 核心事件确定人时 | 2–4 人时（需校验抽取的核心事件） | **0–1 人时**（94% 自带人工开篇单句总览，仅抽检 6% 非模板句） | 0–1 人时 | CaReBench 原生总览直接省去核心事件定义 |
| 3. 模型生成字幕全量人工评价 | **0 人时（取消！）** | **0 人时（取消！）** | **0 人时（取消！）** | 全量改由跨家族开源 LLM 三值评价器执行 |
| 4. Pilot A2 自动评价器双盲校准（30 视频 × 3 臂 × 1 模型 = 90 字幕） | 8–12 人时（若高 `Unmentioned` 触发二轮重校可能升至 14–18 人时） | **7–10 人时**（静态参考完整，一轮通过概率更高） | 7–10 人时 | 最大不确定性来自一轮校准是否过线 |
| 5. Formal B 独立人工验证集（两阶段误差校正基准） | 14–24 人时（因 `Unmentioned` 偏高，大概率需自适应扩至 80 视频） | **10–16 人时**（目标压缩至 50 独立视频即可稳健估计偏差） | 10–16 人时 | 若 A2 校准显示极低偏置，可稳守 50 视频下限 |
| 6. Pilot A3 跨模型与 E3 评价器校验 + 150–180 条 E3 候选适用性看片审核 | 10–15 人时 | 10–15 人时 | 14–22 人时（VidHal 90% 为未核验 GPT-4o 字幕且需人工过滤歧义项） | VidHalluc 单视频仅核验 1 对动作短语，速度极快 |
| 7. 双人分歧联合仲裁（按 10–15% 分歧率计，2×工时） | 4–7 人时 | 3–6 人时 | 5–8 人时 | 已配套 `adjudicate_annotations.py` 一键提取分歧 |
| **合计总双人人时估算** | **38–62 人时**（风险上限 ~68 人时） | **30–48 人时**（风险上限 ~55 人时） | **36–57 人时** | **方案二预计比方案一节省约 15%–25% 人工工时，且显著降低 A2 校准返工风险** |

---

## 三、若从 DREAM-1K 切换至 CaReBench，需要修改哪些既有代码？（精确影响评估）

**结论：改动极小、高度局部化（约 95% 既有离线代码原封不动复用）！**

- **完全不用改动的既有模块（~95%）**：
  - `uarcap_q2/retrieval/`（真实 VideoUFO X-CLIP 检索器、`xclip_worker.py`、D 共识重排、臂构造器）
  - `uarcap_q2/generators/`（冻结 Video-LLaMA 批量适配器、`Qwen3VLLMMultimodalAdapter`、`qwen3_vllm_worker.py`）
  - `uarcap_q2/evaluators/`（三值参考事实评价器、`BlindE3SemanticEvaluator`、A2/A3 严格对账与校准、`backends.py`）
  - `uarcap_q2/annotation/`（网页标注 UI、盲法打包、一键双人仲裁 CLI）
  - `uarcap_q2/statistics/`（配对集群 Bootstrap/置换、Holm、McNemar、DiD、两阶段 IPW 校正、E1 历史分析）
  - `uarcap_q2/runner/orchestrator.py`（断点续跑、Trace 校验、共享 GPU 锁）
  - `uarcap_q2/datasets/vidhalluc.py`、`video_decoder.py`、`asset_transfer.py`、`formal_sampling.py`
- **仅需局部新增/调整的文件（4 处，总计约 200 行）**：
  1. 新增 `uarcap_q2/datasets/carebench.py`（读取 `json/metadata.json`，提取 `caption` / `spatial_caption` / `temporal_caption` 与开篇单句 `core_events`，内置 **913 独立 FineAction 父视频去重** + 按 4 大类分层抽样）；
  2. 在 `deployment/stage_assets.py` 与 `data_preparer.py` 增加 `carebench_metadata` / `carebench_videos` 资产条目与 `dataset == "carebench"` 分支；
  3. 在 `deployment/run_stage_cli.py` 与 `configs/bn_paths.example.yaml` / `pilot_a01_config.yaml` 增加 `--e2-dataset {dream1k,carebench}` 开关（可同时保留 `dream1k` 兼容！）；
  4. 新增 1 组 `carebench` CPU 单元测试（保持全部 59+ 项测试通过）。
