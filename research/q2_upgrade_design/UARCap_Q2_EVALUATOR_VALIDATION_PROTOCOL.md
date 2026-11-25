# UARCap Q2 事实评价器验证与校准协议（EVALUATOR VALIDATION PROTOCOL）

> **文档状态**：`PILOT_PROTOCOL_READY_AWAITING_AUTHORIZATION`（设计协议，未执行任何模型推理或人工标注）  
> **适用范围**：规范 E2（DREAM-1K 参考事实代理评价器）与 E3（VidHalluc / E3-Bridge 双盲单目标语义评价器）的校准、独立验证、差分偏置审计与回退决策。

---

## 1. 三层评价协议严格分层（严禁混名）

为避免将 UARCap 自定义诊断协议与官方 benchmark 指标混为一谈，全研究强制区分以下三层，并在论文与代码配置中使用互斥命名：

| 层级 | 标准名称 | 定义与代码锚点 | 在本研究中的角色 |
|---|---|---|---|
| **Layer 1** | **Official AutoDQ (Faithful)** | 原样运行 `bytedance/tarsier`（pinned commit `63de87a760c2aec6b31470a08046969a13787126`）`evaluation/metrics/evaluate_dream_gpt.py`，按原论文 `arXiv:2407.00634v2` Section 4.2 严格参考支持口径（Recall 与 Precision 均仅计 `entailment`），默认商业 API 模型。 | **可选独立敏感性对照**（非默认必跑付费项；若运行需原样报告，严禁改动聚合逻辑）。 |
| **Layer 2** | **Local AutoDQ-Style Substitute** | 将 Layer 1 的商业 API 替换为本地跨家族开源 LLM，但保持 `Entailed / Total` 严格支持聚合规则不变。 | 仅用于量化“商业 API → 本地开源模型”替换敏感性。 |
| **Layer 3（主协议）** | **UARCap 3-Way Reference-Fact Diagnostic Protocol** | 以本地跨家族开源 LLM 输出三值参考关系（`Entailed` / `Contradicted` / `Unmentioned`），分别计算 `N_Contra_ref`、`N_Entail_ref`、`P_Unment_ref` 及事前预抽核心事件覆盖 `R_core_ref`。 | **E2 核心自动代理评价协议**；**绝不称为“修正版 AutoDQ”**，且所有输出均显式标注为参考代理指标，区别于人工看视频真值 `N_C`、`N_S`、`R_core`。 |

---

## 2. 参考代理 × 人工视频真值的 3×4 交叉评价矩阵（E2 核心验证）

### 2.1 人工视频真值四分类（静音独立看片）

人工标注员在不看参考文本、不知晓实验臂与模型来源的盲态下，直接观看静音视频片段，对每条从生成字幕抽出的原子事实陈述 $a$ 给出四类视频真值标签之一：

- **$G_{\text{Sup}}$（Video-Supported）**：视频画面清晰支持该事实陈述为真；
- **$G_{\text{Err}}$（Video-Contradicted / True Error）**：该事实陈述与视频画面明确矛盾或凭空捏造（视频中不存在该主体、动作、对象或属性）；
- **$G_{\text{Unv}}$（Video-Unverifiable）**：因遮挡、画质、镜头切出、缺少音频或主观推断，仅凭视频画面无法可靠判定真假；
- **$G_{\text{Non}}$（Non-factual / Boilerplate）**：纯修辞套话或无法独立检验真值的空泛框架语（单独统计，不混入真假分母）。

### 2.2 3×4 交叉列联表

将自动评价器给出的三值参考代理标签 $R \in \{\text{Entailed}, \text{Contradicted}, \text{Unmentioned}\}$ 与人工视频真值 $G \in \{G_{\text{Sup}}, G_{\text{Err}}, G_{\text{Unv}}, G_{\text{Non}}\}$ 交叉统计（同时报告总体表，以及按 **`A0 / A1 / A7` 各预算臂** 和 **Video-LLaMA / 现代 VLM 各模型** 分层的子表）：

| 自动参考代理标签 $R$ \ 人工视频真值 $G$ | $G_{\text{Sup}}$（视频为真） | $G_{\text{Err}}$（真实视频错误） | $G_{\text{Unv}}$（视频无法核验） | $G_{\text{Non}}$（非事实） | 行合计 |
|---|---|---|---|---|---|
| **`Ref-Entailed`** | $n_{E,S}$ | $n_{E,E}$（参考误支持错误） | $n_{E,U}$ | $n_{E,N}$ | $n_{E,\cdot}$ |
| **`Ref-Contradicted`** | $n_{C,S}$（假阳性参考冲突） | $n_{C,E}$（真阳性冲突捕获） | $n_{C,U}$ | $n_{C,N}$ | $n_{C,\cdot}$ |
| **`Ref-Unmentioned`** | $n_{U,S}$（真实但参考未载） | $n_{U,E}$（**隐蔽漏检错误！**） | $n_{U,U}$ | $n_{U,N}$ | $n_{U,\cdot}$ |
| **列合计** | $n_{\cdot,S}$ | $n_{\cdot,E}$ | $n_{\cdot,U}$ | $n_{\cdot,N}$ | $N_{\text{atoms}}$ |

### 2.3 六项强制测量指标（全部以视频聚类自助法报告 95% CI）

1. **参考冲突的真实错误精确率**：
   $$\text{Prec}_{\text{Contra}\to\text{Err}} = \frac{n_{C,E}}{n_{C,\cdot}}$$
   衡量 `N_Contra_ref` 中究竟有多大比例是真正的视频事实错误（而非参考粒度差异导致的伪冲突）。
2. **真实视频错误的参考冲突捕获率（召回率）**：
   $$\text{Rec}_{\text{Err}\to\text{Contra}} = \frac{n_{C,E}}{n_{\cdot,E}}$$
   衡量真实视频错误中，有多少能被 `Contradicted` 成功捕获。
3. **`Unmentioned` 中的真实视频错误率（漏检风险率）**：
   $$\text{Leak}_{\text{Unment}\to\text{Err}} = \frac{n_{U,E}}{n_{U,\cdot}}$$
   直接检验 `Unmentioned` 究竟藏匿了多少真实幻觉，彻底杜绝把 `Unmentioned` 误当“无风险”或“正确”。
4. **`Unmentioned` 中的真实但未记载事实率**：
   $$\text{TrueUnrec}_{\text{Unment}\to\text{Sup}} = \frac{n_{U,S}}{n_{U,\cdot}}$$
   量化模型补充“视频为真但参考描述未写”细节的比例，用于解释为何不能把 `Unmentioned` 全部当作幻觉惩罚。
5. **视频无法充分核验率**：
   $$P_{\text{Unv}} = \frac{n_{\cdot,U}}{N_{\text{atoms}}}$$
   监控静音视频画面的可核验边界（若某类陈述如听觉/对话导致 $P_{\text{Unv}}$ 过高，需单独分层敏感性分析）。
6. **跨预算臂与跨模型的系统性误差差异（核心防偏指标）**：
   分别在 $b \in \{A0, A1, A7\}$ 与 $m \in \{\text{Video-LLaMA}, \text{Modern-VLM}\}$ 上计算上述 1–5 项，重点检验：
   - 捕获率差：$\Delta \text{Rec}_{A7 - A0} = \text{Rec}_{\text{Err}\to\text{Contra}}(A7) - \text{Rec}_{\text{Err}\to\text{Contra}}(A0)$
   - 隐蔽漏检率差：$\Delta \text{Leak}_{A7 - A0} = \text{Leak}_{\text{Unment}\to\text{Err}}(A7) - \text{Leak}_{\text{Unment}\to\text{Err}}(A0)$
   - 跨模型漏检率差：$\Delta \text{Leak}_{\text{VLM} - \text{VLlama}}$

---

## 3. 配对变化（Paired-Delta）校准与差分偏置否决规则

### 3.1 为什么总体 $\kappa$ 不够，必须校准配对变化？

UARCap 的 RQ1 关注的不是静态绝对错误率，而是**同一视频 $i$ 上外部证据从 $A0$（或 $A1$）增至 $A7$ 带来的配对事实增量**：
$$\Delta_i^{\text{ref}}(A7 - A0) = N_{\text{Contra\_ref}, i}(A7) - N_{\text{Contra\_ref}, i}(A0)$$
$$\Delta_i^{\text{human}}(A7 - A0) = N_{C, i}(A7) - N_{C, i}(A0)$$

定义视频 $i$ 的**配对差分测量偏置（Differential Measurement Bias）**：
$$B_i(A7 - A0) = \Delta_i^{\text{ref}}(A7 - A0) - \Delta_i^{\text{human}}(A7 - A0)$$

即使单条事实的总体 Cohen's $\kappa$ 很高（例如 0.80），只要参考代理在 `A7` 上系统性遗漏了外部检索注入的新增错词（使 $B_i < 0$ 显著偏负），或在 `A7` 较长字幕上制造假阳性参考冲突（使 $B_i > 0$ 显著偏正），最终得出的 `A7 vs A0` 风险结论就会发生方向歪曲或严重低估。

### 3.2 视频级配对变化验证四项指标

在 Pilot A2（30 条校准集，用于调优与初检）及后续正式 `E2-Val`（50–80 条独立验证集，零调参只读验证）上，必须逐视频计算：

1. **平均差分偏置 $\bar{B}(A7 - A0)$ 与 $\bar{B}(A7 - A1)$**（及视频聚类自助法 95% CI）：检验 95% CI 是否严重偏离 0，以及偏置幅度相对于真实变化幅度的占比；
2. **视频级配对变化相关性**：Spearman $\rho(\Delta_i^{\text{ref}}, \Delta_i^{\text{human}})$ 与 Pearson $r$（带视频级自助法 95% CI）；
3. **非零变化视频的方向一致率（Directional Concordance）**：在人工判定 $\Delta_i^{\text{human}} \neq 0$ 的视频中，自动代理给出同号变化（或至少不给反号变化）的比例；
4. **Bland–Altman 差值-均值趋势检验**：检验偏置 $B_i$ 是否随句子词数差 $\Delta \text{len}_i(A7 - A0)$ 或真实错误总数呈显著单调倾斜（防止长度偏置伪装成事实效应）。

### 3.3 硬性否决门控（触发即判 `EVALUATOR_UNRELIABLE`）

在 Pilot A2 校准定型后（并在 `E2-Val` 独立复核），若出现以下**任一红线**，即便总体原子分类 $\kappa$ 较高，**也禁止将参考代理作为主要风险评价器直接放行**：

- **红线 1（`A7` 系统性漏检偏置）**：`A7` 对真实视频错误的捕获率 $\text{Rec}_{\text{Err}\to\text{Contra}}(A7)$ 显著低于 `A0/A1`（降幅 $> 15\text{ pp}$），或 `A7` 的 `Unmentioned` 真实错误漏检率比 `A0` 高出 $> 10\text{ pp}$ 且导致 $\bar{B}(A7 - A0)$ 系统性低估真实风险增量超过真实变化幅度的 $40\%$；
- **红线 2（参考冲突假阳性失控）**：$\text{Prec}_{\text{Contra}\to\text{Err}} < 0.65$（即超过 1/3 的“参考冲突”并非真实视频错误）；
- **红线 3（配对变化追踪失效）**：视频级配对变化相关 $\rho(\Delta_i^{\text{ref}}, \Delta_i^{\text{human}}) < 0.45$，或在非零变化视频中出现 $> 20\%$ 的**反向误判**（人工判风险上升而代理判显著下降，反之亦然）；
- **红线 4（跨模型严重不对称偏置）**：在 Pilot A3 纳入现代 VLM 后，同一代理对现代 VLM 与 Video-LLaMA 的误差捕获率之差超过 $20\text{ pp}$，导致跨模型交互差（DiD）无法解释。

---

## 4. 视频聚类独立性与 50 条独立验证视频的样本量充分性论证

### 4.1 严禁伪独立扩样

假设 50 条视频 × 2 模型 × 3 臂 = 300 条字幕、约 900–1,500 条原子事实。同一条视频的 6 条字幕共享同一画面、同一参考文本和同一组检索候选，存在显著视频内相关（ICC > 0）。因此：
- **有效样本量基准是视频数 $N_{\text{vid}} = 50$**，而非 300 条字幕或 1,200 条原子事实；
- 所有置信区间均采用**以视频为单位的整群自助法（Cluster Bootstrap over Videos, 10,000 resamples）**。

### 4.2 $N_{\text{vid}} = 50$ 是否足够？定量精度推演

按视频级比例估算二项 Wilson 95% 置信区间半宽（未计视频内聚类膨胀前为下界）：

| 场景参数 | 若 $N_{\text{vid}} = 50$ | 若扩至 $N_{\text{vid}} = 80$ | 科学判断 |
|---|---|---|---|
| **配对均值偏置 $\bar{B}$ 精度**（假设 $\text{SD}(B_i) \approx 0.35$ 事实/视频） | $\text{SE} \approx 0.050$，95% CI 半宽 $\approx \pm 0.10$ | $\text{SE} \approx 0.039$，95% CI 半宽 $\approx \pm 0.08$ | $N=50$ 足以检出 $\ge 0.12\text{–}0.15$ 事实/视频的系统性差分偏置。 |
| **捕获率 $\text{Rec}_{\text{Err}\to\text{Contra}}$**（假设真实错误发生率 = **30%**，约含 15+ 条含错视频 / 30+ 错事实） | 95% CI 宽度约 $\pm 15\text{–}20\text{ pp}$ | 95% CI 宽度收窄至 $\pm 11\text{–}14\text{ pp}$ | $N=50$ 可作基线检验；接近门控阈值时建议扩至 80。 |
| **捕获率 $\text{Rec}_{\text{Err}\to\text{Contra}}$**（假设真实错误发生率仅 **8–10%**，仅 ~5 条含错视频） | 仅 ~5 个阳性视频，95% CI 极宽（$> \pm 30\text{ pp}$），**统计不足！** | 约 8–10 个阳性视频，需结合按代理分层抽样（见下） | **若 Pilot A2 实测错误率 $< 12\%$，50 条纯随机视频不足以精确估计错误捕获率！** |

### 4.3 结论与自适应分层验证规则

- **50 条独立验证视频仅作为基线验证下限，而非无条件充分保证。**
- **自适应扩张与分层加权机制**：
  1. 若 Pilot A2（30 条视频）实测 `A0/A1/A7` 合并后含真实视频错误的视频比例 $\ge 25\%$ 且差分偏置 CI 未跨越红线，则正式独立验证集 `E2-Val` 维持 **`N = 50` 条**（5 域均匀各 10 条）；
  2. 若 Pilot A2 实测真实错误视频比例 $< 25\%$，则 `E2-Val` 升级为 **`N = 80` 条**，并采用**事后不泄密的双层抽样**：50 条纯域分层随机样本（用于无偏估计总体指标与配对偏置 $\bar{B}$）+ 30 条从“代理判有变化或 `Unmentioned` 占比高”层中抽取的富集校验样本（通过 Horvitz–Thompson 逆概率加权合并，大幅压缩错误捕获率与漏检率的置信区间，而不污染无偏性）。

---

## 5. 本地跨家族开源裁判选型、校准冻结与核心事件预抽协议

1. **强制跨家族隔离（Anti-Self-Preference）**：
   - 若现代开源 VLM 选定为 `Qwen2.5-VL-7B-Instruct`，本地三值裁判**严禁使用 Qwen 家族模型**，必须选用非 Qwen 家族的强指令遵循开源模型（首选候选：`Mistral-Small-24B-Instruct-2501` BF16 约 48 GB 边界 / FP8-AWQ ~14 GB VRAM，或 `Gemma-2-27B-IT` AWQ/GPTQ ~16 GB VRAM，或 `Llama-3.1-8B-Instruct` / `Llama-3.3-70B-Instruct-AWQ` ~39 GB VRAM，在 A2 阶段于 RTX A6000 48GB 上实测显存与 JSON 格式稳定率后锁定唯一型号）；
   - 若现代 VLM 选定为 `LLaVA-Video-7B-Qwen2`，同样禁用 Qwen 家族作裁判；若选 `InternVL2.5-8B`（InternLM2.5 底座），则 Qwen / Llama / Mistral / Gemma 均可跨家族使用。
2. **严格校准-验证隔离**：
   - 仅允许在 **Pilot A2 的 30 条校准视频** 上调整三值 NLI 提示词、原子拆解颗粒度与 JSON 解析规则（最多不超过 2 轮定向迭代，严禁过拟合）；
   - A2 结束后立即以 SHA-256 锁定裁判权重版本、提示词文本、解码参数（`temperature=0.0`，确定性输出）与后处理代码；
   - **严禁依据 `E2-Val`（50–80 条独立验证集）或正式 `E2-Formal` 的结果回调裁判提示词！**
3. **`R_core_ref` 事前核心事件预抽规则**：
   - DREAM-1K 元数据本身已提供每条视频的 `events` 列表（通常 4–10 个动作事件）。为避免 9 词短字幕相对 10 个细节事件产生地板压缩，在运行任何待评视频模型之前，预先依据确定性规则从每条视频的参考描述/`events` 中抽取 **1–2 个主干核心事件**（优先主语跨镜头持续出现、处于主句位置、非边缘微表情/背景次动作的事件），并在 Pilot A2 30 条视频上由双人快速核验该核心子集是否确实对应视频最显著主事件。

---

## 6. E3 独立双盲单目标语义评价器验证规程

1. **双盲隔离设计**：
   - 针对 VidHalluc（及可选 `E3-Bridge`）的每条生成字幕 $c$，生成两条**完全独立、打乱顺序、去除一切上下文线索**的核验请求：
     - Request 1：`{"caption": c, "target_proposition": p_1}` → 输出 `expressed ∈ {0, 1}`, `negated ∈ {0, 1}`, `hedged ∈ {0, 1}`, `generic_only ∈ {0, 1}`, `evidence_span`；
     - Request 2：`{"caption": c, "target_proposition": p_2}` → 同上。
   - $(p_1, p_2)$ 为 $(f^+, f^-)$ 的随机排列；裁判看不到实验臂名、看不到干预句、看不到另一条竞争命题、也不知道哪条是真值。
2. **Pilot A3 全量人工对账**：
   - 在 Pilot A3 的 20 条 VidHalluc 视频（× 4 臂 × 2 模型 = 160 条字幕）中，对**全部 160 条字幕的 $(Y^+, Y^-)$** 进行人工盲标复核，计算 E3 语义裁判对 $Y^+$ 与 $Y^-$ 的精确率、召回率、Cohen's $\kappa$，以及最重要的——**在 `Con` 与 `Neu` 两臂之间是否存在差分误判偏置**（门控要求：$Y^+$、$Y^-$ 各自 $\kappa \ge 0.80$ 且 `Con` vs `Neu` 差分偏置 $|\bar{B}_{\text{E3}}| \le 0.03$）。
3. **正式 E3 抽样复核**：正式 E3（150–180 条视频）运行后，对随机 20% 视频及全部 `hedged=1` / `negated=1` 边界案例进行人工复核。

---

## 7. 若 DREAM-1K 参考代理验证失败时的有界回退协议

若 Pilot A2 结束后，DREAM-1K 三值参考代理在最多 2 轮合规提示词校准后仍触碰第 3.3 节任一红线（判定为 `EVALUATOR_UNRELIABLE`），则**不得强行进入正式 E2b，也不得无限扩张数据集清单**，仅允许在以下**两条有界备选路线**中二选一（或直接止损转方案 C）：

1. **备选路线 R1（已定备选集定向审查）**：仅限已审计过的 `ARGUS`（500 条高密人工描述）或 `VCapsBench`（严格过滤实体/动作 QA + 空字幕盲猜基线）中任选 1 个，做同等 30 条微型兼容性重验；严禁同时引入多个新数据集。
2. **备选路线 R2（有限规模直接人工看片主证）**：放弃大规模自动参考代理，转为在 DREAM-1K（或 MSR-VTT + DREAM-1K 合并）的 **120–150 条独立视频** 上开展精简双人直接看视频事实评测，聚焦核心对比 `A7 vs A0` + VidHalluc E3，同步缩减算力并明确向审稿人说明为何弃用失准自动代理。
3. **备选路线 R3（理性止损方案 C）**：若上述两条均超出用户可接受人工/时间投入，立即停止 Q2 升级，以冻结 `v0.8` 完整投稿包直投 SCI 三区。
