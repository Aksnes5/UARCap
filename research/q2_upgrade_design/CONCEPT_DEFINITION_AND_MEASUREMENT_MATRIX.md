# 附录 B：核心概念定义、可观测性与必要性核查矩阵（CONCEPT_DEFINITION_AND_MEASUREMENT_MATRIX）

> **文稿状态**：`DESIGN_FOR_DISCUSSION`｜配套主稿：`UARCap_Q2_统一科研框架与实验设计讨论稿_v1.0.md`  
> **设计原则**：每个保留概念必须具备清晰操作性定义、独立可观测变量与明确的假设支撑；与已有文献或其余概念高度重叠、或无法可靠测量的构念一律删除或降级。

## 一、保留并纳入统一框架的核心概念

| 中文名称 | 英文名称 | 数学 / 操作性定义 | 输入与输出 | 可观测变量 | 是否需人工标注 | 是否需参考答案 | 推理前是否可获得 | 与已有定义的区别 | 支持的假设 | 必要性判定 |
|---|---|---|---|---|---|---|---|---|---|---|
| **四类原子视觉命题** | Compact Four-Facet Atomic Visual Claims, $\mathcal{A}(y)$ | 将自由文本描述拆解为最小可核验命题，归入 `{Entity, Attribute, Action-Role, Temporal-Outcome}` 四类 | 输入：单句字幕 $y$；输出：命题集合 $\{a_1,\dots,a_m\}$ | 拆解后的规范化原子命题列表及类别标签 | 是（主黄金集）；可由经校准的 VLM 辅助扩展 | **否**（以原始查询视频画面为真值） | 仅候选字幕 $c_i$ 推理前可得；生成结果 $y$ 推理后可得 | 比“单一动作相容性”更完整，比 10+ 类事件本体可标注性更高，与 FactVC/VidHalluc/ARGUS 对齐 | H1, H2, H3, H4 | **必要**（全部事实性分析的最小原子单位） |
| **三值视频视觉蕴含** | Three-Valued Video Visual Entailment, $\phi(a \mid v)$ | 相对查询视频 $v$ 全片画面判定命题真伪：$\phi(a \mid v) \in \{\mathrm{Sup}, \mathrm{Con}, \mathrm{Unc}\}$；严禁将未在参考句提及或 `Unc` 混入 `Con` | 输入：$(a, v)$；输出：三值标签 | 盲法双人判定标签及仲裁结果（要求 $\alpha \ge 0.70$） | 是（主黄金集双盲）；辅助集需校准报告 $\kappa$ | **否** | 否（针对 $y$）；针对候选 $c_i$ 需视觉核验 | 明确区分候选对其来源视频 $\tilde{v}_i$ 的保真度与对当前查询视频 $v$ 的事实相容性 | H1, H2, H4 | **必要**（防止将不可核验细节误判为幻觉） |
| **二维净证据效用** | Net Evidence Utility Vector, $(\Delta U_{\text{lex}}, \Delta U_{\text{fact}})$ | 相对同一冻结生成器零外部证据工作点 $y(v,\emptyset)$ 的 CIDEr 差与可核验事实精确率差 | 输入：$y(v,S), y(v,\emptyset), v, R_v$；输出：实数对 | PTB CIDEr 差值与 $\text{FactPrec}$ 差值 | $\Delta U_{\text{lex}}$ 否；$\Delta U_{\text{fact}}$ 是 | $\Delta U_{\text{lex}}$ 是；$\Delta U_{\text{fact}}$ 否 | **否**（为事后评价目标，严禁部署时作已知量） | 将 `v0.8` 的单维 CIDEr 净效用扩展为词面×事实双轴，避免 CIDEr 与语义脱节 | H1, H3, H4 | **必要**（连接 `v0.8` 遗产与 Q2 事实升级） |
| **条件边际效用** | Conditional Marginal Utility, $\Delta_U(e \mid v, S)$ | 在已提供证据子集 $S$ 条件下追加候选 $e$ 产生的效用变化：$U(y(v,S\cup\{e\})) - U(y(v,S))$ | 输入：$(v, S, e)$；输出：效用增量 | 前缀扩展对比（如 $k=1\to 3\to 7$、$B3 \to B7$）的配对差 | 依所选 $U$ 而定 | 依所选 $U$ 而定 | **否**（部署时仅能用代理估计） | 强调集合依赖性，不默认加法性、单调性或次模性 | H1, H3 | **必要**（形式化“为何加多反伤”的集合现象） |
| **四元对称事实转移** | Retrieval-Induced Factual Transition Quadruple, $(\mathrm{Corr}, \mathrm{Gain}, \mathrm{Contam}, \mathrm{Omit})$ | 比较 $y(v,\emptyset)$ 与 $y(v,S)$ 的原子命题变化：事实纠错、真实信息增益、事实污染（含证据对齐子集 $\mathrm{Contam}_{\text{align}}$）、原有正确事实遗漏 | 输入：对齐后的 $\mathcal{A}(y_0), \mathcal{A}(y_S), \phi(\cdot\mid v), S$；输出：每视频 4 维计数 | 盲法核验后的视频级计数与发生率 | 是（核心黄金集） | **否** | **否** | 克服 ARGUS/CapQuiz 仅测单模型静态幻觉的局限，对称刻画检索的双向事实效应 | H1, H3, H4 | **核心必要**（Q2 升级最关键的理论创新） |
| **目标事实定向污染效应** | Targeted Factual Contamination Effect, $\mathrm{TFCE}$ | 视觉与句长锁定下，冲突臂相对中性对照臂诱发靶错误事实 $f^-$ 生成的风险差：$\mathbb{P}(f^-\in y_{\text{Con}}) - \mathbb{P}(f^-\in y_{\text{Neu}})$（辅以保真损失 $\mathrm{TFFL}$） | 输入：三臂最小配对干预输出 $(y_{\text{Sup}}, y_{\text{Con}}, y_{\text{Neu}})$；输出：配对风险差 | 二元靶事实采纳/保留判定 | 是（靶核验 + 盲审） | **否** | **否** | 彻底脱离观察性共现与历史 `HardWrong` 词面重叠代理，提供行为级因果识别 | H2, H4 | **核心必要**（解决因果归因瓶颈） |
| **受约束证据子集决策空间与效用—风险前沿** | Constrained Evidence Decision Space $\mathcal{S}(C)$ & Utility–Risk Frontier | 在成本上界 $\tau$ 下从 $C(v)$ 选择子集 $S \subseteq C(v)$（含 $S=\emptyset$），将所有策略映射至 $(R(\pi), U(\pi))$ 帕累托平面 | 输入：$(v, C(v), \tau)$ 与推理前代理特征 $\mathbf{x}_{\text{pre}}$；输出：选定子集 $S$ 及前沿坐标 | 推理前代理特征、算力成本（调用数/延迟/显存）与事后前沿指标 | 评估前沿需人工黄金集 | 词面轴需；事实轴不需 | 决策输入 $\mathbf{x}_{\text{pre}}$ **推理前可获得** | 将过滤、选 $k$、早停与零证据统一为同一受约束决策问题，避免模块堆砌 | H3, H4 | **必要**（提供统一部署视角与公平比较基准） |

## 二、经必要性审查后主动剔除或降级的概念

| 原候选概念 | 处理决定 | 剔除 / 降级理由 |
|---|---|---|
| **独立“动作相容性”概念** | **剔除（合并吸收）** | 仅覆盖动词/动作，遗漏高频实体、属性与论元角色冲突；已自然吸收为四类原子命题中的 `Action-Role` 子维。 |
| **独立的“检索拒绝模块 / 冲突过滤模块 / 早停模块”** | **剔除（统一收拢）** | 分别命名易造成概念堆砌并与成熟 gating/SURf/Adaptive-$k$ 重叠；统一收拢为子集决策空间 $S \subseteq C$（允许 $S=\emptyset$）。 |
| **内部注意力分散 / 神经级文本整合过载** | **剔除出核心主张** | 当前黑盒行为干预不观测或干预内部注意力权重，无法识别神经机制；诚实限定为行为级输入—输出因果效应，与 MAD-RAG（ICML 2026）清晰划界。 |
| **朴素事后单指标 CIDEr 最优预算上界** | **降级（仅保留劈半校验版）** | 单句 CIDEr 方差远大于平均效应，事后取 $\max_k$ 存在严重胜者诅咒与噪声膨胀；仅在 E1 中以“20 参考句劈半交叉验证上界 + 置换零基线”形式作为前置门控诊断，严禁作为立项依据。 |
