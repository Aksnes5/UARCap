# 内部概率核验与双人标注

复用已经冻结的 60 视频分层概率样本和六个模型×预算条件。360 条原始记录含 **250 条同视频唯一字幕、1,442 个命题出现、783 个同视频逐字唯一命题**。每位标注员约需 5–9 小时，按每视频 5–9 分钟估算。完整观看长视频、修改复杂命题和仲裁会增加实际工时。两人时钟分别记录。旧包可能来自存在跨臂错配的构造路径，以 V3 重建包开始。

1. BN 在独立输出目录运行 `scripts/run_final_analysis.py --prepare-validation-package`。新版目录为 `final_analysis/validation_60_v3`。管理员持有映射文件。每位标注员只获得其任务 JSON、可读视频和帧，以及独立账号或独立保密工作目录。
2. 在各自回环地址运行 `scripts/launch_annotation_ui.py --bundle <该用户任务.json> --save-dir <隔离目录>`。两人在同一主机时用不同端口和操作系统账号。任务先播放视频并锁定核心事件，然后展示匿名字幕、16 帧和候选命题。可添加遗漏、拆分与合并，逐字相同的命题允许在同视频复用。整条字幕逐字一致时，参考终点可复用到其盲法 ID，原始臂权重仍在私有映射中。
3. 视觉命题审查记录支持、矛盾、不确定。书面参考终点是单独标注。依据冻结书面参考记录冲突原子事实整数数量及冻结核心事件是否被表达。没有提及与明确冲突分别记载。按视频完成后导出 JSON。缺失和不确定应保留在原始记录中，对仍不确定的正式终点请求仲裁。不能直接用视频标签替代书面参考终点。
4. 每位评审在独立阶段结束后提供自己确认的收据。工具读取 `annotator_id`、`review_export_sha256`、`independent_blind_review_attested`。使用 `scripts/build_reference_endpoint_import.py --manifest <正式清单> --key <私有键> --first <V1.json> --second <V2.json> --first-attestation <V1收据.json> --second-attestation <V2收据.json> --adjudicator-id <独立仲裁员> --out <私有目录>`。未解决分歧单列。仲裁文件的每行以 `model_id + query_id + arm` 标识，含两个 `endpoints` 及 `rationale`，通过 `--resolutions` 导入。
5. 工具产生四个 `uarcap_q2.human_primary_endpoints.v1` 文件，每个仅绑定一个模型与一个主要终点。分析时重复传入四个 `--human-annotations`。校验包括两份独立原始导出、签收文件、仲裁账本、两个模型各臂、真实纳入概率、参考和核心事件版本。散列用于记录完整性。评审身份及独立执业由管理者实际确认。
6. 原协议的 Pilot A2 校准与 Pilot A3 人工验证继续单独验收。参考终点已经有效仲裁仍需相应先决回执。Gemma/Llama 的机器分歧用于另行组织诊断与仲裁，主概率样本照原概率核验。若另抽分歧富集样本，按单独诊断报告，不将未登记纳入概率带入现有两阶段估计。E3 的全视频真值、原 16 帧可见性和 Neu 兼容性需要按 V2 被批准数量另建人工任务清单。

当前已生成真实盲法任务包，尚无任何真实人工标签、签入或仲裁。严禁公开映射、私有审阅文件和媒体。[样本与工作量](VALIDATION_SET_IDENTITY_RECEIPT.json)
