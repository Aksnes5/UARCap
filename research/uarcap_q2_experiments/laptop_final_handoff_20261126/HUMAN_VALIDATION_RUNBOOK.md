# 内部概率核验与双人标注

复用已经冻结的 60 视频分层概率样本和六个模型×预算条件。360 条原始记录含 **250 条同视频唯一字幕、1,442 个命题出现、783 个同视频逐字唯一命题**。每位标注员约需 5–9 小时，按每视频 5–9 分钟估算。完整观看长视频、修改复杂命题和仲裁会增加实际工时。两人时钟分别记录。旧包可能来自存在跨臂错配的构造路径，以 V3 重建包开始。

1. BN 在独立输出目录运行 `scripts/run_final_analysis.py --prepare-validation-package`。新版目录为 `final_analysis/validation_60_v3`。管理员持有映射文件。每位标注员只获得其任务 JSON、可读视频和帧，以及独立账号或独立保密工作目录。
2. 在各自回环地址运行 `scripts/launch_annotation_ui.py --bundle <该用户任务.json> --save-dir <隔离目录>`。两人在同一主机时用不同端口和操作系统账号。任务先播放视频并锁定核心事件，然后展示匿名字幕、16 帧和候选命题。可添加遗漏、拆分与合并，逐字相同的命题允许在同视频复用。整条字幕逐字一致时，参考终点可复用到其盲法 ID，原始臂权重仍在私有映射中。
3. 视觉命题审查记录支持、矛盾、不确定。书面参考终点是单独标注。依据冻结书面参考记录冲突原子事实整数数量及冻结核心事件是否被表达。没有提及与明确冲突分别记载。按视频完成后导出 JSON。缺失和不确定应保留在原始记录中，对仍不确定的正式终点请求仲裁。不能直接用视频标签替代书面参考终点。
4. 每位评审在独立阶段结束后提供自己确认的收据。工具读取 `annotator_id`、`review_export_sha256`、`independent_blind_review_attested`。使用 `scripts/build_reference_endpoint_import.py --manifest <正式清单> --key <私有键> --first <V1.json> --second <V2.json> --first-attestation <V1收据.json> --second-attestation <V2收据.json> --adjudicator-id <独立仲裁员> --out <私有目录>`。未解决分歧单列。仲裁文件的每行以 `model_id + query_id + arm` 标识，含两个 `endpoints` 及 `rationale`，通过 `--resolutions` 导入。
5. 工具产生四个 `uarcap_q2.human_primary_endpoints.v1` 文件，每个仅绑定一个模型与一个主要终点。分析时重复传入四个 `--human-annotations`。校验包括两份独立原始导出、签收文件、仲裁账本、两个模型各臂、真实纳入概率、参考和核心事件版本。散列用于记录完整性。评审身份及独立执业由管理者实际确认。
6. 原协议的 Pilot A2 校准与 Pilot A3 人工验证继续单独验收。参考终点已经有效仲裁仍需相应先决回执。Gemma/Llama 的机器分歧用于另行组织诊断与仲裁，主概率样本照原概率核验。若另抽分歧富集样本，按单独诊断报告，不将未登记纳入概率带入现有两阶段估计。E3 的全视频真值、原 16 帧可见性和 Neu 兼容性需要按 V2 被批准数量另建人工任务清单。

当前已生成真实盲法任务包，尚无任何真实人工标签、签入或仲裁。严禁公开映射、私有审阅文件和媒体。[样本与工作量](VALIDATION_SET_IDENTITY_RECEIPT.json)

## V2 统一门户

保留原 60 视频概率任务及既定权重。使用现有 Pilot A2 双盲包完成视觉校准。V2 加载 `scripts/q2_human_portal.py build`，用 `--internal` 各传两人原参考包，用 `--pilot-a2` 各传两人现有 Pilot A2 包。按同一顺序给两个 `--annotator`。用随机私有密钥文件 `--secret`，管理员独占 `PRIVATE_PORTAL_KEY.json`。可用多个 `--input` 加载不同独立批次。

`stage-e3 --module A3` 读取原 Pilot 候选账本和已存字幕，旧无效 Neu 留在历史。正式 E3 运行后，以 `E3_CANDIDATE` 及 `E3_BRIDGE` 从各自冻结候选、完整视频、16帧目录构建盲任务。三个角色每个各保留完整视频和帧可见性，界面中隐藏角色。以 `E3_OUTPUT` 从对应冻结包、真实生成收据和字幕文件选固定种子哈希顺序前 30 个查询，保留逐模型全臂双目标作为后设抽检。它不改变原概率样本或确认性检验族。

管理员为每个评审在隔离账号/目录和回环端口启动 `scripts/launch_annotation_ui.py --bundle <blind.json> --save-dir <isolated-dir> --port <unique-port> --access-token-file <private-token-file> --no-browser`。连接该人独有的 SSH 隧道和浏览器配置，以 `/?access=<token>` 换取会话 cookie。仅此人可访问相应门户。完整视频可以定位时间，16 帧任务仅显示对应图像，表达任务只提供字幕和目标句。禁止共享另一位的访问凭证及盲法键。

各自从门户点 Export JSON，附真实签收。运行 `q2_human_portal.py import --key <private> --review <first> --review <second> --attestation <first-receipt> --attestation <second-receipt> --out <private-review.json>`。未审、分歧和不确定保留在输出。仲裁员用 `--resolutions` 提交有独立签收及两份原始导出哈希的逐项判断。用原有参考终点导入脚本处理 60 视频，用 A2 校准入口处理其单独作用域。覆盖完成只表示评审齐备，校准及正式授权保持独立门槛。

已有 A3 归档为 20 个源查询、80 条字幕。排除旧 Neu 后 60 条可供表达诊断，对应每人 120 个目标判断，常规估时约 0.6 小时。正式 E3/Bridge 以实际合格数计，每个候选每人有三份全视频判断和三份实际 16 帧判断。两模型均完成时，表达抽检 30 查询最多每人 480 个判断，约 2.4 小时，完全相同的源、字幕、命题及作用域可复用。实际唯一数、缺失分支和工时由每次包的 `HUMAN_WORKLOAD.json` 输出。视频加载、回看和独立仲裁的实际时间另计。
