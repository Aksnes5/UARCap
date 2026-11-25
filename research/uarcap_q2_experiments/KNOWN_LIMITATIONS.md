# 已知边界与离线实现局限声明（KNOWN LIMITATIONS）

1. **未执行真实 GPU 推理与真实 BN 环境联调**：本轮全部 43 项测试均在 Windows 11 CPU 上利用 PyAV 合成动态视频与 Mock 后端完成，**严禁将 CPU 模拟测试通过等同于 RTX A6000 48GB 真实模型兼容或真实科学假设成立**。具体待现场实测项见 `SERVER_RUNTIME_VERIFICATION_PENDING.md`。
2. **X-CLIP 百万级 VideoUFO 在线检索桥接需在 BN 现场绑定**：离线代码已完成 `RetrieverAdapter` 接口、忠实 `D` 共识重排（`.strip()` + 全字幕平均成对 ROUGE-L）、`A0/A1/A3/A7` 线格式输入构造及 EVCap TSV 打包；百万级 VideoUFO 实际向量索引路径与 X-CLIP 特征抽取脚本需在 BN 现场通过 `configs/bn_paths.local.yaml` 绑定既有 `/home/bn/videoufo_research` 检索实现后运行 A1。
3. **现代 VLM 适配器以 `Qwen2.5-VL-7B-Instruct` 为首选完整实现**：依据协议“不为了工程完整而同时实现三套大型模型”原则，已完成 `Qwen25VLAdapter` 完整代码与隔离环境配置；若 BN A3 timed smoke 实测需切换至备选 `LLaVA-Video-7B-Qwen2` 或 `InternVL2.5-8B`，仅需新增约 60 行对应模型加载子类即可复用全部 prompt、trace、runner 与评测链路。
4. **DREAM-1K 缺乏公开原始来源 ID**：公开元信息仅提供不透明整数 `vid`（1–1000），故本工程严格禁止宣称“DREAM-1K 精确源级零重叠”，采用两阶段近重复筛查+人工复核保守排除+残余风险透明声明。
5. **待试点定容的科学参数尚未冻结**：`formal_b_candidate.yaml` 中的正式样本量、唯一裁判权重、冻结提示词、现代模型选型与 `E3-Bridge` 启停均保持 `PENDING_PILOT_FREEZE`，必须依据 Pilot A0–A3 真实数据裁决。
