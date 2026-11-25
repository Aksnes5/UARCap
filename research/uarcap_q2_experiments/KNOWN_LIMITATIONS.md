# 已知边界与待服务器现场验证事项（KNOWN LIMITATIONS）

本文档严格区分四类状态，避免将离线工程完成与服务器实测或科学参数冻结混淆：

## 1. 已完成的离线代码（Windows 本地已全部实现）
- 真实阶段分发 CLI（`PILOT_A0/A1/A2/A3/FORMAL_B/E1`）、授权与前置收据状态机（`MOCK_PASS` 与 `REAL_RUNTIME_PASS` 严格隔离）。
- 真实 VideoUFO X-CLIP 检索器（含 16 中心帧预处理、百万级 mmap 索引、D 共识重排及跨环境 `xclip_worker.py` 子进程入口）。
- 冻结 Video-LLaMA 批量执行适配器（18 列 TSV、派生 `eval->train` 检查点绑定、只读前向 hook trace）。
- 现代视频生成器 **`Qwen3VLLMMultimodalAdapter`**（首选既有 BN `Qwen/Qwen3.8-27B-FP8`，备用既有 `Qwen/Qwen3.5-27B`，支持 vLLM 0.29.0 离线引擎 / 本地 HTTP / 跨环境子进程、16 帧输入、关闭 thinking 控制与实际输入 Trace；已取消默认下载 `Qwen2.5-VL-7B-Instruct`）。
- 跨家族文本事实裁判接口（首选既有 BN `Mistral Small 3.2 24B`，备用 `Gemma 3 27B`、`Llama 3.3 70B W4A16`、`gpt-oss-20b`、`Phi-4 14B`、`Llama 3.1 8B`，支持 vLLM / OpenAI-compat / Transformers 跨环境子进程）。
- A2 稳定 `claim_id` 严格对账与盲法包复用、A3 E3 双目标 + 跨模型 DREAM-1K 人工核验闭环、本地网页标注工具与一键双人仲裁 CLI。
- Formal B 分层概率抽样、Pilot/近重复/同来源隔离、无零值吞错的完整统计、两阶段 IPW 误差校正、真实 E3-Bridge 执行入口与 MSR-VTT 历史桥接。
- 一键数据准备（`stage_assets.py prepare` / `check-ready`）与共享 GPU 锁 / 显存释放封装。

## 2. 已通过的本地 CPU 测试（`59 passed, 0 failed, 0 skipped`）
- 50 条历史冻结 MSR-VTT 查询的 D 重排序字节级回归（`< 1e-12`）。
- 全套单元、端到端、故障注入、真实入口契约、fake subprocess/processor/vLLM 引擎测试、人工标注导出与仲裁闭环测试、Linux bash `bash -n` 语法与 LF 行尾检查。

## 3. 必须留待 BN 服务器真实运行时才能验证的事项（非离线代码缺陷）
1. **BN 现场路径与权限核实**：确认 `configs/bn_paths.local.yaml` 中历史冻结路径、既有 vLLM 0.29.0 Python 路径以及既有模型目录（`Qwen3.8-27B-FP8`、`Mistral Small 3.2 24B` 等）均存在可读。
2. **`Qwen3.8-27B-FP8` 本地视觉权重完整性与 RTX A6000 48GB 多模态显存实测**：实测 16 帧视频输入下 vLLM 0.29.0（FP8 block-128 / Marlin W8A16）的真实视觉编码加载、`chat_template_kwargs={"enable_thinking": False}` 现场行为、吞吐与峰值显存占用（若现场多模态受限则切换既有 `Qwen3.5-27B` 或调整 `multimodal_input_key="image"`）。
3. **冻结 Video-LLaMA (`Transformers 4.28.0`) 真实前向 Hook 现场验证**：确认 `launch_traced_{arm}.py` 在真实 `eval_evcap...py` 中的张量采集与显存占用无副作用。
4. **百万级 VideoUFO 索引与 X-CLIP 现场速度/显存验证**：确认 `xclip_worker.py` 在既有冻结环境中的真实分块检索耗时与 `.pt` 特征完整性。
5. **DREAM-1K（3.59 GiB gated）与 VidHalluc（15.60 GiB）真实数据获取与解码**：确认 HF gated token 连通性（或经 Windows 分卷导入）及真实视频 PyAV 解码通过率。
6. **既有跨家族裁判现场 JSON 遵循率与延迟**：在 Pilot A2 中实测 `Mistral Small 3.2 24B`（或备用既有模型）的格式合规率与推理速度。

## 4. 必须等待真实 Pilot A0–A3 结果才能冻结的科学参数
- Formal B 最终 `dream1k_formal_n`、`e2_val_human_n`、`vidhalluc_formal_n`。
- 最终冻结的跨家族文本裁判模型版本及 `frozen_judge_prompt_sha256`（由 Pilot A2 双人人工校准裁决）。
- 最终冻结的现代 VLM 版本（`Qwen3.8-27B-FP8` 或 `Qwen3.5-27B`，由 Pilot A3 实测与人工核验裁决）。
- 两阶段抽样误差校正作为主分析还是敏感性分析的最终定位、缺失率硬门槛以及 `E3-Bridge` 是否达到 `>= 25` 对自然冲突的启用条件。
