# Pilot A2 双人盲法标注操作指南（CaReBench E2）

**当前状态：`PILOT_A2_HUMAN_VALIDATION_PENDING`。** 自动事实评价已完成，但尚未经过人工校准；
在两位标注员独立完成并仲裁之前，任何自动标签都不得写成“准确率/幻觉率”或科学结论。

## 1. 盲法规则（必须遵守）

- 两位标注员 **A1、A2 各自独立**，不交流、不查看对方的标注结果。
- 界面中 **不显示** 实验臂（A0/A1/A7）、模型名、检索证据、自动评价器的标签。
- **不要打开** `blind_key_e2_DO_NOT_OPEN.json`（这是解盲映射，只有实验负责人/仲裁时使用）。
- 标注以 **视频本身** 为准；参考字幕/自动事实都只是待核验的候选，不是 Gold。
- 同一视频内完全相同的字幕或原子命题会出现“reused suggestion”提示，可以直接复用同一判断；
  这不会改变实验条件或统计配对。

## 2. 打开标注网页（远程服务器）

服务只绑定 `127.0.0.1`，请在你的电脑上建立 SSH 隧道后访问。

### 2.1 启动两个标注服务（在服务器上）

```bash
cd /home/bn/uarcap_q2_carebench/research/uarcap_q2_experiments
bash scripts/launch_pilot_a2_annotation.sh
```

输出：
- Annotator A1 → `http://127.0.0.1:8765/`
- Annotator A2 → `http://127.0.0.1:8766/`

进度自动保存到：
`/home/bn/uarcap_q2_runtime/outputs/pilot_a2/pilot_a2_carebench_eval_v1/annotation_saves/annotator_A1/`
`/home/bn/uarcap_q2_runtime/outputs/pilot_a2/pilot_a2_carebench_eval_v1/annotation_saves/annotator_A2/`

### 2.2 在你自己的电脑上建立隧道

Linux / macOS / Windows 10+（OpenSSH）：

```bash
ssh -N -L 8765:127.0.0.1:8765 -L 8766:127.0.0.1:8766 bn@172.20.72.8
```

（若你平时走跳板机，请加 `-J 跳板用户名@跳板地址`；若用内网地址，可把 `172.20.72.8` 换成
`192.168.31.127`。保持该终端窗口打开；`-N` 表示只转发端口、不登录 shell。）

PuTTY（Windows）：Connection → SSH → Tunnels，添加
`L8765 127.0.0.1:8765` 与 `L8766 127.0.0.1:8766`，再连接服务器。

然后在浏览器打开：

- 标注员 A1：`http://127.0.0.1:8765/`
- 标注员 A2：`http://127.0.0.1:8766/`

推荐 **Chrome / Edge / Chromium**。CaReBench 视频为 H.264 或 VP9-in-MP4，Safari 对部分 VP9 文件
可能无法播放。

## 3. 标注流程

每个视频是一个任务，共 30 个任务；每个任务内包含 3 条匿名字幕（共 90 条）。

### Phase 1：先只看视频，写核心事件

1. 播放（默认静音）视频；可拖动、重播。
2. 在 “Phase 1: Independent Core Events” 中写 **1–2 条主要核心事件**（每行一条）。
3. 点击 **Confirm Core Events & Unlock Captions**。在此之前，右侧字幕是锁定的。
4. 检查 16 帧预览：点击 **Show/Hide Sampled Frames**，确认帧覆盖了视频的关键过程；
   选择 **Sufficient** 或 **Insufficient / misaligned**（必选其一）。

### Phase 2：逐条核验匿名字幕

对每条字幕中的每个原子命题选择一个标签：

| 标签 | 含义 |
| --- | --- |
| **Video-Supported** | 视频中真实发生/可见，且被该命题支持 |
| **Video-Contradicted** | 视频明确显示该命题是错的（对象、动作、关系、数量、方向等冲突） |
| **Video-Unverifiable** | 视频看不清/被遮挡/信息不足，无法确认（**不等于错误**） |
| **Non-factual** | 不是事实陈述（主观评价、比喻、元话语等） |

补充操作：

- **+ Add Omitted Claim**：字幕遗漏了视频中的重要事实时补录，并给标签。
- **Split**：一个命题包含多个可独立判断的事实（如“女子拿起红杯子并放在桌上”）时拆分。
- **Merge Claims**：多个命题实际是同一事实时合并。
- 视频级备注：可在 “Optional video-level notes” 记录遮挡、剪辑、画质等问题。
- 完成后点击 **Mark Video Complete & Next →**。

### 保存、恢复、导出

- 每次点击都会原子保存；刷新页面/重启服务后自动从第一个未完成任务继续。
- 右上角 **Export JSON / JSONL / CSV** 导出当前标注员的全部记录。
- 标注结束后，请把 **JSON 导出文件** 交给实验负责人（不要交 `progress.json` 之外的原始目录）。

## 4. 典型完整示例（示意）

视频：`V001_xxxxxx`（匿名 ID）

1. 静音观看：女子坐在梳妆台前，用刷子给眉毛上妆。
2. Phase 1 核心事件：
   - `A woman is applying makeup to her eyebrows with a brush.`
   - `She is sitting in front of a mirror.`
3. 16 帧预览：关键动作（取刷子、上妆、照镜子）均可见 → 选 **Sufficient**。
4. Phase 2 字幕：`a woman is applying makeup to her face`
   - 命题 “a woman is applying makeup” → **Video-Supported**
   - 命题 “to her face” → **Video-Supported**（眉毛属于面部）
5. 若字幕是 `a woman is cutting her hair`
   - “cutting her hair” → **Video-Contradicted**（视频是化妆，不是剪发）
6. 若字幕提到 `in a bathroom` 但视频只看到梳妆台、无法确认房间 → **Video-Unverifiable**。
7. 若字幕是 `a beautiful morning routine` → **Non-factual**（主观评价）。
8. 补录遗漏事实：`She holds a makeup brush.` → **Video-Supported**。
9. 点击 **Mark Video Complete & Next →**。

## 5. 预计工作量

- 共 **30 个视频任务 / 90 张字幕卡**。
- 同一视频内完全相同的字幕只有 **50 条唯一字幕**（40 张重复卡可直接复用判断）。
- 自动拆分的原子命题共 **248 条**，其中同一视频内唯一命题 **119 条**；约 **129 条标签**可通过
  盲法复用提示直接确认，不需要重新判断。
- 预计每位标注员 **81–143 分钟**（按每个视频 90–150 秒观看 + 30–60 秒核心事件、每条唯一命题
  8–15 秒、每张唯一字幕 6–10 秒估算）。熟练后通常接近下限。
- 两位标注员独立完成；不要把“模型自动标签”当作自己的判断。

## 5.1 真实视频示例（来自本轮盲法包）

视频匿名 ID 形如 `V0xx_xxxxxx`，对应 `carebench_v_00002093_8`（**解盲信息仅供说明，界面不显示**）：

- Phase 1 核心事件可写：
  - `A woman is applying makeup to her eyebrows with a brush.`
  - `She is sitting in front of a mirror.`
- Phase 2 字幕：`a woman is applying makeup to her face`
  - `A woman exists.` → **Video-Supported**
  - `The woman is applying makeup.` → **Video-Supported**
  - `The makeup is being applied to the woman's face.` → **Video-Supported**
- 若字幕为 `a woman is applying makeup to her face with a black eyeliner`
  - “black eyeliner” 若视频中可见 → **Video-Supported**；若只见刷子/无法确认颜色 → **Video-Unverifiable**。
- 若字幕为 `a woman is talking about her dog`（该视频实际是化妆）
  - “talking about her dog” → **Video-Contradicted**。
- 16 帧预览：化妆动作的关键帧应清晰可见 → 选 **Sufficient**；若关键动作帧缺失 → **Insufficient**。

## 6. 仲裁与回灌（实验负责人执行）

1. 收集两位标注员的 JSON 导出。
2. 解盲与分歧提取：

```bash
cd /home/bn/uarcap_q2_carebench/research/uarcap_q2_experiments
python scripts/adjudicate_annotations.py --task e2 \
  --ann1 <A1_export.json> --ann2 <A2_export.json> \
  --blind-key /home/bn/uarcap_q2_runtime/outputs/pilot_a2/pilot_a2_carebench_eval_v1/blind_annotation_package/blind_key_e2_DO_NOT_OPEN.json \
  --out-dir /home/bn/uarcap_q2_runtime/outputs/pilot_a2/pilot_a2_carebench_eval_v1/adjudication
```

3. 对分歧进行第三方仲裁，生成 `adjudicated_e2.json`。
4. 回灌 Pilot A2（复用已冻结的自动评价与盲法映射，不重跑生成）：

```bash
bash scripts/run_pilot_a2.sh \
  --judge-model-id RedHatAI/gemma-3-27b-it-FP8-dynamic \
  --judge-family gemma \
  --judge-backend-mode vllm_offline \
  --judge-model-path /media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Jnh_rag/models_campaign/M04_gemma3_27b \
  --human-annotations /path/to/adjudicated_e2.json
```

> 回灌步骤会检查 `a2_bundle_manifest.json` 中的 A1 预测 SHA-256 与盲法包 SHA-256；
> 若不一致会拒绝，绝不会静默错配。

## 7. 故障排查

- **页面打不开**：确认隧道终端仍在运行；`ss -tlnp | grep -E '8765|8766'` 应看到两个监听。
- **视频不播放**：换 Chrome/Edge；确认不是 Safari；视频文件在
  `/home/bn/uarcap_q2_runtime/data/carebench/videos/`。
- **16 帧不显示**：点击 Show/Hide；图片来自
  `.../blind_annotation_package/frames/<query_id>/frame_00.jpg … frame_15.jpg`。
- **标注丢失**：检查 `annotation_saves/annotator_<ID>/progress.json`；服务每次点击都会原子写入。
- **两个标注员互相看到结果**：不应发生；每个 bundle 独立、每个 save 目录独立，且服务仅绑定本机。
