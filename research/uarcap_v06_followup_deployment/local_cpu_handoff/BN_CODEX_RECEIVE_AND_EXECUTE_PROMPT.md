# 给 BN Codex 的接收及执行指令

你在 BN Ubuntu 上接收 UARCap v0.6 两项冻结补实验。**当前授权限于下载、接收、只读核查及在项目可写盘准备部署。GPU smoke 和正式运行分别需要用户明确授权。无 GPU 授权时完成第 1–3 步即停止。** 本文件后面的运行步骤供获得相应授权后使用。不得启动排队任务、抢占 RTX A6000、发信号停止其他进程或改动原项目 GPU 锁。

## 1. 获取冻结协议和输入

- 仓库 `Aksnes5/UARCap`，交接分支 `codex/uarcap-v06-local-cpu-handoff-20261124`。
- 下载以传输提交 `bc737176ea700492402be63480a3c6e76f05be2e` 为准。论文冻结 `7c5619ee2e8f4df716414d92af73f5e503499cbd`，实验协议 `ee7d3b9876dea94a1ee9eeb2626b92102c85d458`，部署预审 `94f5529f8a6e4424d2af207df42482cf87c7ce1b`，BN 资产回执 `d184360cb1cda832dbaa2d6ed9f46fb4aaa186e6`。
- 读取固定版本的 `research/uarcap_v06_followup_experiments/`、`EXECUTION_AND_ROLLBACK_PLAN.md`、`MINIMAL_GPU_RUN_MATRIX.json`、`local_cpu_handoff/`。源代码在部署工作区内固定 Git 版本，下载能力与服务器路径核验各自记录。
- 输入 ZIP **2,861,489 字节**，SHA-256 **`93289136546fc2d96d2f457cc9dec57c50fde1b2c517f735f23df0a1dfc63128`**。八份 JSON 和六份视觉 ID 清单是 Windows 已准备的原始字节。A1/A7 用 BN 已核实的 FULL D CAP/SIM。不要重新生成 A0 或 HQ/LQ。

在已核实的项目可写盘上新建目录。以下 Git HTTPS 方式已在 Windows 从独立空仓库验证，会按需取包和两个小文件。若当前 BN 回执的 manifest 路径已变化，先找现有对应文件，并用冻结哈希验证再替换参数。

```bash
set -euo pipefail
project=/home/bn/videoufo_research
payload=bc737176ea700492402be63480a3c6e76f05be2e
prefix=research/uarcap_v06_followup_deployment/local_cpu_handoff
stage=$(mktemp -d -- "$project/uarcap-v06-receive.XXXXXXXX")
cd "$stage"
git init --bare transfer.git
git -C transfer.git remote add origin https://github.com/Aksnes5/UARCap.git
git -C transfer.git config remote.origin.promisor true
git -C transfer.git config remote.origin.partialclonefilter blob:none
git -C transfer.git fetch --depth=1 --filter=blob:none origin "$payload"
test "$(git -C transfer.git rev-parse FETCH_HEAD)" = "$payload"
git -C transfer.git cat-file blob "$payload:$prefix/transfer/UARCap_V06_FOLLOWUP_BN_INPUTS.zip" > UARCap_V06_FOLLOWUP_BN_INPUTS.zip
git -C transfer.git cat-file blob "$payload:$prefix/GITHUB_INPUT_TRANSFER_MANIFEST.json" > GITHUB_INPUT_TRANSFER_MANIFEST.json
git -C transfer.git cat-file blob "$payload:$prefix/receive_bn_inputs.py" > receive_bn_inputs.py
test "$(stat -c %s UARCap_V06_FOLLOWUP_BN_INPUTS.zip)" = 2861489
sha256sum --check - <<'SHA256'
93289136546fc2d96d2f457cc9dec57c50fde1b2c517f735f23df0a1dfc63128  UARCap_V06_FOLLOWUP_BN_INPUTS.zip
178e3dfcbf207f9527be4ed750044357a5e9513cc0e2971c3a7c8e31d00aaf2e  GITHUB_INPUT_TRANSFER_MANIFEST.json
08a533f228a8df58dbd72bcee4a3caadfaad3cda93dc71917bc2bd6657c403b1  receive_bn_inputs.py
SHA256
python3 -B ./receive_bn_inputs.py \
  --zip UARCap_V06_FOLLOWUP_BN_INPUTS.zip \
  --manifest GITHUB_INPUT_TRANSFER_MANIFEST.json \
  --full-ids "$project/manifests/msrvtt_test2990.tsv" \
  --test-ids "$project/manifests/msrvtt_untouched2690.tsv" \
  --out "$stage/inputs"
printf 'Verified staging directory: %s\n' "$stage/inputs"
```

脚本须通过全部成员的大小和 SHA-256、八个对象的冻结规范化哈希、2,990/2,690 的 ID 和顺序及 B 的 2,618 合格名单。输出路径必须是新的。记录绝对路径与包、八个文件的哈希。公开 GitHub 的网络放行与下载成功要由 BN 实际确认。HTTPS 失败时可在该临时仓库把 `origin` 换成既有、已验证的 GitHub SSH 连接并重做接收。两个通道都不可达时保留错误并暂停，无需改网络配置。不要将接收目录或输出放在只读 Display 挂载。

## 2. 增量只读核验

- 复用已提交的八份 BN 资产回执。与拟运行的工作目录、shell 及解释器启动参数对齐，用 `probe_runtime_metadata.py` 记录入口、真实路径、Python/PyTorch/torchvision/Transformers/timm、图像处理及 tokenizer 栈和缓存解析。保留 Display venv 的解释器入口。该 venv 借用 TTP 的 PyTorch，直接运行其 Conda 链接目标会改变包解析。记录并请负责人确认 BN 3.9.13 / 2.1.2+cu121 相对归档的偏差。无需重装。
- 确认现有 checkpoint SHA、Vicuna shard/index/tokenizer 清单和 EVA/BLIP2/Video-LLaMA 真实路径、可读性及 offline cache。先沿用已核实证据检查变化和部署路径。只针对缺项补查，无需全量重哈希大权重。
- 用现有测试 YAML 和 BN 回执核实图像、caption、label TSV 和 `.lineidx` 路径、摘要及文件状态。image TSV 为回执所列的 `test_16frames_img_size224.img.tsv`（1,378,294,858 字节），仍需可信内容 SHA-256。先找已有可绑定摘要，若无，确认低负载及 I/O 许可后在 BN 就地顺序读取一次，记录规范路径、字节数和哈希前后纳秒 mtime。offset SHA、文件 metadata 或样本指纹不能代填内容 SHA。
- 复用 A7 特征路径完整性收据，其中包含 B 的三组共享 ID 集。对实际部署所选缓存目录检查相关文件可读性，在真实 smoke 用 tensor trace 检查解码和视觉内容。无需复制或重建缓存。
- 刷新项目可写盘剩余空间。仅用 `nvidia-smi` 管理查询核实设备 UUID、空闲 VRAM、计算进程及共享锁状态，保持零 CUDA context。开始真正执行前在同一个共享锁下再次检查占用。发现现有任务或锁持有人即暂停，禁止修改锁或处理现有进程。

## 3. 填写部署配置并 dry-run

从冻结版本建立干净的隔离推理树。原项目 checkout 与已有 `outputs/` 均保留。按实际读取得到的值写部署 JSON，参照 `EXECUTION_AND_ROLLBACK_PLAN.md`，避免复制示意值：

| 字段组 | 需要绑定 |
| --- | --- |
| 代码与路径 | 版本 SHA，`--evcap-root` 指向隔离推理树，`original_evcap_root` 指向历史推理树。核对六个冻结源码和九个依赖 digest。实际 launcher 使用 `deploy_guard.py` 与 Python 3.9 适配的 trace 层。 |
| Dataset | `test_yaml`、`test_yaml_sha256`、三组 `frame_receipts`（`role,path,size,sha256,lineidx_sha256,mtime_ns,digest_status`）。匹配已有 test2990 顺序与实际读取文件。 |
| 模型和运行环境 | `canonical_checkpoint`、`checkpoint_name`、`checkpoint_sha256`、`llm_model`、`llm_manifest`/SHA、`python_environment`/SHA，以及 `offline_cache_dirs` 的 `TORCH_HOME`、`HUGGINGFACE_HUB_CACHE`。纪要需包含实际 venv 入口、包版本及路径。原 checkpoint 用各 arm 独立软链接，目标为经核实原文件。 |
| 输入 | `artifacts` 使用 `MINIMAL_GPU_RUN_MATRIX.json` 中的键，指向八份接收 JSON 和已有 `frozen_D_cap`、`frozen_D_sims`，同时绑定 `B_visual_contract`。BN 用真实 `video_feature_dir`。 `--archive-scratch` 指向接收目录 `$stage/inputs`，其中有六份 `expected_feature_ids_*.txt`。 |
| 设备和输出 | `device=cuda:0` 对应实际核实的 `gpu_uuid`，绑定原 `shared_gpu_lock`。自定一个不含 `eval` 的 `run_id`，同一 cohort 贯穿各阶段，`attempt` 从 1 起。输出位于隔离推理树 `outputs/eval_v06_<smoke-or-full>_<run_id>_attempt<attempt>_<ARM>`，checkpoint slot 用同名 `train_v06_...`。输出路径、收据和 trace 相互绑定。 |
| 执行门槛 | 准备期不填 GPU 授权值。`server_data_ready` 只在实际资产核对完成后置真。后续用 Windows 校验报告填 `smoke_report`（按 A/B）和 `A_full_report`。 |

定义 `py`（以上核实的 venv 入口）、`guard`（固定提交的 `deploy_guard.py` 绝对路径）、`evcap_root`（隔离推理树）和 `config`（该次部署 JSON）。对 A0、A1、A7、B3、B7HQ、B7LQ 分别运行纯 dry-run，检查 argv、独立输出槽与 checkpoint slot。下面仅打印 A0 计划：

```bash
"$py" -B "$guard" --evcap-root "$evcap_root" --deployment "$config"   --archive-scratch "$stage/inputs" --arm A0
```

向用户回报解包、缺口、配置和当前空闲状态。**当前任务到此停止。接收或 dry-run 成功不授予任何 GPU 权限。**

## 4. A smoke（仅获单独 GPU smoke 授权后）

使用实际授权编号并按授权 scope 插入部署收据。以下命令中的 `approved_smoke_id` 必须是对应用户授权，按顺序只运行 A0、A1、A7，逐 arm 检查状态：

```bash
"$py" -B "$guard" --evcap-root "$evcap_root" --deployment "$config"   --archive-scratch "$stage/inputs" --arm "$arm" --mode smoke --gpu-approval-id "$approved_smoke_id"
```

每臂只跑冻结顺序前八个查询。逐查询核对实际 checkpoint、prompt 骨架和渲染/掩码、Q-Former 模块状态、RNG、查询帧与投影、CUDA dtype、日志和可用峰值显存。A0 的文本和外部视觉加载均为零，视觉聚合为零，并保留两个 `<VideoHere>` 槽及冻结注意力/融合路径。冻结模型可能保留 video Q-Former dropout，保持原模块状态并用逐查询签名核对。任何不变量失败即停止。输出、完整收据、张量 trace 和日志通过已有授权通道送 Windows。由 Windows 对三臂调用 `validate_bundle.py --family A --mode smoke`，返回绑定本次运行的 gate receipt。发现格式或环境问题时保留失败证据，禁止降低 checks。

## 5. A 正式运行（仅在 A smoke 通过并另获正式授权后）

将 A smoke 的 gate 配置到 `smoke_report.A`。依次 A0→A1→A7，用上方同一 launcher 将 `--mode smoke` 改成 `--mode full`，换成新授权编号。每臂按固定 2,990 条实际运行，一个 GPU 同时一臂。输出与 trace 落到独立目录，保留 `arm_completed.json`、每次运行收据和日志。失败只用递增 attempt 重试该臂，保留已完成且绑定一致的产物。将 A 全部原始输出、trace 和日志交 Windows 验证，返回 `A_full_report` 后才具备 B smoke 的先决条件。

## 6. B smoke（仅在 A 全量验证通过并另获 B smoke 授权后）

配置 `A_full_report`，同上使用 `--mode smoke`，依次 B3→B7HQ→B7LQ，各固定八条。B 的逐查询门槛包含相同查询帧、两路 Q-Former 输入、模块状态和 RNG、三条检索 clip、真实缓存 tensor 及原始相似度和实际 softmax 权重、聚合、attention K/V、gating 输入和视觉投影。仅按冻结前缀与尾部规则改变文本和由它导致的 prompt/decoder token。对齐失败即停 B，不得当作一般 Text7 实验运行。Windows 用 `validate_bundle.py --family B --mode smoke` 确认三臂 parity，回传 gate 收据。

## 7. B 正式运行（仅获单独正式授权后）

A 全量和 B smoke gate 均通过后，配置 `smoke_report.B`，按同一顺序使用 `--mode full`。按冻结顺序运行全部 2,990 条，将原始产物送 Windows。B 正式评估仍是冻结的 2,618 条。BN 不计算正式指标、不筛结果、不改假设。只把新看到的历史收据送 Windows 评审，现有 `EXACT_REUSABLE=0`，六组均需新生成。

Windows 用 `validate_bundle.py --family <A或B> --mode <smoke或full> --arm-directories <三臂路径JSON> --archive-scratch <Windows冻结回放目录> --gate-out <新门槛收据>` 验证。三臂路径 JSON 把 arm 名映射到各自回传的原始目录。全量校验可加 `--scorer-manifest <新manifest>` 保存评分输入。只把与相同运行配置绑定的 gate 收据发回 BN。

Windows 唯一正式评分路径为冻结 `evaluate_followup.py`。A 比较 A1−A0、A7−A0，B 比较 B7HQ−B3、B7LQ−B3、B7HQ−B7LQ。使用正式 PTB CIDEr、source cluster、10,000 次 source-level sign-flip、cluster bootstrap 及各家族 Holm。HQ/LQ 表示 X-CLIP 相关性区间。后续原始输出与 trace 使用既有授权通道交付。
