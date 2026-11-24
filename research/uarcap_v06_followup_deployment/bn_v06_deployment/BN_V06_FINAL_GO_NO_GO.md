# UARCap v0.6 BN deployment preparation — final GO / NO-GO

Checked on **BN Ubuntu** at **2026-11-24 19:56** (local). This round performed staging, incremental environment binding, real deployment configuration and six pure dry-runs. **No model was loaded, no CUDA context was created, no GPU smoke or formal run was started, and no server software or network configuration was changed.**

## 1. Persistent staging

| Item | Value |
| --- | --- |
| Persistent runtime root | `/home/bn/uar_v06_runtime` |
| Inputs directory | `/home/bn/uar_v06_runtime/inputs` |
| Isolated inference checkout | `/home/bn/uar_v06_runtime/evcap_work` |
| Deployment files | `/home/bn/uar_v06_runtime/deployment` |
| Receipts | `/home/bn/uar_v06_runtime/receipts` |
| Runs root | `/home/bn/uar_v06_runtime/runs` |
| Project disk | `/dev/nvme0n1p2`, 1.7 TB free |
| Display mount | `/dev/sda`, read-only, 3.9 TB free |

The transfer ZIP has **2,861,489 bytes** and SHA-256:

`93289136546fc2d96d2f457cc9dec57c50fde1b2c517f735f23df0a1dfc63128`

The first raw download was interrupted at the BN egress. The exact package was then taken from the already-fetched Git object at commit `9f8f8ca5c10612503fadad6359ac9129968d399e` (the same GitHub transfer). Its size and SHA-256 match the frozen transfer manifest exactly. `receive_bn_inputs.py` then verified every member and unpacked into a fresh directory.

**Result:** 18/18 members verified; 8 prepared inputs, 6 feature-ID lists, preflight summary, replay receipt, embedded manifest and README are present. All 2,990 full, 2,690 untouched and 2,618 B-eligible IDs and orders passed.

## 2. Environment binding

The actual historical runner entry is:

`/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/venv/bin/python`

It resolves to `/home/bn/anaconda3/envs/TTP/bin/python3.9`; `sys.prefix` remains the Display venv. `torch` resolves from TTP, while `transformers` and `timm` resolve from the Display venv. This is the same hybrid interpreter used by the historical runner.

| Component | Archived expectation | Live BN | Match |
| --- | --- | --- | --- |
| Python | 3.9.18 | 3.9.13 | NO |
| PyTorch | 2.1.0 | 2.1.2+cu121 | NO |
| torchvision | 0.16.0 | 0.16.2+cu121 | NO |
| Transformers | 4.28.0 | 4.28.0 | YES |
| timm | 0.6.13 | 0.6.13 | YES |
| NumPy | 1.24.3 | 1.26.4 | NO |
| Pillow | 10.0.1 | 11.3.0 | NO |
| OpenCV | 4.7.0.72 | 4.10.0.84 | NO |
| PyAV | 14.0.1 | 14.0.1 | YES |

No package was installed, upgraded or downgraded. `python_environment.json` records the interpreter, sys.path, shell environment, package versions, distribution roots and small module hashes.

### Python 3.9 `zip(strict=True)` adaptation

- `runtime_trace.vision_py39` replaces `launch_pinned.actual_vision` before model execution and checks equal caption/score lengths explicitly.
- The adapter was exercised under Python 3.9.13 on real D CAP/SIM data for `k=0,1,3,7`: **PASS**.
- The frozen evaluator source contains no `strict=True` call.
- `deploy_guard.py` and the six dry-runs import **no torch and no CUDA module** (`torch_imported=False`).
- `prepare_inputs.py` / `evaluate_followup.py` strict paths remain Windows-only and are not used on BN.

## 3. Model, frame and feature assets

| Asset | Receipt |
| --- | --- |
| Checkpoint | 147,337,150 bytes; SHA-256 `a8b5af79…eb8002` |
| Vicuna-7B-v1.3 | config/tokenizer/index present; 2 shards 9,976,634,558 + 3,500,315,539 bytes |
| Video-LLaMA Q-Former | 88,245,048 bytes; SHA-256 `a718a15e…cacb6d1` |
| EVA-ViT-G | 2,025,249,237 bytes |
| BLIP2 Q-Former | 433,481,467 bytes |
| Frame TSV | 1,378,294,858 bytes; SHA-256 `ee0a372585c3bfa7aad9feb0a990e6c22b7e5f42209efb1d7427b008a7dfbb7e` |
| Frame lineidx | SHA-256 `61ae0fc9…42f41ed` |
| Label/caption TSV | 4,015,483 bytes each; SHA-256 `5ca73d0e…ef5e18` |
| A1 feature coverage | 2,539 unique clips, 0 missing |
| A7 feature coverage | 15,213 unique clips, 0 missing |
| B3/B7HQ/B7LQ coverage | 7,035 unique clips, 0 missing |
| Frozen source | 6/6 code hashes match |
| Runtime dependencies | 9/9 hashes match |

The image TSV content SHA was computed once by sequential read-only hashing; size and nanosecond mtime were unchanged before and after. No mtime or offset hash was substituted for content.

## 4. Real deployment configuration

Deployment JSON: `deployment_uarcapv06.json`, SHA-256:

`86f2306d8197ceee105c78780f337e601a5b3224b1d49fd152ed7271ba67f0e6`

| Field | Bound value |
| --- | --- |
| `run_id` / `attempt` | `uarcapv06` / `1` |
| `device` / GPU UUID | `cuda:0` / `GPU-66d83fcc-2695-33a7-33aa-af30b7e3d26d` |
| `original_evcap_root` | `/home/bn/videoufo_research/evcap_external` |
| Isolated root | `/home/bn/uar_v06_runtime/evcap_work` |
| Checkpoint slot | `outputs/train_v06_{smoke|full}_uarcapv06_attempt1_<arm>/<checkpoint>.pt` |
| Output slot | `outputs/eval_v06_{smoke|full}_uarcapv06_attempt1_<arm>` |
| Shared lock | `/home/bn/videoufo_research/gpu.lock` |
| Offline caches | Display `repro_evcap/torch_cache`, `repro_evcap/hf_cache` |
| `server_data_ready` | `true` |
| `approval_id` | `null` |

All six checkpoint slots are symlinks to the single canonical checkpoint. The isolated `outputs/` contains only these 12 placeholder slot directories; **no `eval_*` output directory exists**. The historical tracked prediction copies from the source archive were moved to `evcap_work/_archive_tracked_outputs` and are not used.

`deploy_guard.file_receipts` and `verify_isolation` passed for all six arms. No output directory was created by a dry-run.

## 5. Six pure dry-runs

| Arm | topn | text_topn | visual_topn | Feature IDs | Input CAP / SIM | Result |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| A0 | 0 | 0 | 0 | 0 | `A_k0_cap` / `A_k0_sims` | PASS |
| A1 | 1 | 1 | 1 | 2,539 | frozen D cap / sims | PASS |
| A7 | 7 | 7 | 7 | 15,213 | frozen D cap / sims | PASS |
| B3 | 7 | 3 | 3 | 7,035 | `B_text3_cap` / `B_shared_visual_sims` | PASS |
| B7HQ | 7 | 7 | 3 | 7,035 | `B_text7_hq_cap` / `B_shared_visual_sims` | PASS |
| B7LQ | 7 | 7 | 3 | 7,035 | `B_text7_lq_cap` / `B_shared_visual_sims` | PASS |

Each dry-run returned `loads_model=false`, a unique output path, a unique checkpoint slot, absolute real asset paths and the expected frozen flags. B3, B7HQ and B7LQ all point to the same `B_shared_visual_sims.json` and the same `B_visual_contract.json`; only the text CAP differs. No arm references a historical output directory.

## 6. Final gates

| Gate | Value | Basis |
| --- | --- | --- |
| `INPUTS_STAGED` | **YES** | 18 files in persistent `inputs/` |
| `INPUT_HASH_VERIFIED` | **YES** | 18/18 member hashes and 8 canonical digests |
| `SERVER_DATA_READY` | **YES** | Dataset, frame content, checkpoint, model caches and features bound |
| `RUNTIME_ENVIRONMENT_BOUND` | **YES (static)** | Interpreter/packages/caches bound; deviations documented; native import not exercised |
| `DEPLOYMENT_JSON_READY` | **YES** | Real paths, six slots, config hash above |
| `SIX_ARM_DRY_RUN_PASS` | **YES** | 6/6 |
| `GPU_IDLE_AT_CHECK` | **YES** | 15 MiB used, 0% util, no compute client, lock unheld at 19:54:42 |
| `GPU_SMOKE_READY` | **YES — A0/A1/A7 only** | All non-GPU prerequisites for A passed; authorization still required |
| `GPU_EXECUTION_APPROVED` | **NO** | Not granted |

### A smoke

A0/A1/A7 are technically ready to request a real smoke authorization. The smoke must still prove what static checks cannot: actual `torch` import, CUDA context, fp16 dtype, attention mask, Q-Former module state, RNG, first-eight-query frame tensors, A0 zero external loads, and the exact evaluator path. The GPU must be rechecked under the shared lock immediately before launch.

### B independent gates

B3/B7HQ/B7LQ inputs and dry-runs are ready, but B is **not** smoke-ready. Remaining gates:

1. Validated **A full report** from Windows.
2. Separate **B smoke authorization**.
3. Real per-query visual parity smoke: identical query frames, clip IDs, loaded tensors, softmax weights, K/V, gating inputs and visual projection; only text evidence may differ.
4. Validated B smoke gate before any B full run.

## 7. P0 / P1

**P0 blockers for A smoke: none technical.** The only remaining gate is explicit GPU smoke authorization.

**P1, documented:**

- Python/PyTorch/torchvision/NumPy/Pillow/OpenCV differ from the archived environment. No reinstall was performed, per instruction. The real runtime is the same hybrid venv used historically.
- Native `torch` import and CUDA context were deliberately not exercised; the first authorized smoke must prove startup and tensor behavior.
- The Display dataset mount is read-only; all new outputs go to the read-write project filesystem.

**B-only P0 gates:** A full report, B smoke authorization, visual parity gate.

## 8. Reproducibility note

The six dry-run JSONs, configuration hash, input staging receipt, environment binding, model/frame manifests, GPU state and final status are committed under `research/uarcap_v06_followup_deployment/bn_v06_deployment/`. The actual 12 MB input payload remains in the persistent runtime directory and is not duplicated into Git; its bytes are bound by the transfer manifest and the 18-member receipt.

**Final authorization statement: GPU smoke and formal GPU execution are still NOT authorized. This round stops at the deployment gate.**
