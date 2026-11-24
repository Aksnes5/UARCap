# BN live environment receipt

Checked at **2026-11-24T16:10:00** by `run_readonly_audit.py` (standard library only).

**Overall environment status: `MISMATCH`.** The host is reachable and the historical interpreter
exists, but the live Python/PyTorch versions do not match the archived environment expectation.

| Item | Live value | Status |
| --- | --- | --- |
| Host | `bn` | VERIFIED |
| OS | Ubuntu 24.04.4 LTS | VERIFIED |
| Kernel | `6.8.0-149-generic` | VERIFIED |
| GPU | NVIDIA RTX A6000 | VERIFIED |
| GPU UUID | `GPU-66d83fcc-2695-33a7-33aa-af30b7e3d26d` | VERIFIED |
| PCI bus | `00000000:01:00.0` | VERIFIED |
| VRAM | 49140 MiB total / 15 MiB used | VERIFIED |
| Driver | `580.178.04` | VERIFIED |
| Compute clients | none (`nvidia-smi --query-compute-apps` empty) | VERIFIED |
| GPU lock | `/home/bn/videoufo_research/gpu.lock` present, 0 bytes, no holder | VERIFIED |
| Project disk | `/dev/nvme0n1p2  3.6T  1.8T  1.7T   52% /` | VERIFIED |
| Display disk | `/dev/sda        7.3T  3.1T  3.9T   44% /media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1` | VERIFIED |
| Display mount | `/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1 /dev/sda ext4   ro,relatime,norecovery` | VERIFIED (read-only) |
| Project mount | `/      /dev/nvme0n1p2 ext4   rw,relatime,errors=remount-ro` | VERIFIED (read-write) |
| Project git | `fix/val497-frame-loader-postfix-rerun-20261123` @ `52b5ed3a69e4b84a8e16ed747f141eefb5b1e2ec` | dirty entries: 107 |
| Receipts worktree | `/home/bn/uar_bn_receipts` @ `94f5529f8a6e4424d2af207df42482cf87c7ce1b` | branch `codex/uarcap-bn-readonly-receipts-20261124` |

## Effective Python runtime used by the historical runner

`scripts/run_test.sh` (SHA-256 `42d7d58e534bfcb798f01fc99ee0a10d70e600aa41af05431848ff7d49b2c88f`) launches:

```text
/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/venv/bin/python
-> /home/bn/anaconda3/envs/TTP/bin/python3.9
```

`pyvenv.cfg`:

```text
home = /home/bn/anaconda3/envs/TTP/bin
include-system-site-packages = true
version = 3.9.13
```

| Package | Live | Archived expectation | Match |
| --- | --- | --- | --- |
| Python | `3.9.13` | `3.9.18` | **NO** |
| PyTorch | `2.1.2+cu121` | `2.1.0` | **NO** |
| Transformers | `4.28.0` | `4.28.0` | YES |
| timm | `0.6.13` | `0.6.13` | YES |

- `sys.prefix` = `/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/venv`
- `sys.base_prefix` = `/home/bn/anaconda3/envs/TTP`
- `torch` resolves to `/home/bn/anaconda3/envs/TTP/lib/python3.9/site-packages/torch/__init__.py`
- `transformers` resolves to `/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/venv/lib/python3.9/site-packages/transformers/__init__.py`

The venv is a **hybrid**: it includes the TTP Conda environment's site-packages, so `torch`
comes from TTP while `transformers` comes from the venv. Historical D1/D3/D7 runs used this
interpreter, but no per-job package snapshot exists. This is a reproducibility deviation from the
archived environment receipt, not a missing asset.

Conda environments visible: `go_ai_tf219, GTR, huapo, protracker, TTP, videollama2, videollama2.tar.gz`.

No CUDA context was created, no `torch` import occurred and no model weight was loaded for this
receipt. GPU fields come from `nvidia-smi` management queries only.
