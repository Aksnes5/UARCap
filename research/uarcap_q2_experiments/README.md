# UARCap Q2 Upgrade Experiments — Offline Implementation & BN Deployment Package

> **Engineering Status**: `OFFLINE_REAL_EXECUTION_IMPLEMENTED_AWAITING_BN_RUNTIME_VERIFICATION`  
> **Protocol Baseline**: `761dea2c51ab07c083b88805168ad1757db53d67` (`antigravity/uarcap-q2-pilot-a-protocol-v1.2-final`)  
> **Important Notice**: All 52 CPU tests (43 unit/E2E/fault-injection/`bash -n` tests + 9 real-entrypoint contract & golden historical regression tests) pass on Windows CPU. **No real BN GPU inference, full dataset download, or formal human annotation has been executed yet.**

## Trusted public dataset mirrors (ModelScope)

The stage-gated downloader prefers the official source and then falls back to a
**SHA-256 pinned ModelScope mirror**. A mirror is only accepted when its file
`Sha256` equals the HuggingFace LFS `oid` (bit-identical content):

```bash
# Probe registered mirrors and compare every file's SHA-256 against the pinned HF oid
python deployment/stage_assets.py mirror-check --dataset carebench
# Parallel (16-24 way) resumable Range download with full-file SHA-256 verification
python deployment/stage_assets.py download --data-root /home/bn/uarcap_q2_runtime/data \
  --asset carebench_videos --confirm-download \
  --auth-receipt /home/bn/uarcap_q2_runtime/authorizations/pilot_a01_auth.json
```

Registered mirrors live in `deployment/stage_assets.py::MODELSCOPE_MIRRORS`;
verified SHA-256 pins live in `VERIFIED_SHA256`. When adding a new public
dataset, first check for a ModelScope counterpart and prove the hash match
before registering it.

## Directory Layout

- `uarcap_q2/`: Modular Python package (`common/`, `datasets/`, `retrieval/`, `interventions/`, `generators/`, `evaluators/`, `annotation/`, `statistics/`, `runner/`)
- `configs/`: Declarative YAML configs (`bn_paths.example.yaml`, `pilot_a01_config.yaml`, `pilot_a23_config.yaml`, `formal_b_candidate.yaml`) + `configs/envs/` (4 strictly isolated environments)
- `deployment/`: Non-destructive BN preflight probe, stage-gated asset transfer CLI, authorization gate CLI, unified stage runner CLI
- `scripts/`: One-click Linux bash entrypoints (`bootstrap_bn.sh`, `run_preflight.sh`, `run_pilot_a01.sh`, `run_pilot_a2.sh`, `run_pilot_a3.sh`, `run_formal_b.sh`, `status.sh`, `resume.sh`) + cross-platform `launch_annotation_ui.py`
- `fixtures/`: Synthetic dynamic & pathological MP4 generator (used strictly in `tmp_path` during pytest; zero large binaries committed)
- `tests/`: 9 pytest suites (`43 passed in 4.37s`)

## Quick Links

- Server operator manual after regaining BN access: [`BN_SERVER_QUICKSTART.md`](BN_SERVER_QUICKSTART.md)
- Implementation summary: [`OFFLINE_IMPLEMENTATION_REPORT.md`](OFFLINE_IMPLEMENTATION_REPORT.md)
- Protocol-to-code traceability: [`SCIENTIFIC_PROTOCOL_IMPLEMENTATION_MATRIX.md`](SCIENTIFIC_PROTOCOL_IMPLEMENTATION_MATRIX.md)
- Authorization gates: [`AUTHORIZATION_GATE_MATRIX.md`](AUTHORIZATION_GATE_MATRIX.md)
- Pending BN GPU verifications: [`SERVER_RUNTIME_VERIFICATION_PENDING.md`](SERVER_RUNTIME_VERIFICATION_PENDING.md)
