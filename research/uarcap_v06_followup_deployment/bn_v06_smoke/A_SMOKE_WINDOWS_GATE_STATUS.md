# A smoke official Windows gate — status

**Status: BLOCKED — this session is not on Windows.**

Checked at 2026-11-24 20:35 local. The host is `bn`, Ubuntu 24.04 Linux
(`uname: Linux bn 6.8.0-149-generic`). There is no `D:\UARCap`, no `/mnt/d`, and no
Windows execution endpoint in this session. Therefore no official
`A_SMOKE_WINDOWS_GATE.json` was generated and no Windows PASS is claimed.

## What was verified from the pinned commit

Source commit: `3e49a1f1494a28ba88569b3723bf971bcb71581d`

- `A_SMOKE_ARTIFACT_MANIFEST.json`: 26/26 listed files exist with matching size and SHA-256.
- Manifest SHA-256: `52b7992a42396b95d586f099b3dce78b7528547d1452a55930d46e9adf867bbf`.
- A0/A1/A7 completion seals: mode `smoke`, 8 rows each; prediction and trace hashes match.
- All raw prediction, trace, run receipt, telemetry, completion marker and launcher log files are present in the commit.

## Independent frozen-validator recheck

The frozen `validate_bundle.py --family A --mode smoke` was executed again from a fresh
archive scratch against the **committed raw directories**, not against the earlier live
output directories.

Result:

- `trace_parity = true`
- `rows_per_arm = 8`
- `run_id = smoke_uarcapv06`
- `validated_files_sha256` exactly equals the BN smoke gate
- gate file: `A_SMOKE_INDEPENDENT_RECHECK_GATE.json`
- SHA-256: `b07ad358b4efa1ab3d64b3bad5273b27445f1f4c7c7a7f3bf0c99e7b806a9082`
- This is byte-identical to `A_SMOKE_BN_GATE.json`.

This is an independent re-execution of the frozen validator, but it ran on **BN Linux**. It does not satisfy the requested Windows-host gate provenance.

## Required Windows command

On the Windows checkout containing the frozen `validate_bundle.py` and the replay scratch:

```text
validate_bundle.py --family A --mode smoke \
  --arm-directories <JSON mapping A0/A1/A7 to raw directories> \
  --archive-scratch <Windows frozen replay scratch> \
  --gate-out A_SMOKE_WINDOWS_GATE.json
```

The mapping should point at the committed paths:

```text
research/uarcap_v06_followup_deployment/bn_v06_smoke/raw/A0
research/uarcap_v06_followup_deployment/bn_v06_smoke/raw/A1
research/uarcap_v06_followup_deployment/bn_v06_smoke/raw/A7
```

The Windows command must verify the same raw hashes and produce the official gate; only then may `A_SMOKE_WINDOWS_PASS=YES` and `A_FULL_TECHNICALLY_READY=YES` be declared.

## Final status

| Field | Value |
| --- | --- |
| `A_SMOKE_BN_PASS` | **YES** |
| `A_SMOKE_INDEPENDENT_RECHECK` | **PASS** |
| `A_SMOKE_WINDOWS_PASS` | **PENDING / NOT RUN** |
| `A_FULL_TECHNICALLY_READY` | **PENDING** |
| `P0_TECHNICAL_BLOCKERS` | **NONE found** |
| `P0_ENVIRONMENT_BLOCKER` | **Windows host not accessible from this session** |
| A full execution | **NOT AUTHORIZED / NOT STARTED** |

No full-run command is released from this state. A full execution still requires both the
official Windows gate and a separate user authorization.
