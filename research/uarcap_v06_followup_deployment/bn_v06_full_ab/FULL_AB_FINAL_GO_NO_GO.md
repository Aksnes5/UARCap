# UARCap v0.6 — Full A/B final GO / NO-GO

**Execution completed:** 2026-11-25 07:53 local on BN RTX A6000  
**Authorization:** `USER_APPROVED_ALL_AB_FOLLOWUP_20261124`  
**GPU work stopped after B full.** Formal statistics are Windows-only.

| Gate | Result |
| --- | --- |
| `A0_FULL` | **PASS — 2,990 rows** |
| `A1_FULL` | **PASS — 2,990 rows** |
| `A7_FULL` | **PASS — 2,990 rows** |
| `A_FULL_VALIDATED_GATE` | **PASS**, `trace_parity=true` |
| `B3_SMOKE` | **PASS — 8 rows** |
| `B7HQ_SMOKE` | **PASS — 8 rows** |
| `B7LQ_SMOKE` | **PASS — 8 rows** |
| `B_SMOKE_VALIDATED_GATE` | **PASS**, `trace_parity=true` |
| `B3_FULL` | **PASS — 2,990 rows** |
| `B7HQ_FULL` | **PASS — 2,990 rows** |
| `B7LQ_FULL` | **PASS — 2,990 rows** |
| `B_FULL_VALIDATED_GATE` | **PASS**, `trace_parity=true` |
| `A0_ZERO_EVIDENCE` | **PASS — 2,990/2,990** |
| `A_FULL_TECHNICALLY_READY` | **YES** |
| `B_FULL_TECHNICALLY_READY` | **YES** |
| `WINDOWS_FORMAL_STATS_READY` | **YES** |
| `FAILURES` | **0** |
| `RETRIES` | **0** |

## Counts and cost

| Metric | Value |
| --- | ---: |
| Full generations | **17,940** |
| B smoke generations | **24** |
| This run total | **17,964** |
| Full GPU hours | **11.0846 h** |
| B smoke GPU hours | **0.0708 h** |
| This run GPU hours | **11.1554 h** |
| All A/B hours including earlier A smoke | **11.2245 h** |
| Peak allocated VRAM | **18.71 GB** |

## Gate hashes

- A full: `bef8ce36341a4ae6712525e1777d76940cb5a0ffd05ded3f004d55c8e9414ec1`
- B smoke: `acd95c7aadcf7be588ab0f1bf4b03cff4f3c64342f8ab0b08a5da257c31e4408`
- B full: `f9255d9cfe884ec26c89accaaa2031802fdbb92f84dd25a71557008b9975c01e`

## Decision

- **A/B GPU execution: COMPLETE.**
- **Windows formal statistical analysis: GO.**
- **Further GPU runs: NOT AUTHORIZED / NOT NEEDED for this follow-up.**
- **No B result was used to alter A or B design.**

Windows should use the committed scorer manifests and raw ZIPs, then run the frozen PTB CIDEr / source-cluster / 10,000 sign-flip / bootstrap / Holm pipeline. BN did not compute or interpret significance.
