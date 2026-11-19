# Session transcripts

Raw Claude Code session logs, kept for provenance of the research decisions
(not part of the runnable code base).

| File | Session |
|---|---|
| `2026-11-19-uar-reproduction-session.jsonl` | EVCap Table-1 reproduction → VideoUFO download/verification → network watchdog forensics → Phase 1–3 → FAITHFUL_V2 preprocessing fix + GitHub setup (2026-11-19) |

Format: one JSON object per line (Claude Code transcript format: user/assistant
messages, internal reasoning blocks, tool calls and their raw outputs).

Notes:
- These logs are **verbatim**; they contain absolute server paths, command
  output, and intermediate reasoning. Large binary/payload outputs were never
  logged, but logs may still include file listings and metric dumps.
- A secret scan was run before committing: no plaintext Wi-Fi PSK (NetworkManager
  logs show `<hidden>`), no private keys, no GitHub/API tokens, no passwords.
- This repository is public. Do not add logs here that contain credentials.
