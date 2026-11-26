# E3 Exploratory Report (machine-assisted, uncalibrated)

{
  "schema": "uarcap_q2.e3_exploratory_receipt.v1",
  "status": "EXPLORATORY_E3_UNCALIBRATED",
  "utc": "2026-11-26T05:12:04.944113+00:00",
  "n_approved_candidates": 20,
  "n_generations_videollama": 80,
  "n_generations_qwen": 80,
  "n_eval_rows": 320,
  "n_eval_pairs": 160,
  "true_agreement": 0.9125,
  "conflict_agreement": 0.9875,
  "note": "Machine-assisted exploratory E3; no human validation; no confirmatory statistics."
}

## Per generator/arm/judge

- ('qwen', 'E3_BASE', 'gemma-3-27b-fp8'): {'n': 20, 'y_true': 17, 'y_conflict': 1}
- ('qwen', 'E3_BASE', 'llama-3.3-70b-w4a16'): {'n': 20, 'y_true': 16, 'y_conflict': 1}
- ('qwen', 'E3_CON', 'gemma-3-27b-fp8'): {'n': 20, 'y_true': 15, 'y_conflict': 3}
- ('qwen', 'E3_CON', 'llama-3.3-70b-w4a16'): {'n': 20, 'y_true': 15, 'y_conflict': 2}
- ('qwen', 'E3_NEU', 'gemma-3-27b-fp8'): {'n': 20, 'y_true': 17, 'y_conflict': 1}
- ('qwen', 'E3_NEU', 'llama-3.3-70b-w4a16'): {'n': 20, 'y_true': 17, 'y_conflict': 2}
- ('qwen', 'E3_SUP', 'gemma-3-27b-fp8'): {'n': 20, 'y_true': 17, 'y_conflict': 1}
- ('qwen', 'E3_SUP', 'llama-3.3-70b-w4a16'): {'n': 20, 'y_true': 18, 'y_conflict': 1}
- ('videollama', 'E3_BASE', 'gemma-3-27b-fp8'): {'n': 20, 'y_true': 7, 'y_conflict': 1}
- ('videollama', 'E3_BASE', 'llama-3.3-70b-w4a16'): {'n': 20, 'y_true': 8, 'y_conflict': 1}
- ('videollama', 'E3_CON', 'gemma-3-27b-fp8'): {'n': 20, 'y_true': 8, 'y_conflict': 1}
- ('videollama', 'E3_CON', 'llama-3.3-70b-w4a16'): {'n': 20, 'y_true': 8, 'y_conflict': 1}
- ('videollama', 'E3_NEU', 'gemma-3-27b-fp8'): {'n': 20, 'y_true': 8, 'y_conflict': 1}
- ('videollama', 'E3_NEU', 'llama-3.3-70b-w4a16'): {'n': 20, 'y_true': 9, 'y_conflict': 1}
- ('videollama', 'E3_SUP', 'gemma-3-27b-fp8'): {'n': 20, 'y_true': 9, 'y_conflict': 1}
- ('videollama', 'E3_SUP', 'llama-3.3-70b-w4a16'): {'n': 20, 'y_true': 11, 'y_conflict': 1}
