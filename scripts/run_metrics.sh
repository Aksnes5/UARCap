#!/bin/bash
# Usage: run_metrics.sh <generated_captions.json> <metrics.json>
# Paths may be given relative to the repo root or to evcap_external/.
set -e
GEN="$1"; OUT="$2"
GEN="${GEN#evcap_external/}"; OUT="${OUT#evcap_external/}"
cd /home/bn/videoufo_research/evcap_external
/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/venv/bin/python evaluation/cocoeval.py \
  --result_file_path "$GEN" --eval_file "$OUT"
