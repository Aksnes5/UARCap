#!/bin/bash
# Untouched-test pipeline: retrieval -> grid -> missing features -> D1/3/5/7 -> metrics.
cd /home/bn/videoufo_research
PY=/home/bn/anaconda3/envs/videollama2/bin/python
VENV=/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/venv/bin/python
LEDGER=reports/runtime_ledger.csv
GEN=eval_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review_generated_captions.json

if [ ! -f external/external_test_reranked.json ]; then
  PYTHONDONTWRITEBYTECODE=1 $PY -u scripts/make_external_retrieval_test.py > logs/retrieval_test.log 2>&1
  echo "TEST_RETRIEVAL_DONE"
fi
$PY scripts/build_test_grid.py > logs/build_test_grid.log 2>&1
echo "TEST_GRID_DONE"

$PY - <<'PYEOF' > /tmp/test_missing.txt
import os
sel=[l.strip() for l in open('/home/bn/videoufo_research/external/external_test_grid_clips.txt') if l.strip()]
have=set(f[:-3] for f in os.listdir('/home/bn/videoufo_research/features/videollama_faithful_v2') if f.endswith('.pt'))
print('\n'.join(c for c in sel if c not in have))
PYEOF
NMISS=$(grep -c . /tmp/test_missing.txt || true)
echo "TEST_FEATURES_MISSING $NMISS"
if [ "$NMISS" -gt 0 ]; then
  TORCH_HOME=/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/torch_cache \
  HUGGINGFACE_HUB_CACHE=/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/hf_cache \
  TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 USE_TF=0 TOKENIZERS_PARALLELISM=false \
  PYTHONDONTWRITEBYTECODE=1 $PY -u scripts/extract_external_features_v2.py /tmp/test_missing.txt > logs/videollama_test.log 2>&1
  echo "TEST_FEATURES_DONE"
fi

for K in 1 3 5 7; do
  V="D${K}"
  OUT="evcap_external/outputs/eval_test2990_${V}/${GEN}"
  if [ -f "$OUT" ] && [ "$($PY -c "import json;print(len(json.load(open('$OUT'))))")" = "2990" ]; then echo "SKIP_GEN_${V}"; continue; fi
  T0=$(date +%s)
  bash scripts/run_test.sh D $K > logs/gen_test_${V}.log 2>&1 || { echo "GEN_${V}_FAILED"; tail -3 logs/gen_test_${V}.log; continue; }
  T1=$(date +%s)
  ( cd evcap_external && $VENV evaluation/cocoeval.py --result_file_path "outputs/eval_test2990_${V}/${GEN}" --eval_file "outputs/eval_test2990_${V}/metrics.json" > ../logs/metrics_test_${V}.log 2>&1 )
  T2=$(date +%s)
  N=$($PY -c "import json;print(len(json.load(open('$OUT'))))")
  echo "test2990_${V},FULL,D,${K},${N},,$((T1-T0)),$((T2-T1)),$(echo "scale=3;($T2-$T0)/3600"|bc),done" >> "$LEDGER"
  echo "GEN_${V}_DONE"
done
echo "TEST_GRID_DONE_ALL"
