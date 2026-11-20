#!/bin/bash
cd /home/bn/videoufo_research
while [ ! -f logs/queue_state/d5_ok ]; do sleep 30; done
echo "MSVD_START"
PY=/home/bn/anaconda3/envs/videollama2/bin/python
VENV=/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/venv/bin/python
GEN=eval_evcap_top3_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_msvd_generated_captions.json

PYTHONDONTWRITEBYTECODE=1 $PY -u scripts/extract_query_features_v2.py \
  /media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/lj/datasets/MSVD/test.caption.tsv \
  msvd_test_video_xclip_faithful_v2 > logs/msvd_query_features.log 2>&1
echo "MSVD_QUERY_FEATURES_DONE"
PYTHONDONTWRITEBYTECODE=1 $PY -u scripts/validate_msvd_pipeline.py > logs/msvd_validate.log 2>&1 || echo "MSVD_VALIDATE_FAILED"
echo "MSVD_VALIDATE_DONE"

if [ ! -f external/external_msvd_reranked.json ]; then
  PYTHONDONTWRITEBYTECODE=1 $PY -u scripts/make_external_retrieval_msvd.py > logs/msvd_retrieval.log 2>&1
  echo "MSVD_RETRIEVAL_DONE"
fi
$PY scripts/build_msvd_grid.py > logs/msvd_grid.log 2>&1
echo "MSVD_GRID_DONE"
$PY - <<'PYEOF' > /tmp/msvd_missing.txt
import os
sel=[l.strip() for l in open('/home/bn/videoufo_research/external/external_msvd_grid_clips.txt') if l.strip()]
have=set(f[:-3] for f in os.listdir('/home/bn/videoufo_research/features/videollama_faithful_v2') if f.endswith('.pt'))
print('\n'.join(c for c in sel if c not in have))
PYEOF
NM=$(grep -c . /tmp/msvd_missing.txt || true)
echo "MSVD_FEATURES_MISSING $NM"
if [ "$NM" -gt 0 ]; then
  TORCH_HOME=/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/torch_cache \
  HUGGINGFACE_HUB_CACHE=/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/hf_cache \
  TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 USE_TF=0 TOKENIZERS_PARALLELISM=false \
  PYTHONDONTWRITEBYTECODE=1 $PY -u scripts/extract_external_features_v2.py /tmp/msvd_missing.txt > logs/videollama_msvd.log 2>&1
  echo "MSVD_FEATURES_DONE"
fi

for K in 1 3 5 7; do
  V="D${K}"
  OUT="evcap_external/outputs/eval_msvd_${V}/${GEN}"
  if [ -f "$OUT" ] && [ "$($PY -c "import json;print(len(json.load(open('$OUT'))))")" = "670" ]; then echo "SKIP_GEN_${V}"; else
    bash scripts/run_msvd.sh $K > logs/gen_msvd_${V}.log 2>&1 || { echo "GEN_${V}_FAILED"; continue; }
    echo "GEN_${V}_DONE"
  fi
  ( cd evcap_external && $VENV evaluation/cocoeval.py --result_file_path "outputs/eval_msvd_${V}/${GEN}" --eval_file "outputs/eval_msvd_${V}/metrics.json" > ../logs/metrics_msvd_${V}.log 2>&1 )
  echo "METRICS_${V}_DONE"
done
$PY scripts/msvd_confirmatory_prepare.py > logs/msvd_prepare.log 2>&1 && echo "MSVD_PREPARE_DONE"
( cd evcap_external && $VENV evaluation/cocoeval.py --result_file_path "outputs/eval_msvd_Adaptive/${GEN}" --eval_file "outputs/eval_msvd_Adaptive/metrics.json" > ../logs/metrics_msvd_Adaptive.log 2>&1 )
echo "METRICS_ADAPTIVE_DONE"
$PY scripts/msvd_stats.py > logs/msvd_stats.log 2>&1
echo "MSVD_STATS_DONE"
