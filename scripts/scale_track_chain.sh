#!/bin/bash
# Usage: scale_track_chain.sh <scale> <D|CF> [ks...]
cd /home/bn/videoufo_research
S="$1"; T="$2"; shift 2; KS="${*:-1 3 5 7}"
PY=/home/bn/anaconda3/envs/videollama2/bin/python
VENV=/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/venv/bin/python
GEN=eval_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review_generated_captions.json
PREFIX=external_val_${S}
if [ ! -f external/${PREFIX}_reranked.json ]; then
  PYTHONDONTWRITEBYTECODE=1 $PY -u scripts/make_external_retrieval_val_scale.py $S > logs/retrieval_val_${S}.log 2>&1 || { echo "RETRIEVAL_${S}_FAILED"; exit 1; }
  echo "RETRIEVAL_${S}_DONE"
fi
$PY scripts/build_val_grid_scale.py $S > logs/build_val_grid_${S}.log 2>&1
$PY scripts/audit_candidate_features.py ${PREFIX} ${T} > logs/cache_audit_${S}_${T}.log 2>&1
NM=$(grep -c . /tmp/missing_${PREFIX}_${T}.txt || true)
echo "MISSING_${S}_${T} $NM"
if [ "$NM" -gt 0 ]; then
  TORCH_HOME=/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/torch_cache \
  HUGGINGFACE_HUB_CACHE=/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/hf_cache \
  TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 USE_TF=0 TOKENIZERS_PARALLELISM=false \
  PYTHONDONTWRITEBYTECODE=1 $PY -u scripts/extract_external_features_v2.py /tmp/missing_${PREFIX}_${T}.txt > logs/features_${S}_${T}.log 2>&1
  echo "FEATURES_${S}_${T}_DONE"
fi
for K in $KS; do
  V="${S}_${T}${K}"
  OUT="evcap_external/outputs/eval_val497_${V}/${GEN}"
  if [ -f "$OUT" ] && [ "$($PY -c "import json;print(len(json.load(open('$OUT'))))")" = "497" ]; then echo "SKIP_GEN_${V}"; else
    bash scripts/run_val_scale.sh $S $T $K > logs/gen_val_${V}.log 2>&1 || { echo "GEN_${V}_FAILED"; continue; }
    echo "GEN_${V}_DONE"
  fi
  M="evcap_external/outputs/eval_val497_${V}/metrics.json"
  if [ ! -f "$M" ]; then
    ( cd evcap_external && $VENV evaluation/cocoeval.py --result_file_path "outputs/eval_val497_${V}/${GEN}" --eval_file "outputs/eval_val497_${V}/metrics.json" > ../logs/metrics_val_${V}.log 2>&1 )
  fi
  echo "METRICS_${V}_DONE"
done
if [ "$T" = "D" ]; then $PY scripts/val_retrieval_metrics_scale.py $S > logs/retrieval_metrics_${S}.log 2>&1; fi
echo "STAGE_${S}_${T}_DONE"
