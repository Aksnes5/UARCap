#!/bin/bash
cd /home/bn/videoufo_research
while ! grep -q "REPAIR_DONE" logs/repair_chain.log 2>/dev/null; do sleep 30; done
PY=/home/bn/anaconda3/envs/videollama2/bin/python
VENV=/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/venv/bin/python
GEN=eval_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review_generated_captions.json
echo "FIX_START"
NM=$(grep -c . /tmp/indomain_missing.txt || true); echo "INDOMAIN_FEATURES_NEEDED $NM"
TORCH_HOME=/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/torch_cache \
HUGGINGFACE_HUB_CACHE=/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/hf_cache \
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 USE_TF=0 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 \
PYTHONDONTWRITEBYTECODE=1 $PY -u scripts/extract_msrvtt_train_features.py /tmp/indomain_missing.txt > logs/indomain_features.log 2>&1
echo "INDOMAIN_FEATURES_RC $?"
for K in 1 3 5 7; do
  rm -rf "evcap_external/outputs/eval_val497_indomain_D${K}" "evcap_external/outputs/train_val497_indomain_D${K}"
  bash scripts/run_indomain.sh $K > logs/gen_indomain_D${K}.log 2>&1 && echo "GEN_indomain_D${K}_DONE"
  ( cd evcap_external && $VENV evaluation/cocoeval.py --result_file_path "outputs/eval_val497_indomain_D${K}/${GEN}" --eval_file "outputs/eval_val497_indomain_D${K}/metrics.json" > ../logs/metrics_indomain_D${K}.log 2>&1 )
  echo "METRICS_indomain_D${K}_DONE"
done
# oracle3/7 rerun with fixed-DF scorer
for K in 3 7; do
  rm -rf "evcap_external/outputs/eval_val497_oracle${K}" "evcap_external/outputs/train_val497_oracle${K}"
  bash scripts/run_val_variant.sh oracle${K} $K 0 0 > logs/gen_oracle${K}.log 2>&1 && echo "GEN_oracle${K}_DONE"
  ( cd evcap_external && $VENV evaluation/cocoeval.py --result_file_path "outputs/eval_val497_oracle${K}/${GEN}" --eval_file "outputs/eval_val497_oracle${K}/metrics.json" > ../logs/metrics_oracle${K}.log 2>&1 )
  echo "METRICS_oracle${K}_DONE"
done
echo "FIX_DONE"
