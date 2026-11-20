#!/bin/bash
# Usage: run_val.sh <CF|D> <k>   e.g. run_val.sh CF 3
set -e
TYPE="$1"; K="$2"
V="${TYPE}${K}"
if [ "$TYPE" = "CF" ]; then
  EXT=/home/bn/videoufo_research/external/external_val_cf7_cap.json
  SIM=/home/bn/videoufo_research/external/external_val_cf7_sims.json
elif [ "$TYPE" = "D" ]; then
  EXT=/home/bn/videoufo_research/external/external_val_d7_cap.json
  SIM=/home/bn/videoufo_research/external/external_val_d7_sims.json
else
  echo "TYPE must be CF or D"; exit 2
fi
cd /home/bn/videoufo_research/evcap_external
CKPT_NAME=train_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review.pt
mkdir -p "outputs/train_val497_${V}"
ln -sf /media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/lj/EVCapCode/outputs/train_evcap_msrvtt_quality_text/${CKPT_NAME} \
       "outputs/train_val497_${V}/${CKPT_NAME}"
export TORCH_HOME=/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/torch_cache
export HUGGINGFACE_HUB_CACHE=/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/hf_cache
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 USE_TF=0 TOKENIZERS_PARALLELISM=false
PY=/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/venv/bin/python
exec 9>/home/bn/videoufo_research/gpu.lock
flock -w 7200 9
exec "$PY" -u eval_evcap_fix_video_llama_xclip_msrvtt_fusion_gate_prompt5.py \
 --device cuda:0 --test_yaml val497/val16_video.yaml \
 --out_path outputs/eval_val497_${V} \
 --ckpt_name "$CKPT_NAME" \
 --bs 1 --topn $K --random_seed 42 --max_num_frames 16 --num_video_query_token 32 \
 --freeze_video_qformer True --vit_precision fp16 --freeze_llm True --freeze_vit True \
 --freeze_qformer True --load_video_feature True --share_query_tokens True --retriever True \
 --prompt_path prompts/prompt_evcap_msrvtt_detail_smallcap_prompt5.txt \
 --llm_model /media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/models/vicuna-7b-v1.3 \
 --deterministic True --use_fusion True \
 --ext_path "$EXT" --sim_scores_path "$SIM" \
 --video_feature_dir /home/bn/videoufo_research/features/videollama_faithful_v2
