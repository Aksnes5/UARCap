#!/bin/bash
# Usage: run_test.sh <CF|D> <k>
set -e
TYPE="$1"; K="$2"; V="${TYPE}${K}"
if [ "$TYPE" = "CF" ]; then
  EXT=/home/bn/videoufo_research/external/external_test_cf7_cap.json
  SIM=/home/bn/videoufo_research/external/external_test_cf7_sims.json
else
  EXT=/home/bn/videoufo_research/external/external_test_d7_cap.json
  SIM=/home/bn/videoufo_research/external/external_test_d7_sims.json
fi
cd /home/bn/videoufo_research/evcap_external
CKPT_NAME=train_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review.pt
mkdir -p "outputs/train_test2990_${V}"
ln -sf /media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/lj/EVCapCode/outputs/train_evcap_msrvtt_quality_text/${CKPT_NAME} "outputs/train_test2990_${V}/${CKPT_NAME}"
export TORCH_HOME=/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/torch_cache
export HUGGINGFACE_HUB_CACHE=/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/hf_cache
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 USE_TF=0 TOKENIZERS_PARALLELISM=false
PY=/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/venv/bin/python
exec "$PY" -u eval_evcap_fix_video_llama_xclip_msrvtt_fusion_gate_prompt5.py \
 --device cuda:0 --test_yaml test2990/test2990_16frames.yaml \
 --out_path outputs/eval_test2990_${V} --ckpt_name "$CKPT_NAME" \
 --bs 1 --topn $K --random_seed 42 --max_num_frames 16 --num_video_query_token 32 \
 --freeze_video_qformer True --vit_precision fp16 --freeze_llm True --freeze_vit True \
 --freeze_qformer True --load_video_feature True --share_query_tokens True --retriever True \
 --prompt_path prompts/prompt_evcap_msrvtt_detail_smallcap_prompt5.txt \
 --llm_model /media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/models/vicuna-7b-v1.3 \
 --deterministic True --use_fusion True \
 --ext_path "$EXT" --sim_scores_path "$SIM" \
 --video_feature_dir /home/bn/videoufo_research/features/videollama_faithful_v2
