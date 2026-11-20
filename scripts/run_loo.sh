#!/bin/bash
set -e
I="$1"; V="loo${I}"
EXT=/home/bn/videoufo_research/external/external_val_loo_rm${I}_cap.json
SIM=/home/bn/videoufo_research/external/external_val_loo_rm${I}_sims.json
cd /home/bn/videoufo_research/evcap_external
CKPT_NAME=train_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review.pt
mkdir -p outputs/train_val150_${V}
ln -sf /media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/lj/EVCapCode/outputs/train_evcap_msrvtt_quality_text/${CKPT_NAME} outputs/train_val150_${V}/${CKPT_NAME}
export TORCH_HOME=/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/torch_cache
export HUGGINGFACE_HUB_CACHE=/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/hf_cache
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 USE_TF=0 TOKENIZERS_PARALLELISM=false
PY=/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/venv/bin/python
exec 9>/home/bn/videoufo_research/gpu.lock
flock -w 7200 9
exec "$PY" -u eval_evcap_fix_video_llama_xclip_msrvtt_fusion_gate_prompt5.py \
 --device cuda:0 --test_yaml val150/val150_video.yaml \
 --out_path outputs/eval_val150_${V} --ckpt_name "$CKPT_NAME" \
 --bs 1 --topn 6 --random_seed 42 --max_num_frames 16 --num_video_query_token 32 \
 --freeze_video_qformer True --vit_precision fp16 --freeze_llm True --freeze_vit True \
 --freeze_qformer True --load_video_feature True --share_query_tokens True --retriever True \
 --prompt_path prompts/prompt_evcap_msrvtt_detail_smallcap_prompt5.txt \
 --llm_model /media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/models/vicuna-7b-v1.3 \
 --deterministic True --use_fusion True --ext_path "$EXT" --sim_scores_path "$SIM" \
 --video_feature_dir /home/bn/videoufo_research/features/videollama_faithful_v2
