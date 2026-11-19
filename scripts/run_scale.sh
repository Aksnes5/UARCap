#!/bin/bash
set -e
L="$1"
cd /home/bn/videoufo_research/evcap_external
export TORCH_HOME=/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/torch_cache
export HUGGINGFACE_HUB_CACHE=/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/hf_cache
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 USE_TF=0 TOKENIZERS_PARALLELISM=false
PY=/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/venv/bin/python
EXT=/home/bn/videoufo_research/external
exec "$PY" -u eval_evcap_fix_video_llama_xclip_msrvtt_fusion_gate_prompt5.py \
 --device cuda:0 --test_yaml dev300/dev300_16frames.yaml \
 --out_path outputs/eval_dev300_D${L} \
 --ckpt_name train_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review.pt \
 --bs 1 --topn 7 --random_seed 42 --max_num_frames 16 --num_video_query_token 32 \
 --freeze_video_qformer True --vit_precision fp16 --freeze_llm True --freeze_vit True \
 --freeze_qformer True --load_video_feature True --share_query_tokens True --retriever True \
 --prompt_path prompts/prompt_evcap_msrvtt_detail_smallcap_prompt5.txt \
 --llm_model /media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/models/vicuna-7b-v1.3 \
 --deterministic True --use_fusion True \
 --ext_path $EXT/external_${L}_reranked.json \
 --sim_scores_path $EXT/external_${L}_sims.json \
 --video_feature_dir /home/bn/videoufo_research/features/external_top7
