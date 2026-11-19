#!/bin/bash
# Usage: run_baseline.sh A|B|C|D
set -e
M="$1"
cd /home/bn/videoufo_research/evcap_external
export TORCH_HOME=/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/torch_cache
export HUGGINGFACE_HUB_CACHE=/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/hf_cache
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 USE_TF=0 TOKENIZERS_PARALLELISM=false
PY=/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/venv/bin/python
EXT=/home/bn/videoufo_research/external
COMMON="--device cuda:0 --test_yaml dev300/dev300_16frames.yaml \
 --ckpt_name train_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review.pt \
 --bs 1 --random_seed 42 --max_num_frames 16 --num_video_query_token 32 \
 --freeze_video_qformer True --vit_precision fp16 --freeze_llm True --freeze_vit True \
 --freeze_qformer True --load_video_feature True --share_query_tokens True --retriever True \
 --prompt_path prompts/prompt_evcap_msrvtt_detail_smallcap_prompt5.txt \
 --llm_model /media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/models/vicuna-7b-v1.3 \
 --deterministic True"
case "$M" in
 A) ARGS="--out_path outputs/eval_dev300_A --topn 0 --use_fusion True --ext_path $EXT/external_empty_cap.json --sim_scores_path $EXT/external_empty_sims.json" ;;
 B) ARGS="--out_path outputs/eval_dev300_B --topn 7 --use_fusion True --ext_path ext_data/rerank_sentence_by_sentence_as_inference_rouge_quality_no_duplicate_with_video_ids.json" ;;
 CF) ARGS="--out_path outputs/eval_dev300_CF --topn 7 --use_fusion True --ext_path $EXT/external_naive_top7_cap.json --sim_scores_path $EXT/external_sims_full.json --video_feature_dir /home/bn/videoufo_research/features/external_top7" ;;
 C7) ARGS="--out_path outputs/eval_dev300_C7 --topn 7 --use_fusion False --ext_path $EXT/external_naive_top7_cap.json --sim_scores_path $EXT/external_sims_full.json" ;;
 C) ARGS="--out_path outputs/eval_dev300_C --topn 20 --use_fusion False --ext_path $EXT/external_naive_top20_cap.json --sim_scores_path $EXT/external_sims_full.json" ;;
 D) ARGS="--out_path outputs/eval_dev300_D --topn 7 --use_fusion True --ext_path $EXT/external_reranked_full.json --sim_scores_path $EXT/external_sims_full.json --video_feature_dir /home/bn/videoufo_research/features/external_top7" ;;
 *) echo "usage: $0 A|B|C|D"; exit 2 ;;
esac
exec "$PY" -u eval_evcap_fix_video_llama_xclip_msrvtt_fusion_gate_prompt5.py $COMMON $ARGS
