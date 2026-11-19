FILE_NAME=top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_train_videoqformer_prompt5

TIME_START=$(date "+%Y-%m-%d-%H-%M-%S")
LOG_FOLDER=logs/train_video
mkdir -p $LOG_FOLDER
NOCAPS_LOG_FILE="$LOG_FOLDER/train_evcap_${FILE_NAME}_${TIME_START}.log"

CUDA_VISIBLE_DEVICES="0,1" OMP_NUM_THREADS=2 torchrun --nproc_per_node 2 --master_port 29508 ./train_evcap_fix_video_llama_xclip_msrvtt_fusion_gate_prompt5.py  \
    --out_dir outputs/train_evcap_msrvtt_quality \
    --ckpt_name train_evcap_${FILE_NAME}.pt \
    --train_yaml datasets/MSRVTT-v2/train_16frames.yaml \
    --bs 1 \
    --max_num_frames 16 \
    --num_video_query_token 32 \
    --vit_precision fp16 \
    --retriever True \
    --freeze_video_qformer False \
    --freeze_llm True \
    --freeze_vit True \
    --freeze_qformer True \
    --load_video_feature True \
    --share_query_tokens True \
    --llm_model vicuna-7b-v1.3 \
    --topn 7 \
    --random_seed 42 \
    --ext_path ext_data/rerank_sentence_by_sentence_as_inference_rouge_quality_no_duplicate_with_video_ids.json \
    --epochs 1 \
    --prompt_path prompts/prompt_evcap_msrvtt_detail_smallcap_prompt5.txt \
    |& tee -a ${NOCAPS_LOG_FILE}