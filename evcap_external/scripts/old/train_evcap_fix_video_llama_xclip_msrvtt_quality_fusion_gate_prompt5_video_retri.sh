
FILE_NAME=text_top7_verb_noun_top2_bs1_load_video_feature_sim_score_cross_gating_fusion_not_share_linear_prompt8

TIME_START=$(date "+%Y-%m-%d-%H-%M-%S")
LOG_FOLDER=logs/train_retri_video
mkdir -p $LOG_FOLDER
NOCAPS_LOG_FILE="$LOG_FOLDER/train_evcap_${FILE_NAME}_${TIME_START}.log"

CUDA_VISIBLE_DEVICES="2,3" OMP_NUM_THREADS=2 torchrun --nproc_per_node 2 --master_port 29509 ./train_evcap_fix_video_llama_xclip_msrvtt_fusion_gate_prompt5_video_retri.py \
    --out_dir outputs/train_evcap_msrvtt_quality_retri_video \
    --ckpt_name train_evcap_${FILE_NAME}.pt \
    --train_yaml datasets/MSRVTT-v2/train_16frames.yaml \
    --bs 1 \
    --max_num_frames 16 \
    --num_video_query_token 32 \
    --vit_precision fp16 \
    --retriever True \
    --freeze_video_qformer True \
    --freeze_llm True \
    --freeze_vit True \
    --freeze_qformer True \
    --freeze_qformer_txt True \
    --load_video_feature True \
    --share_query_tokens True \
    --topn 7 \
    --video_topn 2 \
    --random_seed 42 \
    --ext_path ext_data/rerank_sentence_by_sentence_as_inference_rouge_quality_no_duplicate_with_video_ids.json \
    --sim_scores_path ext_data/msrvtt_train_allGT_avg_frames_xclip_retrieve_20_captions_no_duplicate_sim_scores.json \
    --video_ext_path ext_data/retrieved_video_verbs_nouns_per_video.json \
    --epochs 1 \
    --prompt_path prompts/prompt_evcap_msrvtt_detail_smallcap_prompt8.txt \
    |& tee -a ${NOCAPS_LOG_FILE}