SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER/..

EXP_NAME=eval_evcap_msrvtt_quality_retri_video
DEVICE=2
NOCAPS_OUT_PATH=outputs/$EXP_NAME

TIME_START=$(date "+%Y-%m-%d-%H-%M-%S")
LOG_FOLDER=logs/eval_retri_video
mkdir -p $LOG_FOLDER
FILE_NAME=text_top7_verb_noun_top3_bs1_load_video_feature_sim_score_cross_gating_fusion_not_share_linear_prompt8

NOCAPS_LOG_FILE="$LOG_FOLDER/eval_evcap_${FILE_NAME}_${TIME_START}.log"

python -u eval_evcap_fix_video_llama_xclip_msrvtt_fusion_gate_prompt5_video_retri3.py \
    --device cuda:$DEVICE \
    --test_yaml datasets/MSRVTT-v2/test_16frames.yaml \
    --out_path=$NOCAPS_OUT_PATH \
    --ckpt_name train_evcap_${FILE_NAME}.pt \
    --bs 1 \
    --epochs 1 \
    --topn 7 \
    --video_topn 3 \
    --random_seed 42 \
    --max_num_frames 16 \
    --num_video_query_token 32 \
    --freeze_video_qformer True \
    --vit_precision fp16 \
    --freeze_llm True \
    --freeze_vit True \
    --freeze_qformer True\
    --load_video_feature True \
    --share_query_tokens True \
    --retriever True \
    --ext_path ext_data/rerank_sentence_by_sentence_as_inference_rouge_quality_no_duplicate_with_video_ids.json \
    --sim_scores_path ext_data/msrvtt_train_allGT_avg_frames_xclip_retrieve_20_captions_no_duplicate_sim_scores.json \
    --video_ext_path ext_data/retrieved_video_verbs_nouns_per_video.json \
    --prompt_path prompts/prompt_evcap_msrvtt_detail_smallcap_prompt8.txt \
    |& tee -a  ${NOCAPS_LOG_FILE}

echo "==========================COCO EVAL================================"
python evaluation/cocoeval.py --result_file_path $NOCAPS_OUT_PATH/eval_evcap_${FILE_NAME}_generated_captions.json \
    --eval_file $NOCAPS_OUT_PATH/eval_evcap_${FILE_NAME}.json \
    |& tee -a  ${NOCAPS_LOG_FILE}