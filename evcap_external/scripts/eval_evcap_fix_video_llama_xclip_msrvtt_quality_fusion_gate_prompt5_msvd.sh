SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER/..

EXP_NAME=eval_evcap_msrvtt_quality
DEVICE=0
NOCAPS_OUT_PATH=outputs/old/$EXP_NAME

TIME_START=$(date "+%Y-%m-%d-%H-%M-%S")
LOG_FOLDER=logs/eval_quality_video
mkdir -p $LOG_FOLDER
FILE_NAME=top3_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_msvd

NOCAPS_LOG_FILE="$LOG_FOLDER/eval_evcap_${FILE_NAME}_${TIME_START}.log"

python -u eval_evcap_fix_video_llama_xclip_msrvtt_fusion_gate_prompt5_msvd.py \
    --device cuda:$DEVICE \
    --test_yaml datasets/MSVD/train_16frames.yaml \
    --out_path=$NOCAPS_OUT_PATH \
    --ckpt_name train_evcap_${FILE_NAME}.pt \
    --bs 1 \
    --topn 3 \
    --random_seed 42 \
    --max_num_frames 16 \
    --num_video_query_token 32 \
    --load_video_feature True \
    --retriever True \
    --ext_path ext_data/msvd_rerank_sentence_by_sentence_as_inference_rouge_quality_with_video_ids.json \
    --prompt_path prompts/prompt_evcap_msrvtt_detail_smallcap_prompt5.txt \
    |& tee -a  ${NOCAPS_LOG_FILE}

echo "==========================COCO EVAL================================"
python evaluation/cocoeval.py --result_file_path $NOCAPS_OUT_PATH/eval_evcap_${FILE_NAME}_generated_captions.json \
    --eval_file $NOCAPS_OUT_PATH/eval_evcap_${FILE_NAME}.json \
    |& tee -a  ${NOCAPS_LOG_FILE}