# file: scripts/eval_evcap_fix_video_llama_xclip_msrvtt_quality_video_batch.sh
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER/..

EXP_NAME=eval_evcap_msrvtt_quality_video
NOCAPS_OUT_PATH=outputs/$EXP_NAME

TIME_START=$(date "+%Y-%m-%d-%H-%M-%S")
# LOG_FOLDER=logs/${EXP_NAME}_EVAL
LOG_FOLDER=logs/eval_batch
mkdir -p $LOG_FOLDER
FILE_NAME=top5_bs4_load_video_feature_prompt1

NOCAPS_LOG_FILE="$LOG_FOLDER/eval_evcap_${FILE_NAME}_${TIME_START}.log"

CUDA_VISIBLE_DEVICES="2,3" OMP_NUM_THREADS=2 torchrun --nproc_per_node 2 ./eval_evcap_fix_video_llama_xclip_msrvtt_video_batch.py \
    --distributed True \
    --test_yaml datasets/MSRVTT-v2/test_16frames.yaml \
    --out_path=$NOCAPS_OUT_PATH \
    --ckpt_name train_evcap_${FILE_NAME}.pt \
    --bs 4 \
    --topn 5 \
    --max_num_frames 16 \
    --num_video_query_token 32 \
    --retriever True \
    --ext_path ext_data/rerank_sentence_by_sentence_as_inference_rouge_quality_no_duplicate_with_video_ids.json \
    --prompt_path prompts/prompt_evcap_msrvtt_detail_smallcap.txt \
    |& tee -a  ${NOCAPS_LOG_FILE}

echo "==========================COCO EVAL================================"
python evaluation/cocoeval.py --result_file_path $NOCAPS_OUT_PATH/eval_evcap_${FILE_NAME}_generated_captions2.json \
    --eval_file $NOCAPS_OUT_PATH/eval_evcap_${FILE_NAME}2.json \
    |& tee -a  ${NOCAPS_LOG_FILE}