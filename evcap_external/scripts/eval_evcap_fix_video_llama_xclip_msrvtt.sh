SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER/..

EXP_NAME=eval_evcap_msrvtt
DEVICE=2
NOCAPS_OUT_PATH=outputs/old/$EXP_NAME

TIME_START=$(date "+%Y-%m-%d-%H-%M-%S")
LOG_FOLDER=logs/eval
mkdir -p $LOG_FOLDER
FILE_NAME=bs1_no_retriever_prompt0_basepretrained_test

NOCAPS_LOG_FILE="$LOG_FOLDER/eval_evcap_${FILE_NAME}_${TIME_START}.log"

python -u eval_evcap_fix_video_llama_xclip_msrvtt.py \
    --device cuda:$DEVICE \
    --test_yaml datasets/MSRVTT-v2/test_16frames.yaml \
    --out_path=$NOCAPS_OUT_PATH \
    --ckpt_name outputs/old/train_evcap_msrvtt/train_evcap_top7_bs1_no_retriever_prompt0.pt \
    --bs 1 \
    --topn 7 \
    --max_num_frames 16 \
    --num_video_query_token 32 \
    --retriever False \
    --ext_path None \
    --prompt_path prompts/prompt_evcap_msrvtt.txt \
    |& tee -a  ${NOCAPS_LOG_FILE}

# echo "==========================COCO EVAL================================"
# python evaluation/cocoeval.py --result_file_path $NOCAPS_OUT_PATH/eval_evcap_${FILE_NAME}_generated_captions.json \
#     --eval_file $NOCAPS_OUT_PATH/eval_evcap_${FILE_NAME}.json \
#     |& tee -a  ${NOCAPS_LOG_FILE}