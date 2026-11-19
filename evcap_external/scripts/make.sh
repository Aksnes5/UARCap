SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER/..

EXP_NAME=eval_evcap_msrvtt_quality
DEVICE=0
NOCAPS_OUT_PATH=outputs/old/$EXP_NAME

TIME_START=$(date "+%Y-%m-%d-%H-%M-%S")
LOG_FOLDER=logs/${EXP_NAME}_EVAL
mkdir -p $LOG_FOLDER
FILE_NAME=fix_video_llama_xclip_20_avg_frames_allGT_msrvtt_16frames_rouge_sentence_as_inference_top10_smallcap_quality_use_basepretrained_retriever

NOCAPS_LOG_FILE="$LOG_FOLDER/eval_evcap_${FILE_NAME}_${TIME_START}.log"

python -u eval_evcap_fix_video_llama_xclip_msrvtt_global_make.py \
    --device cuda:$DEVICE \
    --test_yaml datasets/MSVD/train_alternate_16frames.yaml \
    --out_path=$NOCAPS_OUT_PATH \
    --ckpt_name train_evcap_${FILE_NAME}.pt \
    --bs 1 \
    --max_num_frames 16 \
    --num_video_query_token 32 \
    --retriever False \
    --ext_path None \
    --prompt_path prompts/prompt_evcap_msrvtt.txt \