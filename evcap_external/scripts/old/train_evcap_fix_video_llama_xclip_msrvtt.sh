FILE_NAME=top7_bs1_no_retriever_prompt0

TIME_START=$(date "+%Y-%m-%d-%H-%M-%S")
NOCAPS_LOG_FILE="logs/train/train_evcap_${FILE_NAME}_${TIME_START}.log"

CUDA_VISIBLE_DEVICES="2,3" OMP_NUM_THREADS=2 torchrun --nproc_per_node 2 ./train_evcap_fix_video_llama_xclip_msrvtt.py  \
    --out_dir outputs/train_evcap_msrvtt \
    --ckpt_name train_evcap_${FILE_NAME}.pt \
    --train_yaml datasets/MSRVTT-v2/train_16frames.yaml \
    --bs 1 \
    --max_num_frames 16 \
    --num_video_query_token 32 \
    --retriever False \
    --freeze_video_qformer True \
    --topn 7 \
    --ext_path None \
    --epochs 1 \
    --prompt_path prompts/prompt_evcap_msrvtt.txt \
    |& tee -a ${NOCAPS_LOG_FILE}