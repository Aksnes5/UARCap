#!/usr/bin/env bash
# Read only. Supply the verified roots and the existing Python interpreter explicitly.
set -euo pipefail
[[ $# == 4 ]] || { printf 'Usage: bash -s -- PROJECT DISPLAY MSRVTT_ROOT PYTHON\n' >&2; exit 2; }
project_root=$1
display_root=$2
msrvtt_root=$3
python_exe=$4
[[ $project_root = /* && $display_root = /* && $msrvtt_root = /* && $python_exe = /* ]] || exit 2

inspect() {
    local target=$1 size
    printf '\nASSET %s\n' "$target"
    if [[ ! -e $target ]]; then printf 'MISSING_AT_THIS_PATH\n'; return; fi
    readlink -e -- "$target"
    stat -Lc 'bytes=%s mtime=%y type=%F' -- "$target"
    if [[ ! -r $target ]]; then printf 'UNREADABLE\n'; return; fi
    printf 'READABLE\n'
    if [[ -f $target ]]; then
        size=$(stat -Lc %s -- "$target")
        if (( size <= 16*1024*1024 )); then sha256sum -- "$target"
        else printf 'LARGE_FILE_HASH_DEFERRED\n'; fi
    fi
}

uname -srmo
cat /etc/os-release
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free,driver_version --format=csv || printf 'GPU_QUERY_FAILED\n'
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv || printf 'COMPUTE_QUERY_FAILED\n'
ps -eo pid,comm | awk '$2 ~ /^(python|torchrun|accelerate)/'
df -hP -- "$project_root" "$display_root" "$msrvtt_root" || printf 'FILESYSTEM_QUERY_INCOMPLETE\n'
inspect "$project_root"
inspect "$project_root/gpu.lock"
inspect "$python_exe"
if [[ -x $python_exe ]]; then
    "$python_exe" -B -S --version
    shopt -s nullglob
    env_root=$(dirname "$(dirname "$python_exe")")
    for meta in "$env_root"/lib/python*/site-packages/{torch,transformers,timm,einops,tokenizers}-*.dist-info/METADATA; do
        printf '\nPACKAGE %s\n' "$meta"
        awk '/^(Name|Version):/ {print; n++; if(n==2) exit}' "$meta"
    done
fi
inspect "$project_root/scripts/run_test.sh"
inspect "$project_root/evcap_external/test2990/test2990_16frames.yaml"
inspect "$project_root/features/videollama_faithful_v2"
inspect "$project_root/external/external_test_d7_cap.json"
inspect "$project_root/external/external_test_d7_sims.json"
inspect "$project_root/external/external_test_top20.json"
inspect "$project_root/manifests/msrvtt_test2990.tsv"
inspect "$project_root/manifests/msrvtt_untouched2690.tsv"
inspect "$project_root/evcap_external/outputs/eval_test2990_D1"
inspect "$project_root/evcap_external/outputs/eval_test2990_D3"
inspect "$project_root/evcap_external/outputs/eval_test2990_D7"
for target in "$msrvtt_root/test_16frm/frame_tsv/test_16frames_img_size224.img.tsv" "$msrvtt_root/test.label.tsv" "$msrvtt_root/test.caption.tsv"; do
    inspect "$target"
    inspect "${target%.*}.lineidx"
done
inspect "$display_root/lj/EVCapCode/outputs/train_evcap_msrvtt_quality_text/train_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review.pt"
inspect "$display_root/repro_evcap/models/vicuna-7b-v1.3"
for name in config.json generation_config.json tokenizer.model tokenizer_config.json special_tokens_map.json pytorch_model.bin.index.json model.safetensors.index.json; do
    inspect "$display_root/repro_evcap/models/vicuna-7b-v1.3/$name"
done
inspect "$display_root/repro_evcap/hf_cache"
inspect "$display_root/repro_evcap/torch_cache"
inspect "$display_root/repro_evcap/torch_cache/hub/checkpoints/eva_vit_g.pth"
inspect "$display_root/repro_evcap/torch_cache/hub/checkpoints/blip2_pretrained_flant5xxl.pth"
inspect "$project_root/evcap_external/download/videollama/baseline_2_7B_Pretrained.pth"
