#!/usr/bin/env bash
set -x
set -euo pipefail

export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
# 为 CUDA >= 10.2 的 cuBLAS 提供确定性工作区配置，减少 PyTorch deterministic 警告。
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
RUN_TS="$(date +"%Y%m%d_%H%M%S")"
RAW_ARGS=("$@")

log() {
    echo "[INFO] $*"
}

die() {
    echo "[ERROR] $*" >&2
    exit 1
}

require_non_empty() {
    local name="$1"
    local value="${2:-}"
    [[ -n "${value}" ]] || die "参数 ${name} 不能为空"
}

require_file() {
    local p="$1"
    [[ -f "${p}" ]] || die "文件不存在: ${p}"
}

to_abs_path() {
    local p="$1"
    if [[ "${p}" == /* ]]; then
        echo "${p}"
    else
        echo "${ROOT_DIR}/${p}"
    fi
}

csv_to_array() {
    local input="$1"
    local out_name="$2"
    local old_ifs="${IFS}"
    local raw_arr=()
    IFS=','
    read -r -a raw_arr <<< "${input}"
    IFS="${old_ifs}"
    local cleaned=()
    local item
    for item in "${raw_arr[@]}"; do
        item="$(echo "${item}" | xargs)"
        [[ -n "${item}" ]] || continue
        cleaned+=("${item}")
    done
    eval "${out_name}=(\"\${cleaned[@]}\")"
}

link_latest() {
    local target_file="$1"
    local latest_name="$2"
    (
        cd "$(dirname "${target_file}")" && ln -sfn "$(basename "${target_file}")" "${latest_name}"
    )
}

print_help() {
    cat <<'EOF'
统一流程脚本：训练 / 推理 / 评估 / 全流程
产物统一保存于: <output_base_dir>/<exp_tag>/

用法:
  bash scripts/pipeline.sh --stage <train|infer|eval|all> [选项]

核心选项:
  --stage                    执行阶段: train / infer / eval / all
  --exp_tag                  实验标识（同时作为 model_tag）
  --output_base_dir          输出根目录
  --root_dir                 项目根目录（默认仓库根）

训练相关:
  --use_epl                  True/False
  --lr                       学习率
  --mode                     训练模式（如 aug-wo-pc）
  --tokenizer_path           tokenizer 路径
  --train_model_path         训练初始化 base model 路径
  --train_data_path          训练数据路径
  --conf_version             压缩配置版本（默认 v1）
  --train_gpus               训练卡号，逗号分隔（默认 0,1,2,3,4,5,6,7）
  --max_length               默认 4096
  --epochs                   默认 5
  --save_steps               默认 2
  --micro_batch_size         默认 2
  --gradient_accumulation_steps  默认 4
  --warmup_ratio             默认 0.05
  --warmup_steps             默认 0
  --lr_scheduler_type        默认 cosine
  --deepspeed_config         默认 configs/ds_z3_offload_config.json
  --seed                     随机种子（默认 42）

推理相关:
  --infer_model_path         直接指定推理模型路径（不需训练目录）；不传则从 <exp_root>/train/checkpoint-* 取
  --tokenizer_path           tokenizer 路径；使用 --infer_model_path 直接推理时可不传，默认与 infer_model_path 一致
  --ckpt                     checkpoint 编号；不传自动取最新（仅当未传 infer_model_path 时有效）
  --repetition_penalty       默认 1.1
  --target_gpus              推理卡号，逗号分隔（默认 0,1,2,3,4,5,6,7）
  --process_per_gpu          每卡进程数（默认 4）
  --max_new_tokens           默认 10240
  --spec_decode             是否启用投机解码 true/false（默认 false）

兼容参数:
  --model_path               兼容旧参数：按 stage 映射到 train/infer；stage=all 时仅用于训练

评估相关:
  --eval_method              默认 normal
  --datasets                 逗号分隔（默认 mmlu,gsm8k,gpqa,bbh）
  --comp_config              默认 configs/LightThinker/qwen/v1.json
  --model_type               默认 qwen
  --bos_token                默认 <|im_start|>
  --eos_token                默认 <|im_end|>
  --cache_size               默认 1024
  --interaction              true/false（默认 false）
EOF
}

resolve_ckpt() {
    local train_dir="$1"
    local ckpt="${2:-}"
    if [[ -n "${ckpt}" ]]; then
        echo "${ckpt}"
        return 0
    fi
    [[ -d "${train_dir}" ]] || die "训练目录不存在，无法自动推断ckpt: ${train_dir}"
    local latest_dir
    latest_dir="$(ls -d "${train_dir}"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)"
    [[ -n "${latest_dir}" ]] || die "未找到 checkpoint-* 目录，请先训练或手动传 --ckpt"
    basename "${latest_dir}" | sed 's/^checkpoint-//'
}

prepare_experiment_root() {
    require_non_empty "--exp_tag" "${EXP_TAG}"
    require_non_empty "--output_base_dir" "${OUTPUT_BASE_DIR}"
    EXP_ROOT="${OUTPUT_BASE_DIR}/${EXP_TAG}"
    mkdir -p "${EXP_ROOT}"
    local snapshot="${EXP_ROOT}/pipeline_${RUN_TS}.sh"
    cp -f "${SCRIPT_PATH}" "${snapshot}"
    link_latest "${snapshot}" "pipeline_latest.sh"
}

save_param_snapshot() {
    local snapshot="${EXP_ROOT}/run_${RUN_TS}.txt"
    {
        echo "timestamp=${RUN_TS}"
        echo "script=${SCRIPT_PATH}"
        echo "stage=${STAGE}"
        echo "root_dir=${ROOT_DIR}"
        echo "exp_root=${EXP_ROOT}"
        echo -n "argv="
        printf '%q ' "${RAW_ARGS[@]}"
        echo
        echo "--- resolved params ---"
        echo "exp_tag=${EXP_TAG}"
        echo "output_base_dir=${OUTPUT_BASE_DIR}"
        echo "use_epl=${USE_EPL}"
        echo "lr=${LR}"
        echo "mode=${MODE}"
        echo "tokenizer_path=${TOKENIZER_PATH}"
        echo "train_model_path=${TRAIN_MODEL_PATH}"
        echo "infer_model_path=${INFER_MODEL_PATH}"
        echo "legacy_model_path=${MODEL_PATH}"
        echo "train_data_path=${TRAIN_DATA_PATH}"
        echo "model_type=${MODEL_TYPE}"
        echo "datasets=${DATASETS}"
        echo "ckpt=${CKPT}"
        echo "target_gpus=${TARGET_GPUS}"
        echo "process_per_gpu=${PROCESS_PER_GPU}"
        echo "max_new_tokens=${MAX_NEW_TOKENS}"
        echo "spec_decode=${SPEC_DECODE}"
        echo "seed=${SEED}"
    } > "${snapshot}"
    link_latest "${snapshot}" "run_latest.txt"
}

run_train() {
    require_non_empty "--use_epl" "${USE_EPL}"
    require_non_empty "--lr" "${LR}"
    require_non_empty "--mode" "${MODE}"
    require_non_empty "--tokenizer_path" "${TOKENIZER_PATH}"
    require_non_empty "--train_model_path" "${TRAIN_MODEL_PATH}"
    require_non_empty "--train_data_path" "${TRAIN_DATA_PATH}"

    local train_py="${ROOT_DIR}/LightThinker/train.py"
    require_file "${train_py}"

    local output_dir="${EXP_ROOT}/train"
    mkdir -p "${output_dir}"
    local log_file="${output_dir}/train_${RUN_TS}.log"

    local train_comp_config="${ROOT_DIR}/configs/LightThinker/${MODEL_TYPE}/${CONF_VERSION}.json"
    require_file "${train_comp_config}"

    local ds_cfg
    ds_cfg="$(to_abs_path "${DEEPSPEED_CONFIG}")"
    require_file "${ds_cfg}"

    log "开始训练..."
    deepspeed --master_port=54212 --include "localhost:${TRAIN_GPUS}" "${train_py}" \
        --model_type "${MODEL_TYPE}" \
        --model_path "${TRAIN_MODEL_PATH}" \
        --tokenizer_path "${TOKENIZER_PATH}" \
        --train_path "${TRAIN_DATA_PATH}" \
        --output_dir "${output_dir}" \
        --max_length "${MAX_LENGTH}" \
        --compress_config "${train_comp_config}" \
        --bos_token "${BOS_TOKEN}" \
        --eos_token "${EOS_TOKEN}" \
        --see_current "${SEE_CURRENT}" \
        --bi_directional "${BI_DIRECTIONAL}" \
        --diagonal "${DIAGONAL}" \
        --mode "${MODE}" \
        --exclude_continue "${EXCLUDE_CONTINUE}" \
        --qkv "${QKV}" \
        --freeze_model "${FREEZE_MODEL}" \
        --train_on_input "${TRAIN_ON_INPUT}" \
        --output_compress_instruction "${OUTPUT_COMPRESS_INSTRUCTION}" \
        --epochs "${EPOCHS}" \
        --lr "${LR}" \
        --save_steps "${SAVE_STEPS}" \
        --deepspeed "${ds_cfg}" \
        --micro_batch_size "${MICRO_BATCH_SIZE}" \
        --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
        --warmup_ratio "${WARMUP_RATIO}" \
        --warmup_steps "${WARMUP_STEPS}" \
        --hybrid "${HYBRID}" \
        --prefill_compress "${PREFILL_COMPRESS}" \
        --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
        --use_EPL "${USE_EPL}" \
        --seed "${SEED}" 2>&1 | tee "${log_file}"
    local exit_code=${PIPESTATUS[0]}
    link_latest "${log_file}" "train_latest.log"
    [[ "${exit_code}" -eq 0 ]] || die "训练失败，日志: ${log_file}"
}

run_infer() {
    local infer_py="${ROOT_DIR}/LightThinker/inference.py"
    require_file "${infer_py}"

    local infer_model_path
    local resolved_ckpt
    local tokenizer_path_infer

    # 若指定了可用的 --infer_model_path，则直接用于推理（无需训练目录）
    if [[ -n "${INFER_MODEL_PATH}" ]] && [[ -e "${INFER_MODEL_PATH}" ]]; then
        infer_model_path="${INFER_MODEL_PATH}"
        resolved_ckpt="-1"
        tokenizer_path_infer="${TOKENIZER_PATH:-${INFER_MODEL_PATH}}"
        log "使用指定模型路径进行推理: ${infer_model_path}"
    else
        require_non_empty "--tokenizer_path" "${TOKENIZER_PATH}"
        tokenizer_path_infer="${TOKENIZER_PATH}"
        local train_dir="${EXP_ROOT}/train"
        resolved_ckpt="$(resolve_ckpt "${train_dir}" "${CKPT}")"
        infer_model_path="${train_dir}/checkpoint-${resolved_ckpt}"
    fi

    local output_tag="${EXP_ROOT}/inference"
    local model_short_tag="${EXP_TAG}"

    mkdir -p "${output_tag}/inference_log/false_false"
    mkdir -p "${output_tag}/inference_log/false_true"
    mkdir -p "${output_tag}/inference_log/true_false"
    mkdir -p "${output_tag}/inference_log/true_true"

    local gpu_arr=()
    csv_to_array "${TARGET_GPUS}" gpu_arr
    [[ "${#gpu_arr[@]}" -gt 0 ]] || die "--target_gpus 不能为空"
    [[ "${PROCESS_PER_GPU}" -gt 0 ]] || die "--process_per_gpu 必须 > 0"
    local split_size=$(( ${#gpu_arr[@]} * PROCESS_PER_GPU ))

    local ds_arr=()
    csv_to_array "${DATASETS}" ds_arr
    [[ "${#ds_arr[@]}" -gt 0 ]] || die "--datasets 不能为空"

    log "开始推理... (ckpt=${resolved_ckpt}, split_size=${split_size})"
    local pids=()
    local logical_id=0
    for device in "${gpu_arr[@]}"; do
        local start_index_0based=$(( logical_id * PROCESS_PER_GPU ))
        local end_index_0based=$(( start_index_0based + PROCESS_PER_GPU - 1 ))
        log ">>> Launching on GPU ${device}"
        for ((idx=start_index_0based; idx<=end_index_0based; idx++)); do
            local real_index=$(( idx + 1 ))
            log "    Starting task index ${real_index}/${split_size}"
            CUDA_VISIBLE_DEVICES="${device}" nohup python "${infer_py}" \
                --model_tag "${EXP_TAG}" \
                --model_short_tag "${model_short_tag}" \
                --ckpt "${resolved_ckpt}" \
                --tokenizer_path "${tokenizer_path_infer}" \
                --compress_config "${COMP_CONFIG}" \
                --max_new_tokens "${MAX_NEW_TOKENS}" \
                --repetition_penalty "${REPETITION_PENALTY}" \
                --output_tag "${output_tag}" \
                --model_type "${MODEL_TYPE}" \
                --bos_token "${BOS_TOKEN}" \
                --eos_token "${EOS_TOKEN}" \
                --rolling_rope "false" \
                --diagonal "false" \
                --bi_directional "false" \
                --see_current "false" \
                --exclude_continue "false" \
                --output_compress_instruction "None" \
                --prefill_compress "false" \
                --compress_prompt "false" \
                --update_attention_method "local" \
                --split_size "${split_size}" \
                --use_EPL "${USE_EPL}" \
                --model_path "${infer_model_path}" \
                --index "${real_index}" \
                --spec_decode "${SPEC_DECODE}" \
                --datasets "${ds_arr[@]}" \
                > "${output_tag}/inference_log/false_false/${real_index}_${model_short_tag}_${resolved_ckpt}.txt" 2>&1 &
            pids+=("$!")
            sleep 2
        done
        logical_id=$((logical_id + 1))
    done

    log "所有推理进程已启动，等待完成..."
    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            failed=1
        fi
    done
    [[ "${failed}" -eq 0 ]] || die "某推理子进程失败，请检查日志: ${output_tag}/inference_log"
}

run_eval() {
    require_non_empty "--tokenizer_path" "${TOKENIZER_PATH}"
    local eval_py="${ROOT_DIR}/evaluation/eval_file.py"
    require_file "${eval_py}"

    local infer_base="${EXP_ROOT}/inference"
    local ds_arr=()
    csv_to_array "${DATASETS}" ds_arr
    [[ "${#ds_arr[@]}" -gt 0 ]] || die "--datasets 不能为空"

    local comp_cfg
    comp_cfg="$(to_abs_path "${COMP_CONFIG}")"
    require_file "${comp_cfg}"

    for ds in "${ds_arr[@]}"; do
        local base_path="${infer_base}/${ds}"
        [[ -d "${base_path}" ]] || die "推理结果目录不存在: ${base_path}"
        shopt -s nullglob
        local files=( "${base_path}"/*.jsonl )
        shopt -u nullglob
        [[ "${#files[@]}" -gt 0 ]] || die "未找到评估文件: ${base_path}/*.jsonl"

        local eval_result_dir="${EXP_ROOT}/eval/${EVAL_METHOD}/${MODEL_TYPE}/${ds}"
        mkdir -p "${eval_result_dir}"
        local eval_log_file="${eval_result_dir}/eval_${RUN_TS}.log"

        log "开始评估数据集: ${ds}"
        local python_args=(
            "${eval_py}"
            --method "${EVAL_METHOD}"
            --tokenizer_path "${TOKENIZER_PATH}"
            --comp_config "${comp_cfg}"
            --model_type "${MODEL_TYPE}"
            --dataset "${ds}"
            --model_tag "${EXP_TAG}"
            --files "${files[@]}"
            --cache_size "${CACHE_SIZE}"
            --bos_token "${BOS_TOKEN}"
            --eos_token "${EOS_TOKEN}"
        )
        [[ "${INTERACTION}" == "true" ]] && python_args+=(--interaction)

        python "${python_args[@]}" 2>&1 | tee -a "${eval_log_file}"
        local exit_code=${PIPESTATUS[0]}

        local src_eval_dir="${ROOT_DIR}/eval_results/${EVAL_METHOD}/${MODEL_TYPE}/${EXP_TAG}/${ds}"
        [[ -d "${src_eval_dir}" ]] && cp -a "${src_eval_dir}/." "${eval_result_dir}/"
        link_latest "${eval_log_file}" "eval_latest.log"
        [[ "${exit_code}" -eq 0 ]] || die "评估失败(${ds})，日志: ${eval_log_file}"
    done
}

# ---------- 默认值 ----------
STAGE=""
ROOT_DIR="${PROJECT_ROOT}"
EXP_TAG=""
OUTPUT_BASE_DIR=""
EXP_ROOT=""

# 训练参数
USE_EPL="false"
LR=""
MODE=""
TOKENIZER_PATH=""
TRAIN_MODEL_PATH=""
INFER_MODEL_PATH=""
MODEL_PATH=""
TRAIN_DATA_PATH=""
CONF_VERSION="v1"
TRAIN_GPUS="0,1,2,3,4,5,6,7"
MAX_LENGTH=4096
EPOCHS=5
SAVE_STEPS=2
MICRO_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=4
WARMUP_RATIO=0.05
WARMUP_STEPS=0
LR_SCHEDULER_TYPE="cosine"
DEEPSPEED_CONFIG="configs/ds_z3_offload_config.json"
SEED=42
SEE_CURRENT="false"
BI_DIRECTIONAL="false"
DIAGONAL="false"
EXCLUDE_CONTINUE="false"
QKV="no"
FREEZE_MODEL="false"
TRAIN_ON_INPUT="false"
HYBRID="false"
OUTPUT_COMPRESS_INSTRUCTION="None"
PREFILL_COMPRESS="false"

# 推理参数
CKPT=""
REPETITION_PENALTY="1.1"
SPEC_DECODE="false"

# 评估参数
EVAL_METHOD="normal"
DATASETS="mmlu,gsm8k,gpqa,bbh"
COMP_CONFIG="configs/LightThinker/qwen/v1.json"
MODEL_TYPE="qwen"
BOS_TOKEN="<|im_start|>"
EOS_TOKEN="<|im_end|>"
CACHE_SIZE="1024"
INTERACTION="false"

# 推理并发参数
TARGET_GPUS="0,1,2,3,4,5,6,7"
PROCESS_PER_GPU=4
MAX_NEW_TOKENS=10240

# ---------- 参数解析 ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        # 通用参数
        --stage) STAGE="${2:-}"; shift 2 ;;
        --root_dir) ROOT_DIR="${2:-}"; shift 2 ;;
        --exp_tag) EXP_TAG="${2:-}"; shift 2 ;;
        --output_base_dir) OUTPUT_BASE_DIR="${2:-}"; shift 2 ;;

        # 训练参数
        --use_epl) USE_EPL="${2:-}"; shift 2 ;;
        --lr) LR="${2:-}"; shift 2 ;;
        --mode) MODE="${2:-}"; shift 2 ;;
        --tokenizer_path) TOKENIZER_PATH="${2:-}"; shift 2 ;;
        --train_model_path) TRAIN_MODEL_PATH="${2:-}"; shift 2 ;;
        --infer_model_path) INFER_MODEL_PATH="${2:-}"; shift 2 ;;
        --model_path) MODEL_PATH="${2:-}"; shift 2 ;;
        --train_data_path) TRAIN_DATA_PATH="${2:-}"; shift 2 ;;
        --conf_version) CONF_VERSION="${2:-}"; shift 2 ;;
        --train_gpus) TRAIN_GPUS="${2:-}"; shift 2 ;;
        --max_length) MAX_LENGTH="${2:-}"; shift 2 ;;
        --epochs) EPOCHS="${2:-}"; shift 2 ;;
        --save_steps) SAVE_STEPS="${2:-}"; shift 2 ;;
        --micro_batch_size) MICRO_BATCH_SIZE="${2:-}"; shift 2 ;;
        --gradient_accumulation_steps) GRADIENT_ACCUMULATION_STEPS="${2:-}"; shift 2 ;;
        --warmup_ratio) WARMUP_RATIO="${2:-}"; shift 2 ;;
        --warmup_steps) WARMUP_STEPS="${2:-}"; shift 2 ;;
        --lr_scheduler_type) LR_SCHEDULER_TYPE="${2:-}"; shift 2 ;;
        --deepspeed_config) DEEPSPEED_CONFIG="${2:-}"; shift 2 ;;
        --seed) SEED="${2:-}"; shift 2 ;;
        --see_current) SEE_CURRENT="${2:-}"; shift 2 ;;
        --bi_directional) BI_DIRECTIONAL="${2:-}"; shift 2 ;;
        --diagonal) DIAGONAL="${2:-}"; shift 2 ;;
        --exclude_continue) EXCLUDE_CONTINUE="${2:-}"; shift 2 ;;
        --qkv) QKV="${2:-}"; shift 2 ;;
        --freeze_model) FREEZE_MODEL="${2:-}"; shift 2 ;;
        --train_on_input) TRAIN_ON_INPUT="${2:-}"; shift 2 ;;
        --hybrid) HYBRID="${2:-}"; shift 2 ;;
        --output_compress_instruction) OUTPUT_COMPRESS_INSTRUCTION="${2:-}"; shift 2 ;;
        --prefill_compress) PREFILL_COMPRESS="${2:-}"; shift 2 ;;

        # 推理参数
        --ckpt) CKPT="${2:-}"; shift 2 ;;
        --repetition_penalty) REPETITION_PENALTY="${2:-}"; shift 2 ;;
        --target_gpus) TARGET_GPUS="${2:-}"; shift 2 ;;
        --process_per_gpu) PROCESS_PER_GPU="${2:-}"; shift 2 ;;
        --max_new_tokens) MAX_NEW_TOKENS="${2:-}"; shift 2 ;;
        --spec_decode) SPEC_DECODE="${2:-}"; shift 2 ;;

        # 评估参数
        --eval_method) EVAL_METHOD="${2:-}"; shift 2 ;;
        --datasets) DATASETS="${2:-}"; shift 2 ;;
        --comp_config) COMP_CONFIG="${2:-}"; shift 2 ;;
        --model_type) MODEL_TYPE="${2:-}"; shift 2 ;;
        --bos_token) BOS_TOKEN="${2:-}"; shift 2 ;;
        --eos_token) EOS_TOKEN="${2:-}"; shift 2 ;;
        --cache_size) CACHE_SIZE="${2:-}"; shift 2 ;;
        --interaction) INTERACTION="${2:-}"; shift 2 ;;
        -h|--help) print_help; exit 0 ;;
        *) die "未知参数: $1（使用 -h 查看帮助）" ;;
    esac
done

require_non_empty "--stage" "${STAGE}"
[[ -d "${ROOT_DIR}" ]] || die "root_dir 不存在: ${ROOT_DIR}"

# 兼容旧参数 --model_path：
# - stage=train: 作为 --train_model_path
# - stage=infer: 作为 --infer_model_path
# - stage=all: 仅作为 --train_model_path（infer 默认走训练产物）
if [[ -n "${MODEL_PATH}" ]]; then
    case "${STAGE}" in
        train)
            [[ -n "${TRAIN_MODEL_PATH}" ]] || TRAIN_MODEL_PATH="${MODEL_PATH}"
            ;;
        infer)
            [[ -n "${INFER_MODEL_PATH}" ]] || INFER_MODEL_PATH="${MODEL_PATH}"
            ;;
        all)
            [[ -n "${TRAIN_MODEL_PATH}" ]] || TRAIN_MODEL_PATH="${MODEL_PATH}"
            ;;
    esac
fi

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
prepare_experiment_root
save_param_snapshot

log "stage=${STAGE}"
log "root_dir=${ROOT_DIR}"
log "exp_tag=${EXP_TAG}"
log "output_base_dir=${OUTPUT_BASE_DIR}"
log "exp_root=${EXP_ROOT}"

case "${STAGE}" in
    train) run_train ;;
    infer) 
        run_infer
        run_eval ;;
    eval) run_eval ;;
    all)
        run_train
        run_infer
        run_eval
        ;;
    *) die "--stage 仅支持: train / infer / eval / all" ;;
esac

log "执行完成: ${STAGE}"


# # 运行示例 train
# bash /mnt/RRcot/scripts/pipeline.sh \
#   --stage train \
#   --exp_tag vanilla_qwen \
#   --output_base_dir /mnt/RRcot/experiments \
#   --use_epl false \
#   --lr 1e-5 \
#   --mode normal \
#   --model_type qwen \
#   --tokenizer_path /mnt/hf_models/Qwen2.5-1.5B-Instruct \
#   --train_model_path /mnt/hf_models/DeepSeek-R1-Distill-Qwen-1.5B \
#   --train_data_path /mnt/RRcot/data/train/train_debug.jsonl \
#   --train_gpus 0,1,2,3

# # 运行示例 all
# bash /mnt/mem-co-t/scripts/pipeline.sh \
#   --stage train \
#   --exp_tag epl_mtp_1 \
#   --output_base_dir /mnt/rrcot_test/experiments \
#   --use_epl true \
#   --lr 2e-5 \
#   --mode aug-wo-pc-apa-mtp \
#   --model_type qwen \
#   --tokenizer_path /mnt/models/qwen2-0.5B-Instruct \
#   --train_model_path /mnt/models/qwen2-0.5B-Instruct \
#   --train_data_path /mnt/mem-co-t/data/train/train_test.jsonl \
#   --train_gpus 0,5,6,7 \
#   --target_gpus 0,5,6,7 \
#   --process_per_gpu 1 \
#   --comp_config "adaptive_mtp_v1" \
#   --conf_version "adaptive_mtp_v1" \
#   --max_length 2048 \
#   --spec_decode True \
#   --datasets mmlu,gsm8k,gpqa,bbh


# # 运行示例 infer
# bash /mnt/lxy/RRcot/scripts/pipeline.sh \
#   --stage infer \
#   --infer_model_path /mnt/rrcot_test/epl_apa_mtp_w3e-1/train/checkpoint-245 \
#   --output_base_dir /mnt/RRcot/experiments/debug_infer_spec_decode \
#   --use_epl false \
#   --spec_decode true \
#   --model_type qwen \
#   --tokenizer_path /mnt/hf_models/DeepSeek-R1-Distill-Qwen-1.5B \
#   --target_gpus 0 \
#   --process_per_gpu 1 \
#   --datasets mmlu
