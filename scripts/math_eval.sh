#!/bin/bash

PROMPT_TYPE="qwen25-math-cot"
DRAFT_MODEL="Qwen/Qwen2.5-Math-1.5B-Instruct"
TARGET_MODEL="Qwen/Qwen2.5-Math-7B-Instruct"
PRM="Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"
DRAFT_IP_ADDRESS="http://localhost:12340/v1"
TARGET_IP_ADDRESS="http://localhost:12341/v1"
PRM_IP_ADDRESS="http://localhost:12342/v1"
OUTPUT_DIR="outputs/draft_Qwen2.5-Math-1.5B-Instruct_target_Qwen2.5-Math-7B-Instruct_prm_Skywork-o1-Open-PRM-Qwen-2.5-1.5B/math_eval"

SPLIT="test"
NUM_TEST_SAMPLE=-1
for PRM_THRESHOLD in 0.7; do

DATA_NAME="math500,gsm8k,gaokao2023en,olympiadbench"
TOKENIZERS_PARALLELISM=false \
python3 -u main_online.py \
    --data_name ${DATA_NAME} \
    --data_dir "./external/qwen25_math_evaluation/data" \
    --draft_model_name_or_path ${DRAFT_MODEL} \
    --target_model_name_or_path ${TARGET_MODEL} \
    --prm_name_or_path ${PRM} \
    --draft_model_ip_address ${DRAFT_IP_ADDRESS} \
    --target_model_ip_address ${TARGET_IP_ADDRESS} \
    --prm_ip_address ${PRM_IP_ADDRESS} \
    --prm_threshold ${PRM_THRESHOLD} \
    --max_steps 100 \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 0 \
    --temperature 0 \
    --n_sampling 1 \
    --top_p 1 \
    --start 0 \
    --end -1 \
    --save_outputs \
    --overwrite \
    --max_tokens_per_call 4096 \

# multiple-choice datasets
DATA_NAME="gpqa,mmlu_stem"
TOKENIZERS_PARALLELISM=false \
python3 -u main_online.py \
    --data_name ${DATA_NAME} \
    --data_dir "./external/qwen25_math_evaluation/data" \
    --draft_model_name_or_path ${DRAFT_MODEL} \
    --target_model_name_or_path ${TARGET_MODEL} \
    --prm_name_or_path ${PRM} \
    --draft_model_ip_address ${DRAFT_IP_ADDRESS} \
    --target_model_ip_address ${TARGET_IP_ADDRESS} \
    --prm_ip_address ${PRM_IP_ADDRESS} \
    --prm_threshold ${PRM_THRESHOLD} \
    --max_steps 100 \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 0 \
    --temperature 0 \
    --n_sampling 1 \
    --top_p 1 \
    --start 0 \
    --end -1 \
    --save_outputs \
    --overwrite \
    --max_tokens_per_call 4096 \
    --num_shots 0 \

done