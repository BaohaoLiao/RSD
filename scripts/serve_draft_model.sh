#!/bin/bash

hostname --ip-address
MODEL="Qwen/Qwen2.5-Math-1.5B-Instruct"
MODEL_NAME="Qwen2.5-Math-1.5B-Instruct"

export CUDA_VISIBLE_DEVICES=0
python -m vllm.entrypoints.openai.api_server \
        --model $MODEL \
        --served-model-name $MODEL_NAME \
        --tensor-parallel-size 1 \
        --port 12340 \
        --host 0.0.0.0 \
        --trust-remote-code \
        --max-model-len 8192 \
        --enforce-eager \
        --enable_prefix_caching \
        --gpu_memory_utilization 0.95