#!/bin/bash

hostname --ip-address
MODEL="Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"
MODEL_NAME="Skywork-o1-Open-PRM-Qwen-2.5-1.5B"

export CUDA_VISIBLE_DEVICES=2
python -m vllm.entrypoints.openai.api_server \
        --model $MODEL \
        --served-model-name $MODEL_NAME \
        --tensor-parallel-size 1 \
        --port 12342 \
        --host 0.0.0.0 \
        --trust-remote-code \
        --enforce-eager \
        --enable_prefix_caching \
        --gpu_memory_utilization 0.95