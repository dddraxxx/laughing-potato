#!/bin/bash

if [ "$dp" = "1" ]; then
    dbg="debugpy --listen 5678 --wait-for-client"
else
    dbg="python"
fi
$dbg scripts/model_merger.py \
    --backend fsdp \
    --hf_model_path /scratch/doqihu/work/backbone/qwen25/Qwen2.5-VL-7B-Instruct \
    --local_dir /scratch/doqihu/work/verl_logs/20250606/agent_vlagent/debug_for_single_node/global_step_80/actor \