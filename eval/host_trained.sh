trained_weights="/scratch/doqihu/work/verl_logs/20250606/agent_vlagent/debug_for_single_node/global_step_80/actor/huggingface"

vllm serve $trained_weights \
    --port 18900 \
    --gpu-memory-utilization 0.8 \
    --tensor-parallel-size 1 \
    --served-model-name "trained_80steps" \
    --trust-remote-code \
    --disable-log-requests \
    --limit_mm_per_prompt "image=10"