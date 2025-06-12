set -e

step=${step:-152}
model_step="/scratch/doqihu/work/verl_logs/20250606/agent_vlagent/debug_for_single_node/global_step_${step}"
hf_model_path="/scratch/doqihu/work/backbone/qwen25/Qwen2.5-VL-7B-Instruct"
trained_weights="${model_step}/actor/"
hgf_trained_weights="${trained_weights}/huggingface"

# if no model-*-safetensor in trained weights, convert to safetensor
if ! ls $hgf_trained_weights/model-*.safetensors 1> /dev/null 2>&1; then
    python scripts/model_merger.py \
        --hf_model_path $hf_model_path \
        --local_dir $trained_weights \
        --target_dir $hgf_trained_weights
fi


vllm serve $hgf_trained_weights \
    --port 18900 \
    --gpu-memory-utilization 0.8 \
    --tensor-parallel-size 1 \
    --served-model-name "trained_${step}steps" \
    --trust-remote-code \
    --disable-log-requests \
    --limit_mm_per_prompt "image=10"