home=${home:-$HOME}
export trained_weights="${home}/work/trained_weights"

mkdir -p $trained_weights

# download checkpoints
huggingface-cli download --resume-download ChenShawn/DeepEyes-7B --local-dir $trained_weights --local-dir-use-symlinks False

vllm serve $trained_weights \
    --port 18900 \
    --gpu-memory-utilization 0.8 \
    --tensor-parallel-size 2 \
    --served-model-name "de7b" \
    --trust-remote-code \
    --disable-log-requests \
    --limit_mm_per_prompt "image=10" \
    --max-model-len 50240 \