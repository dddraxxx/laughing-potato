home=${home:-$HOME}
export trained_weights="${home}/work/trained_weights"

mkdir -p $trained_weights

# download checkpoints
huggingface-cli download --resume-download ChenShawn/DeepEyes-7B --local-dir $trained_weights --local-dir-use-symlinks False

vllm serve $trained_weights \
    --port 18900 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 32768 \
    --tensor-parallel-size 1 \
    --served-model-name "de7b" \
    --trust-remote-code \
    --disable-log-requests