home=${home:-$HOME}
export JUDGE_MODEL_PATH="${home}/work/backbone/qwen25/Qwen2.5-72B-Instruct"

mkdir -p $JUDGE_MODEL_PATH

# download model
huggingface-cli download --resume-download Qwen/Qwen2.5-72B-Instruct --local-dir $JUDGE_MODEL_PATH --local-dir-use-symlinks False

# run it
vllm serve $JUDGE_MODEL_PATH \
    --port 18901 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 32768 \
    --tensor-parallel-size 8 \
    --served-model-name "judge" \
    --trust-remote-code \
    --disable-log-requests