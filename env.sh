home=${home:-$HOME}

# From download_eval.sh
export EVAL_DATA_PATH="${home}/work/eval_data"

# From download_judge.sh
export JUDGE_MODEL_PATH="${home}/work/backbone/qwen25/Qwen2.5-72B-Instruct"

# From download_trained_weights.sh
export trained_weights="${home}/work/trained_weights"

# From download_model_data.sh
export SAVE_CHECKPOINT_DIR="${home}/work/verl_checkpoints"
export REF_MODEL_PATH="${home}/work/backbone/qwen25/Qwen2.5-VL-7B-Instruct"
export DATA_PATH="${home}/work/data"
