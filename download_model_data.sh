home=${home:-$HOME}
export SAVE_CHECKPOINT_DIR="${home}/work/verl_checkpoints"
export REF_MODEL_PATH="${home}/work/backbone/qwen25/Qwen2.5-VL-7B-Instruct"
export DATA_PATH="${home}/work/data"

mkdir -p $SAVE_CHECKPOINT_DIR
mkdir -p $REF_MODEL_PATH

# download model
huggingface-cli download --resume-download Qwen/Qwen2.5-VL-7B-Instruct --local-dir $REF_MODEL_PATH --local-dir-use-symlinks False

# download data
huggingface-cli download --resume-download ChenShawn/DeepEyes-Datasets-47k --local-dir $DATA_PATH --local-dir-use-symlinks False --repo-type dataset

# download checkpoints
