home=${home:-$HOME}
export EVAL_DATA_PATH="${home}/work/eval_data"

mkdir -p $EVAL_DATA_PATH

# download data
# vstar
huggingface-cli download --resume-download craigwu/vstar_bench --local-dir $EVAL_DATA_PATH/vstar_bench --local-dir-use-symlinks False --repo-type dataset

# hrbench
huggingface-cli download --resume-download DreamMr/HR-Bench --local-dir $EVAL_DATA_PATH/hr_bench --local-dir-use-symlinks False --repo-type dataset