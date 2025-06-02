home=${home:-$HOME}
export EVAL_DATA_PATH="${home}/work/eval_data"

mkdir -p $EVAL_DATA_PATH

# download data
huggingface-cli download --resume-download craigwu/vstar_bench --local-dir $EVAL_DATA_PATH/vstar_bench --local-dir-use-symlinks False --repo-type dataset