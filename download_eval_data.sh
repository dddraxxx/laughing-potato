source env.sh
if [ -d "/data/doqihu" ]; then
    export EVAL_DATA_PATH="/data/doqihu/work/eval_data"
    mkdir -p $EVAL_DATA_PATH
    ln -s $EVAL_DATA_PATH "${home}/work/eval_data"
else
    export EVAL_DATA_PATH="${home}/work/eval_data"
fi

mkdir -p $EVAL_DATA_PATH

# download data
# vstar
huggingface-cli download --resume-download craigwu/vstar_bench --local-dir $EVAL_DATA_PATH/vstar_bench --local-dir-use-symlinks False --repo-type dataset

sleep 0.1

# hrbench
huggingface-cli download --resume-download DreamMr/HR-Bench --local-dir $EVAL_DATA_PATH/hr_bench --local-dir-use-symlinks False --repo-type dataset

sleep 0.1

# hgf/mme
huggingface-cli download --resume-download lmms-lab/MME --local-dir $EVAL_DATA_PATH/hgf/mme --local-dir-use-symlinks False --repo-type dataset

sleep 0.1

# hgf/charxiv
huggingface-cli download --resume-download princeton-nlp/CharXiv --local-dir $EVAL_DATA_PATH/hgf/charxiv --local-dir-use-symlinks False --repo-type dataset
unzip $EVAL_DATA_PATH/hgf/charxiv/images.zip -d $EVAL_DATA_PATH/hgf/charxiv/images