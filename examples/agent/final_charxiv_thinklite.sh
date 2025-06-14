set -e
set -x
source env.sh

PROJECT_NAME="agent_charxiv"
EXPERIMENT_NAME="charxiv_for_single_node"

homedir=$(readlink -f "${home:-$HOME}")
date_str=$(date +%Y%m%d_%H%M%S)
export LLM_AS_A_JUDGE_BASE="http://10.0.127.192:18901/v1"

# if /checkpoints-fsx/doqihu exists, use it
if [ -d "/checkpoints-fsx/doqihu" ]; then
    checkpoint_dir="/checkpoints-fsx/doqihu/verl_logs"
    if [ ! -d "${homedir}/work/verl_logs" ]; then
        ln -s ${checkpoint_dir} ${homedir}/work/verl_logs
    fi
else
    checkpoint_dir="${homedir}/work/verl_logs/"
fi
export SAVE_CHECKPOINT_DIR="${checkpoint_dir}/${PROJECT_NAME}/${EXPERIMENT_NAME}/${date_str}"

# export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues
export REF_MODEL_PATH="${homedir}/work/backbone/qwen25/Qwen2.5-VL-7B-Instruct"
export WORLD_SIZE=1

BASEDIR="${homedir}/work/data"
VISUAL_DATASET_TRAIN_0_6_2=${BASEDIR}/data_v0.6.2_reason.parquet
VISUAL_DATASET_TRAIN_0_1_2=${BASEDIR}/data_0.1.2_visual_toolbox_v2.parquet
VISUAL_DATASET_TRAIN_0_8=${BASEDIR}/data_v0.8_visual_toolbox_v2.parquet
VISUAL_DATASET_TEST=${BASEDIR}/seekworld_test.parquet
EUREKA_DATASET_TRAIN=${BASEDIR}/data_thinklite_reasoning_acc.parquet

mkdir -p ${SAVE_CHECKPOINT_DIR}/logs

charxiv_train=${EVAL_DATA_PATH}/hgf/charxiv_converted/val.parquet
    # data.train_files=[${VISUAL_DATASET_TRAIN_0_1_2},${VISUAL_DATASET_TRAIN_0_8},${EUREKA_DATASET_TRAIN}] \
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    +debug=False \
    +vs_debug=False \
    data.train_files=[${charxiv_train}] \
    data.val_files=[${EUREKA_DATASET_TRAIN}] \
    data.train_batch_size=256 \
    data.max_prompt_length=8192 \
    data.max_response_length=20480 \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path=${REF_MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.checkpoint.contents=['model','hf_model','optimizer','extra'] \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.agent.activate_agent=True \
    actor_rollout_ref.rollout.agent.tool_name_key=env_name \
    actor_rollout_ref.rollout.agent.single_response_max_tokens=10240 \
    actor_rollout_ref.rollout.agent.max_turns=5 \
    actor_rollout_ref.rollout.agent.concurrent_workers=1 \
    actor_rollout_ref.rollout.agent.show_tqdm=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console','rl_logging_board','tensorboard','mlflow'] \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${WORLD_SIZE} \
    trainer.save_freq=8 \
    trainer.test_freq=10000 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    +trainer.tensorboard_dir=${SAVE_CHECKPOINT_DIR}/logs/tensorboard \
    +trainer.rl_logging_board_dir=${SAVE_CHECKPOINT_DIR}/logs/rl_logging_board \
    trainer.total_epochs=32 2>&1 | tee ${SAVE_CHECKPOINT_DIR}/logs/${EXPERIMENT_NAME}.log
