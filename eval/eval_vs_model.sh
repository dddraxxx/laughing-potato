# assume we have model running on localhost:18900 by vllm

# Get model name from vllm server
model_name=$(curl -s localhost:18900/v1/models | jq -r '.data[0].id')

echo "Using model: $model_name"

# for vstar,
python eval/eval_vstar.py
python eval/watch_demo_vstar.py --modelwname $model_name --dataset_version direct_attributes
python eval/watch_demo_vstar.py --model_name $model_name --dataset_version relative_position

# for eval
python eval/eval_hrbench.py
python eval/watch_demo_hrbench.py --model_name $model_name --dataset_version 4k
python eval/watch_demo_hrbench.py --model_name $model_name --dataset_version 8k