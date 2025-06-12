from datasets import load_dataset
import io
from PIL import Image
import os

hgf_charxiv_dir='/scratch/doqihu/work/eval_data/hgf/charxiv'

# convert keys to images, data_source, reward_model, extra_info, prompt, ability, env_name
# we have: [image], [reasoning_q], [reasoning_a],
# images: list of [image]
# [image]: pil image, ==> images: [image]
# [reasoning_q]: single question ==> prompt: template.format(reasoning_q)
# [reasoning_a]: single answer ==> reward_model: {ground_truth: reasoning_a}
# [reasoning_a_type]: int ==> ability: "charxiv_{}"

# converting...
# data_source: "charxiv"
# extra_info:{
#     question: reasoning_q,
#     answer: reasoning_a,
#     figure_path: figure_path,
# }

data_source = 'charxiv'
env_name = 'visual_toolbox_v2'

system_prompt =\
{
"content": "You are a helpful assistant.\n\n# Tools\nYou may call one or more functions to assist with the user query.\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{\"type\":\"function\",\"function\":{\"name\":\"image_zoom_in_tool\",\"description\":\"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an optional object label.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"bbox_2d\":{\"type\":\"array\",\"items\":{\"type\":\"number\"},\"minItems\":4,\"maxItems\":4,\"description\":\"The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.\"},\"label\":{\"type\":\"string\",\"description\":\"The name or label of the object in the specified bounding box (optional).\"}},\"required\":[\"bbox\"]}}}\n</tools>\n\n# How to call a tool\nReturn a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>\n\n**Example**: \n<tool_call> \n{\"name\": \"image_zoom_in_tool\", \"arguments\": {\"bbox_2d\": [10, 20, 100, 200], \"label\": \"the apple on the desk\"}} \n</tool_call>",
"role": "system"
}
user_prompt_template = \
{
"content": "<image>\n{question}\nThink first, call **image_zoom_in_tool** if needed, then answer. Format strictly as: <think>...</think> <tool_call>...</tool_call> (if tools needed) <answer>...</answer> ",
"role": "user"
}

from pathlib import Path
output_path = Path(hgf_charxiv_dir).parent / 'charxiv_converted'
output_path.mkdir(parents=True, exist_ok=True)


def load_charxiv_data():
    """Load CharXiv validation dataset using HuggingFace datasets"""
    dataset = load_dataset('parquet', data_files={
        'validation': os.path.join(hgf_charxiv_dir, 'val.parquet'),
        'test': os.path.join(hgf_charxiv_dir, 'test.parquet')
    })
    return dataset['validation']

def convert_image(image_data):
    """Convert image data to PIL Image - handles both dict format and direct PIL"""
    if isinstance(image_data, dict) and 'bytes' in image_data:
        # Original format with bytes
        image_bytes = image_data['bytes']
        image = Image.open(io.BytesIO(image_bytes))
    else:
        # Already a PIL image (HuggingFace datasets auto-conversion)
        image = image_data
    return image

def format_prompt(question):
    """Format question using the prompt template"""
    user_prompt = {
        "content": user_prompt_template["content"].format(question=question),
        "role": "user"
    }
    return [system_prompt, user_prompt]

def convert_sample(example):
    """Convert a single sample to target format - for use with ds.map()"""
    # Convert image
    image = convert_image(example['image'])

    # Create converted sample
    converted = {
        'images': [image],  # List of PIL images
        'data_source': data_source,
        'reward_model': {'ground_truth': example['reasoning_a']},
        'extra_info': {
            'question': example['reasoning_q'],
            'answer': example['reasoning_a'],
            'figure_path': example['figure_path']
        },
        'prompt': format_prompt(example['reasoning_q']),
        'ability': f"charxiv_{example['reasoning_a_type']}",
        'env_name': env_name
    }

    return converted

def convert_dataset():
    """Convert entire CharXiv validation dataset using HuggingFace datasets"""
    print("Loading CharXiv validation dataset...")
    dataset = load_charxiv_data()
    print(f"Loaded {len(dataset)} samples")

    print("Converting samples using ds.map()...")
    converted_dataset = dataset.map(
        convert_sample,
        remove_columns=dataset.column_names,  # Remove original columns
        desc="Converting CharXiv samples",
        num_proc=16,
    )

    print(f"Successfully converted {len(converted_dataset)} samples")


    return converted_dataset

def test_loading(parquet_file):
    dataframe = load_dataset("parquet", data_files=parquet_file)["train"]
    print(f"Loaded {len(dataframe)} samples")
    print(f"Features: {list(dataframe.features.keys())}")
    print(f"Sample keys: {list(dataframe[0].keys())}")

if __name__ == "__main__":
    converted_dataset = convert_dataset()
    # Save converted dataset
    output_parquet_path = os.path.join(output_path, 'val.parquet')
    converted_dataset.to_parquet(output_parquet_path)
    print(f"Saved converted dataset to: {output_path}")
    print(f"Conversion complete! Dataset length: {len(converted_dataset)}")
    print(f"Features: {list(converted_dataset.features.keys())}")
    print(f"Sample keys: {list(converted_dataset[0].keys())}")
    # save a head to jsonl, drop images
    converted_dataset.take(5).remove_columns('images').to_json(os.path.join(output_path, 'head.jsonl'))
    test_loading(output_parquet_path)