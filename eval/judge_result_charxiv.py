import os
import json
import numpy as np
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import argparse
import torch
from tqdm import tqdm
from io import BytesIO
from PIL import Image
import base64
import io
from openai import OpenAI
import requests

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_version', type=str, default='validation', help='Dataset version')
parser.add_argument('--test_model_name', type=str, default='trained_152steps', help='Model name for result save')
parser.add_argument('--model_name', type=str, default='qwen', help='Model name for result save')
parser.add_argument('--api_key', type=str, default='EMPTY', help='API key')
parser.add_argument('--api_url', type=str, default='http://10.0.127.192:18901/v1', help='API URL')
parser.add_argument('--save_path', type=str, default='/scratch/doqihu/laughing-potato/eval_results/charxiv', help='Path to save the results')
parser.add_argument('--eval_model_name', type=str, default=None, help='Model name for evaluation')
parser.add_argument('--num_workers', type=int, default=8)
args = parser.parse_args()

openai_api_key = args.api_key
openai_api_base = args.api_url
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
if args.eval_model_name is None:
    response = requests.get(f"{openai_api_base}/models")
    models = response.json()
    eval_model_name = models['data'][0]['id']
else:
    eval_model_name = args.eval_model_name

all_acc = []

def get_chat_template():
    chat_template = """
Below are two answers to a question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question. Determine whether these two answers are consistent.
Note that [Model Answer] is consistent with [Standard Answer] whenever they are essentially the same. If the meaning is expressed in the same way, it is considered consistent. The answers don't need to be word-for-word identical, but should convey the same meaning or information.
If they are consistent, Judgement is 1; if they are different, Judgement is 0. Just output Judgement and don't output anything else.\n\n
"""
    return chat_template

def get_gpt4_score_ICE():
    example_1 = """
[Question]: What is the main character doing in the image?
[Standard Answer]: The main character is reading a book while sitting on a bench.
[Model_answer]: reading a book on a bench
Judgement: 1
""" # noqa

    example_2 = """
[Question]: What color is the car in the image?
[Standard Answer]: The car is red.
[Model_answer]: red
Judgement: 1
""" # noqa

    example_3 = """
[Question]: How many people are visible in the scene?
[Standard Answer]: There are three people visible in the scene.
[Model_answer]: 3
Judgement: 1
""" # noqa

    example_4 = """
[Question]: What is written on the sign?
[Standard Answer]: The sign says "No Parking".
[Model_answer]: No Parking
Judgement: 1
""" # noqa

    example_5 = """
[Question]: What is the weather like in the image?
[Standard Answer]: It is sunny and clear.
[Model_answer]: It's raining heavily.
Judgement: 0
""" # noqa

    example_6 = """
[Question]: What animal is shown in the picture?
[Standard Answer]: A cat is shown in the picture.
[Model_answer]: There is a dog in the image.
Judgement: 0
""" # noqa

    example_7 = """
[Question]: What time of day does this appear to be?
[Standard Answer]: This appears to be during the afternoon.
[Model_answer]: This looks like it was taken at night.
Judgement: 0
""" # noqa

    return [example_1, example_2, example_3, example_4, example_5, example_6, example_7]

def get_prompt(predict_str, ground_truth, question):
    examples = get_gpt4_score_ICE()
    chat_template = get_chat_template()
    demo_prompt = chat_template
    for example in examples:
        demo_prompt += example + '\n\n'
    test_prompt = f"""
[Question]: {question}
[Standard Answer]: {ground_truth}
[Model_answer]: {predict_str}
Judgement:"""
    full_prompt = f'{demo_prompt}{test_prompt}'

    return full_prompt

result_root_path = args.save_path
result_root_path = os.path.join(result_root_path, args.test_model_name)
all_acc = []
error_nums = 0

def process(line):
    line = line.strip()
    data = json.loads(line)
    question = data['question']
    answer = data['answer']
    pred_ans = data['pred_ans']
    pred_output = data['pred_output']

    if '\\boxed' in pred_ans:
        pred_ans = pred_ans.split('\\boxed{')[1].split('}')[0]

    # Direct text comparison first
    acc_reward = 0.0
    if pred_ans.strip().lower() == answer.strip().lower():
        acc_reward = 1.0
    elif answer.lower() in pred_ans.lower() or pred_ans.lower() in answer.lower():
        acc_reward = 1.0
    else:
        # Use GPT-4 judge for more nuanced comparison
        full_prompt = get_prompt(pred_ans, answer, question)

        try:
            chat_response = client.chat.completions.create(
                model=eval_model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": full_prompt},
                ],
                temperature=0.0,
            )
            response = chat_response.choices[0].message.content.strip()

            if 'ERROR' in pred_ans:
                global error_nums
                error_nums += 1

            if 'Judgement:' in response:
                response = response.split('Judgement:')[-1].strip()
                if '1' in response:
                    acc_reward = 1.0
                elif '0' in response:
                    acc_reward = 0.0
                else:
                    print(f' [WARNING] resp format error {response=}')
                    acc_reward = 0.0
            else:
                if response == '1':
                    acc_reward = 1.0
                elif response == '0':
                    acc_reward = 0.0
                else:
                    print(f' [WARNING] resp format error {response=}')
                    acc_reward = 0.0
        except Exception as e:
            print(f"Error in API call: {e}")
            acc_reward = 0.0

    for p_out in pred_output:
        if p_out['role'] == 'system' or p_out['role'] == 'user':
            continue
        p_content = p_out['content']
        if type(p_content) == str:
            p_content_msg = p_content.strip()
        elif type(p_content) == list:
            for _p_content in p_content:
                if _p_content['type'] == 'text':
                    p_content_msg = _p_content['text']

    return acc_reward, data

if __name__ == '__main__':
    error_preds = []

    # Process single charxiv result file
    save_name = f"result_{args.dataset_version}_{args.model_name}.jsonl"
    result_path = os.path.join(result_root_path, save_name)

    if not os.path.exists(result_path):
        print(f"Result file not found: {result_path}")
        print("Available files in directory:")
        if os.path.exists(result_root_path):
            for file in os.listdir(result_root_path):
                if file.endswith('.jsonl'):
                    print(f"  {file}")
        exit(1)

    save_json = []
    pool = multiprocessing.Pool(processes=args.num_workers)

    with open(result_path, 'r') as f:
        lines = f.readlines()

    with tqdm(total=len(lines), desc="Judging Charxiv") as pbar:
        for result in pool.imap(process, lines):
            if result is not None:
                acc_reward, data = result
                acc = acc_reward
                all_acc.append(acc)

                if acc_reward != 1.0:
                    error_preds.append({
                        'pred_ans': data['pred_ans'],
                        'question': data['question'],
                        'answer': data['answer']
                    })

                data['acc'] = acc
                save_json.append(data)
                pbar.update(1)

    pool.close()
    pool.join()

    # Save results with accuracy scores
    with open(os.path.join(result_root_path, save_name.replace('.jsonl', '_acc.jsonl')), 'w') as f:
        for item in save_json:
            f.write(json.dumps(item) + '\n')

    # Calculate and save final results
    final_acc = {}
    overall_accuracy = np.mean(all_acc) * 100
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")
    final_acc['overall'] = overall_accuracy
    final_acc['error_nums'] = error_nums
    final_acc['error_preds'] = error_preds
    final_acc['total_samples'] = len(all_acc)

    with open(os.path.join(result_root_path, f'final_{args.dataset_version}_{args.model_name}_acc.json'), 'w') as f:
        json.dump(final_acc, f, indent=4)

    print(f"Results saved to: {result_root_path}")
    print(f"Total samples: {len(all_acc)}")
    print(f"Error samples: {error_nums}")