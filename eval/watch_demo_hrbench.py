# %%
import os
import json
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw
import numpy as np
import base64
import io
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# %%
def is_base64_image(image_data):
    """Check if the image data is base64 encoded"""
    if not isinstance(image_data, str):
        return False

    # Check if it's a data URI format
    if image_data.startswith('data:image/'):
        return True

    # Check if it looks like base64 (very long string with base64 characters)
    if len(image_data) > 1000 and re.match(r'^[A-Za-z0-9+/]+=*$', image_data.replace('\n', '').replace('\r', '')):
        return True

    return False

def decode_base64_image(image_data):
    """Decode base64 image data to PIL Image"""
    try:
        # Handle data URI format
        if image_data.startswith('data:image/'):
            # Extract the base64 part after the comma
            base64_data = image_data.split(',', 1)[1]
        else:
            base64_data = image_data

        # Decode base64
        image_bytes = base64.b64decode(base64_data)

        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        return image
    except Exception as e:
        print(f"Error decoding base64 image: {e}")
        # Return a dummy image as fallback
        return Image.new('RGB', (800, 600), color='white')

def format_print_str(long_str, line_length=175):
    formatted_str = '\n'.join([long_str[i:i+line_length] for i in range(0, len(long_str), line_length)])
    return formatted_str

def print_content(content):
    lines = content.splitlines()
    for line in lines:
        print(format_print_str(line))

def create_crop_visualization(ori_image, bbox_data, image_name, output_dir):
    """Create visualization images with bounding boxes drawn and extended crop areas"""
    if not bbox_data:
        return []

    visualization_paths = []

    try:
        for bbox_idx, bbox in enumerate(bbox_data):
            x1, y1, x2, y2 = bbox

            # Create a copy of the original image to draw on
            vis_image = ori_image.copy()
            draw = ImageDraw.Draw(vis_image)

            # Draw bounding box with width 5 (red color)
            box_width = 5
            for i in range(box_width):
                draw.rectangle([x1-i, y1-i, x2+i, y2+i], outline='red', fill=None)

            # Calculate extended crop area (256 pixels on each side)
            img_width, img_height = ori_image.size
            extend_pixels = 256

            crop_x1 = max(0, x1 - extend_pixels)
            crop_y1 = max(0, y1 - extend_pixels)
            crop_x2 = min(img_width, x2 + extend_pixels)
            crop_y2 = min(img_height, y2 + extend_pixels)

            # Crop the extended area from the visualization image
            crop_vis_image = vis_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

            # Save the crop visualization
            vis_filename = f"{image_name.replace('.jpg', '')}_crop_visualization_{bbox_idx}.jpg"
            vis_path = os.path.join(output_dir, 'images', 'cropped', vis_filename)
            Path(vis_path).parent.mkdir(parents=True, exist_ok=True)
            crop_vis_image.save(vis_path)

            visualization_paths.append(vis_path)

    except Exception as e:
        print(f"Error creating crop visualization for {image_name}: {e}")
        return []

    return visualization_paths

def extract_thinking_data(ori_image, conversation, image_name, output_dir):
    """Extract thinking process data and save cropped images"""
    turn_depth = 0
    thinking_text = []
    bbox_data = []
    cropped_paths = []

    for idx, conv in enumerate(conversation):
        conv_role = conv['role']
        content = conv['content']
        if conv_role == 'system':
            continue
        elif idx == 1:
            for _content in content:
                if _content['type'] == 'text':
                    _content = _content['text']
                    _content = _content.split('\n')[0]
                    thinking_text.append(f"USER: {_content}")
        elif conv_role == 'assistant':
            if isinstance(content, str):
                _content = content
            elif isinstance(content, list):
                for _content in content:
                    if _content['type'] == 'text':
                        _content = _content['text']
            else:
                continue

            thinking_text.append(f"ASSISTANT: {_content}")

            if "<tool_call>" in _content and "</answer>" not in _content:
                try:
                    _bbox_str = _content.split("<tool_call>")[1].split("</tool_call>")[0]
                    _bbox = eval(_bbox_str)['arguments']
                    for bbox_idx, _box in enumerate([_bbox]):
                        _box = _box['bbox_2d']
                        x1, y1, x2, y2 = _box
                        bbox_data.append(_box)

                        # Crop and save image
                        _crop_img = ori_image.crop((x1, y1, x2, y2))

                        # Save cropped image
                        crop_filename = f"{image_name.replace('.jpg', '')}_crop_{bbox_idx}.jpg"
                        crop_path = os.path.join(output_dir, 'cropped', crop_filename)
                        Path(crop_path).parent.mkdir(parents=True, exist_ok=True)
                        _crop_img.save(crop_path)
                        cropped_paths.append(crop_path)
                except Exception as e:
                    print(f"Error processing bbox for {image_name}: {e}")

            turn_depth += 1
            if "</answer>" in _content:
                break
        else:
            if isinstance(content, str):
                _content = content
            elif isinstance(content, list):
                for _content in content:
                    if _content['type'] == 'text':
                        _content = _content['text']
            else:
                continue
            thinking_text.append(f"USER: {_content}")

    return {
        'turn_depth': turn_depth,
        'thinking_conversation': '\n\n'.join(thinking_text),
        'bbox_coordinates': bbox_data,
        'cropped_images_paths': cropped_paths
    }

def format_dialogue_string(pred_output):
    """Format conversation into a readable string format"""
    dialogue_parts = []

    for turn in pred_output:
        role = turn['role']
        content = turn['content']
        if role == 'system':
            continue
        # Handle different content formats
        if isinstance(content, str):
            content_text = content
        elif isinstance(content, list):
            # Extract text from list format
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get('type') == 'text':
                    text_parts.append(item.get('text', ''))
                elif isinstance(item, str):
                    text_parts.append(item)
            content_text = ' '.join(text_parts)
        else:
            content_text = str(content)

        # Replace actual newlines with literal \n
        content_text = content_text.rstrip('\n')
        content_text = content_text.replace('\n', '\\n')

        # Format the turn with bold and red role
        role_html = f'<span style="color: red; font-weight: bold;">{role}</span>'
        # Escape angle brackets to prevent HTML tag rendering in content_text
        content_text_escaped = content_text.replace('<', '&lt;').replace('>', '&gt;')
        # context_text may have many \n in the end, strip all of them

        dialogue_parts.append(f'{role_html}: "{content_text_escaped}"')

    return '<br>'.join(dialogue_parts)

# %%
# Configuration - you can change these paths as needed
import os
DATASET_VERSION = os.environ.get('ver', '8k')  # Change to '4k' if you want to use the 4k version
output_base_dir = f'/home/ubuntu/work/laughing-potato/eval/output_data/hrbench/{DATASET_VERSION}'

# Multi-threading configuration
MAX_WORKERS = None  # Set to None for auto-detection, or specify a number (e.g., 16)
os.makedirs(os.path.join(output_base_dir, 'images', 'full'), exist_ok=True)
os.makedirs(os.path.join(output_base_dir, 'images', 'cropped'), exist_ok=True)
os.makedirs(os.path.join(output_base_dir, 'images', 'annotated'), exist_ok=True)
os.makedirs(os.path.join(output_base_dir, 'cropped'), exist_ok=True)

# Paths
dataset_path = f'/home/ubuntu/work/eval_data/hr_bench/hr_bench_{DATASET_VERSION}.tsv'
json_path = f'/home/ubuntu/work/laughing-potato/eval_results/hrbench/qwen/result_hr_bench_{DATASET_VERSION}_qwen.jsonl'

# %%
# Load dataset
print("Loading HR Bench dataset...")
dataset_df = pd.read_csv(dataset_path, sep='\t')
print(f"Dataset shape: {dataset_df.shape}")
print(f"Columns: {list(dataset_df.columns)}")

# Load evaluation results
print("Loading evaluation results...")
with open(json_path, 'r') as f:
    lines = f.readlines()
    results = [json.loads(line) for line in lines]

print(f"Number of result entries: {len(results)}")

# %%
# Create mapping between dataset and results
# Note: HR Bench results don't include image paths directly, so we need to match by index
dataset_by_index = {row['index']: row for _, row in dataset_df.iterrows()}

# Create optimized lookup by question for faster matching
dataset_by_question = {}
for _, row in dataset_df.iterrows():
    dataset_by_question[row['question']] = row

def process_case(result_idx, result_data, output_dir):
    """Process a single case and return data for dataframe"""

    # Extract data from result
    question = result_data['question']
    correct_answer = result_data['answer']  # A, B, C, or D
    answer_str = result_data['answer_str']  # The actual answer text
    pred_ans = result_data['pred_ans']
    pred_output = result_data['pred_output']
    category = result_data.get('category', 'unknown')
    status = result_data.get('status', 'unknown')

    # Find corresponding dataset entry by matching question (optimized lookup)
    dataset_entry = dataset_by_question.get(question)

    if dataset_entry is None:
        print(f"Warning: Could not find dataset entry for question: {question[:50]}...")
        # Create a dummy entry
        dataset_entry = {
            'index': result_idx,
            'image': f'unknown_{result_idx}.jpg',
            'A': 'N/A', 'B': 'N/A', 'C': 'N/A', 'D': 'N/A',
            'category': category
        }

    # Handle image data - check if it's base64 or filename
    raw_image_data = dataset_entry.get('image', None)

    # Generate clean image filename
    image_name = f'case_{result_idx}.jpg'

    # Load or create image
    if raw_image_data and is_base64_image(raw_image_data):
        # Decode base64 image
        actual_image = decode_base64_image(raw_image_data)
        # print(f"Decoded base64 image for case {result_idx}")
    elif raw_image_data and isinstance(raw_image_data, str) and len(raw_image_data) < 500:
        # It's likely a filename, but we'll create a placeholder since we don't have the actual file
        actual_image = Image.new('RGB', (800, 600), color='white')
        # Use the original filename if it's reasonable
        if raw_image_data.endswith(('.jpg', '.jpeg', '.png')):
            image_name = raw_image_data
        elif '.' in raw_image_data:
            image_name = raw_image_data.rsplit('.', 1)[0] + '.jpg'
    else:
        # Create dummy image
        actual_image = Image.new('RGB', (800, 600), color='white')

    # Check if prediction is correct
    # Extract predicted choice (A, B, C, D) from pred_ans
    pred_choice = 'unknown'
    if pred_ans:
        pred_ans_upper = pred_ans.upper()
        for choice in ['A', 'B', 'C', 'D']:
            if pred_ans_upper.startswith(choice + '.') or pred_ans_upper.startswith(choice + ' '):
                pred_choice = choice
                break
        if pred_choice == 'unknown':
            # Try to find choice at the start
            if pred_ans_upper.startswith('A'):
                pred_choice = 'A'
            elif pred_ans_upper.startswith('B'):
                pred_choice = 'B'
            elif pred_ans_upper.startswith('C'):
                pred_choice = 'C'
            elif pred_ans_upper.startswith('D'):
                pred_choice = 'D'

    is_correct = (pred_choice == correct_answer)

    # Save full image
    full_image_path = os.path.join(output_dir, 'images', 'full', image_name)
    actual_image.save(full_image_path)

    # Create annotated image (just a copy since no ground truth boxes)
    annotated_path = os.path.join(output_dir, 'images', 'annotated', image_name)
    actual_image.save(annotated_path)

    # Extract thinking data and save cropped images
    thinking_data = extract_thinking_data(actual_image, pred_output, image_name, output_dir)

    # Create crop visualizations with bounding boxes
    crop_visualization_paths = create_crop_visualization(
        actual_image, thinking_data['bbox_coordinates'], image_name, output_dir
    )

    # Prepare answer choices
    answer_choices = {
        'A': dataset_entry.get('A', 'N/A'),
        'B': dataset_entry.get('B', 'N/A'),
        'C': dataset_entry.get('C', 'N/A'),
        'D': dataset_entry.get('D', 'N/A')
    }

    return {
        'index': dataset_entry.get('index', result_idx),
        'image_filename': image_name,
        'question': question,
        'correct_answer_choice': correct_answer,
        'correct_answer_text': answer_str,
        'predicted_choice': pred_choice,
        'predicted_answer': pred_ans,
        'answer_choices': json.dumps(answer_choices),
        'full_pred_output': format_dialogue_string(pred_output),
        'is_correct': is_correct,
        'category': category,
        'status': status,
        'turn_depth': thinking_data['turn_depth'],
        'full_image_saved_path': full_image_path,
        'annotated_image_path': annotated_path,
        'cropped_images_paths': json.dumps(thinking_data['cropped_images_paths']),
        'bbox_coordinates': json.dumps(thinking_data['bbox_coordinates']),
        'crop_visualization_paths': json.dumps(crop_visualization_paths),
        'thinking_conversation': thinking_data['thinking_conversation']
    }

# Main processing loop with multi-threading
print("Processing all cases and creating dataframe...")
all_data = []

# Thread-safe progress tracking
progress_lock = threading.Lock()
processed_count = 0

def process_case_wrapper(args):
    """Wrapper function for thread-safe processing"""
    global processed_count
    result_idx, result_data = args

    try:
        case_data = process_case(result_idx, result_data, output_base_dir)

        # Thread-safe progress update
        with progress_lock:
            processed_count += 1
        return case_data
    except Exception as e:
        print(f"Error processing case {result_idx}: {e}")
        return None

# Prepare arguments for parallel processing
process_args = [(result_idx, result_data) for result_idx, result_data in enumerate(results)]

# Use ThreadPoolExecutor for parallel processing
# Adjust max_workers based on your system (typically 2-4x CPU cores for I/O bound tasks)
if MAX_WORKERS is None:
    max_workers = min(32, os.cpu_count() * 4)  # Cap at 32 to avoid too many threads
else:
    max_workers = MAX_WORKERS

print(f"Using {max_workers} threads for parallel processing...")
print(f"Total cases to process: {len(results)}")

start_time = time.time()
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    # Submit all tasks
    future_to_idx = {executor.submit(process_case_wrapper, args): args[0] for args in process_args}

    # Process completed tasks with progress bar
    if HAS_TQDM:
        with tqdm(total=len(results), desc="Processing cases") as pbar:
            for future in as_completed(future_to_idx):
                case_data = future.result()
                if case_data is not None:
                    all_data.append(case_data)
                pbar.update(1)
    else:
        completed = 0
        for future in as_completed(future_to_idx):
            case_data = future.result()
            if case_data is not None:
                all_data.append(case_data)
            completed += 1

# Sort results by index to maintain order
all_data.sort(key=lambda x: x['index'])

end_time = time.time()
processing_time = end_time - start_time
print(f"\nMulti-threaded processing completed in {processing_time:.2f} seconds")
print(f"Average time per case: {processing_time/len(results):.3f} seconds")

# Create DataFrame
df = pd.DataFrame(all_data)

# Calculate summary statistics
correct_count = df['is_correct'].sum()
total_count = len(df)
accuracy = correct_count / total_count

print(f"\nProcessing complete!")
print(f"Total cases: {total_count}")
print(f"Correct predictions: {correct_count}")
print(f"Accuracy: {accuracy:.3f}")

# Category-wise analysis
if 'category' in df.columns:
    category_stats = df.groupby('category').agg({
        'is_correct': ['count', 'sum', 'mean']
    }).round(3)
    category_stats.columns = ['Total', 'Correct', 'Accuracy']
    print(f"\nCategory-wise performance:")
    print(category_stats)

# Save DataFrame
csv_path = os.path.join(output_base_dir, 'analysis_results.csv')
df.to_csv(csv_path, index=False)
print(f"DataFrame saved to: {csv_path}")

# Save summary statistics
summary_stats = {
    'dataset_version': DATASET_VERSION,
    'total_cases': total_count,
    'correct_predictions': correct_count,
    'accuracy': accuracy,
    'avg_turn_depth': df['turn_depth'].mean(),
    'categories': df['category'].value_counts().to_dict() if 'category' in df.columns else {}
}

# Create markdown report
def create_markdown_report(df, summary_stats, output_dir):
    """Create a comprehensive markdown report"""
    md_content = []

    # Title and summary
    md_content.append("# HR Bench Evaluation Analysis Report\n")
    md_content.append(f"**Generated on:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    md_content.append(f"**Dataset Version:** HR Bench {summary_stats['dataset_version'].upper()}\n")
    md_content.append("---\n")

    # Summary Statistics
    md_content.append("## Summary Statistics\n")
    md_content.append("| Metric | Value |")
    md_content.append("|--------|-------|")
    md_content.append(f"| Dataset Version | HR Bench {summary_stats['dataset_version'].upper()} |")
    md_content.append(f"| Total Cases | {summary_stats['total_cases']} |")
    md_content.append(f"| Correct Predictions | {summary_stats['correct_predictions']} |")
    md_content.append(f"| Accuracy | {summary_stats['accuracy']:.3f} ({summary_stats['accuracy']*100:.1f}%) |")
    md_content.append(f"| Average Turn Depth | {summary_stats['avg_turn_depth']:.2f} |")
    md_content.append("")

    # Dataset Info
    md_content.append("## Dataset Information\n")
    md_content.append(f"- **Total Questions:** {len(df)}")
    md_content.append(f"- **Data Source:** `{json_path}`")
    md_content.append(f"- **Dataset File:** `{dataset_path}`")
    md_content.append(f"- **Output Directory:** `{output_dir}`")
    md_content.append("")

    # Performance Analysis
    md_content.append("## Performance Analysis\n")

    # Category Analysis
    if 'category' in df.columns and len(df['category'].unique()) > 1:
        md_content.append("### Performance by Category\n")
        category_stats = df.groupby('category').agg({
            'is_correct': ['count', 'sum', 'mean']
        }).round(3)
        category_stats.columns = ['Total', 'Correct', 'Accuracy']

        md_content.append("| Category | Total | Correct | Accuracy |")
        md_content.append("|----------|-------|---------|----------|")
        for category, row in category_stats.iterrows():
            md_content.append(f"| {category} | {row['Total']} | {row['Correct']} | {row['Accuracy']:.3f} ({row['Accuracy']*100:.1f}%) |")
        md_content.append("")

    # Turn Depth Analysis
    md_content.append("### Turn Depth Analysis\n")
    turn_depth_analysis = df.groupby('turn_depth').agg({
        'is_correct': ['count', 'sum', 'mean']
    }).round(3)
    turn_depth_analysis.columns = ['Count', 'Correct', 'Accuracy']
    turn_depth_analysis['Percentage'] = (turn_depth_analysis['Count'] / len(df) * 100).round(1)

    md_content.append("| Turn Depth | Count | Percentage | Correct | Accuracy |")
    md_content.append("|------------|-------|------------|---------|----------|")
    for depth, row in turn_depth_analysis.iterrows():
        md_content.append(f"| {depth} | {row['Count']} | {row['Percentage']:.1f}% | {row['Correct']} | {row['Accuracy']:.3f} ({row['Accuracy']*100:.1f}%) |")
    md_content.append("")

    # Answer Choice Distribution
    if 'predicted_choice' in df.columns:
        md_content.append("### Predicted Answer Distribution\n")
        choice_analysis = df.groupby('predicted_choice').agg({
            'is_correct': ['count', 'sum', 'mean']
        }).round(3)
        choice_analysis.columns = ['Count', 'Correct', 'Accuracy']
        choice_analysis['Percentage'] = (choice_analysis['Count'] / len(df) * 100).round(1)

        md_content.append("| Predicted Choice | Count | Percentage | Correct | Accuracy |")
        md_content.append("|------------------|-------|------------|---------|----------|")
        for choice, row in choice_analysis.iterrows():
            md_content.append(f"| {choice} | {row['Count']} | {row['Percentage']:.1f}% | {row['Correct']} | {row['Accuracy']:.3f} ({row['Accuracy']*100:.1f}%) |")
        md_content.append("")

    # Sample Cases
    md_content.append("## Sample Cases\n")

    # Helper functions (same as vstar version but adapted)
    def format_image_path(path, max_width=150, max_height=150, use_base64=False):
        """Convert image path to HTML img tag with preserved aspect ratio"""
        if pd.isna(path) or path == '':
            return ''
        try:
            rel_path = os.path.relpath(path, output_dir)
            return f'<img src="{rel_path}" style="max-width: {max_width}px; max-height: {max_height}px; height: auto; width: auto; object-fit: contain; border: 1px solid #ddd; border-radius: 4px;">'
        except:
            return f'<span style="color: red;">Path error</span>'

    def format_cropped_images(cropped_paths_json, max_width=100, max_height=100):
        """Convert cropped image paths JSON to HTML img tags"""
        if pd.isna(cropped_paths_json) or cropped_paths_json == '':
            return ''
        try:
            paths = json.loads(cropped_paths_json)
            img_tags = []
            for i, path in enumerate(paths[:3]):  # Show max 3 cropped images
                rel_path = os.path.relpath(path, output_dir)
                img_tag = f'<img src="{rel_path}" style="max-width: {max_width}px; max-height: {max_height}px; height: auto; width: auto; object-fit: contain; border: 1px solid #ddd; margin: 2px; border-radius: 3px;" title="Crop {i+1}">'
                img_tags.append(img_tag)
            return f'<div style="display: flex; flex-wrap: wrap; gap: 2px; align-items: center;">{" ".join(img_tags)}</div>'
        except Exception as e:
            return '<span style="color: red;">Cropped images error</span>'

    def format_correctness(is_correct):
        """Format correctness as emoji"""
        return '✅' if is_correct else '❌'

    def format_answer_choices(choices_json):
        """Format answer choices"""
        if pd.isna(choices_json):
            return ''
        try:
            choices = json.loads(choices_json)
            formatted = []
            for choice, text in choices.items():
                formatted.append(f"{choice}. {text}")
            return '<br>'.join(formatted)
        except:
            return 'Error parsing choices'

    # Table configuration
    display_columns = ['index', 'question', 'answer_choices', 'correct_answer_choice',
                      'predicted_choice', 'full_pred_output', 'is_correct', 'turn_depth', 'category',
                      'full_image_saved_path', 'cropped_images_paths', 'crop_visualization_paths']

    column_names = {
        'index': 'Index',
        'question': 'Question',
        'answer_choices': 'Choices',
        'correct_answer_choice': 'Correct',
        'predicted_choice': 'Predicted',
        'full_pred_output': 'Full Dialogue',
        'is_correct': 'Result',
        'turn_depth': 'Turns',
        'category': 'Category',
        'full_image_saved_path': 'Image',
        'cropped_images_paths': 'Cropped Regions',
        'crop_visualization_paths': 'Crop in Image'
    }

    formatters = {
        'Image': lambda x: format_image_path(x, 150, 150),
        'Cropped Regions': lambda x: format_cropped_images(x, 100, 100),
        'Crop in Image': lambda x: format_cropped_images(x, 150, 150),
        'Result': format_correctness,
        'Choices': format_answer_choices,
        'Question': lambda x: x if not pd.isna(x) else '',
        'Full Dialogue': lambda x: x if not pd.isna(x) else '',
        'Category': lambda x: x if not pd.isna(x) else ''
    }

    def generate_styled_table(display_df, table_id, formatters):
        html_table = display_df.to_html(
            formatters=formatters,
            escape=False,
            index=False,
            classes='table table-striped table-bordered',
            table_id=table_id
        )

        styled_table = f"""
<style>
#{table_id} {{
    width: 100%;
    border-collapse: collapse;
    margin: 0 0 20px 0;
    font-family: Arial, sans-serif;
}}
#{table_id} th, #{table_id} td {{
    border: 1px solid #ddd;
    padding: 12px;
    text-align: left;
    vertical-align: top;
}}
#{table_id} th {{
    background-color: #f8f9fa;
    font-weight: bold;
    font-size: 14px;
    color: #333;
}}
#{table_id} tr:nth-child(even) {{
    background-color: #f9f9f9;
}}
#{table_id} tr:hover {{
    background-color: #f0f8ff;
}}
#{table_id} td {{
    max-width: 200px;
    word-wrap: break-word;
}}
</style>

{html_table}
"""
        return styled_table

        # Sample Cases - Group 1: Incorrect Cases (Cross and Single)
    wrong_cases = df[df['is_correct'] == False]
    if len(wrong_cases) > 0:
        md_content.append("### 1. Incorrect Cases (Cross and Single)\n")

        # Cross incorrect cases
        cross_wrong = wrong_cases[wrong_cases['category'].str.contains('cross', case=False, na=False)]
        sample_cross_wrong = cross_wrong.sample(min(5, len(cross_wrong)), random_state=42) if len(cross_wrong) > 0 else pd.DataFrame()

        # Single incorrect cases
        single_wrong = wrong_cases[wrong_cases['category'].str.contains('single', case=False, na=False)]
        sample_single_wrong = single_wrong.sample(min(5, len(single_wrong)), random_state=43) if len(single_wrong) > 0 else pd.DataFrame()

        # Combine and display
        combined_wrong = pd.concat([sample_cross_wrong, sample_single_wrong], ignore_index=True)
        if len(combined_wrong) > 0:
            md_content.append(f"**Cross Category:** {len(sample_cross_wrong)} cases, **Single Category:** {len(sample_single_wrong)} cases\n")
            wrong_display = combined_wrong[display_columns].copy().rename(columns=column_names)
            styled_table = generate_styled_table(wrong_display, 'wrong-cases-table', formatters)
            md_content.append(styled_table)
        else:
            md_content.append("*No incorrect cases found in cross/single categories.*\n")
        md_content.append("")

    # Sample Cases - Group 2: Correct Cases (Cross and Single)
    correct_cases = df[df['is_correct'] == True]
    if len(correct_cases) > 0:
        md_content.append("### 2. Correct Cases (Cross and Single)\n")

        # Cross correct cases
        cross_correct = correct_cases[correct_cases['category'].str.contains('cross', case=False, na=False)]
        sample_cross_correct = cross_correct.sample(min(2, len(cross_correct)), random_state=44) if len(cross_correct) > 0 else pd.DataFrame()

        # Single correct cases
        single_correct = correct_cases[correct_cases['category'].str.contains('single', case=False, na=False)]
        sample_single_correct = single_correct.sample(min(2, len(single_correct)), random_state=45) if len(single_correct) > 0 else pd.DataFrame()

        # Combine and display
        combined_correct = pd.concat([sample_cross_correct, sample_single_correct], ignore_index=True)
        if len(combined_correct) > 0:
            md_content.append(f"**Cross Category:** {len(sample_cross_correct)} cases, **Single Category:** {len(sample_single_correct)} cases\n")
            correct_display = combined_correct[display_columns].copy().rename(columns=column_names)
            styled_table = generate_styled_table(correct_display, 'correct-cases-table', formatters)
            md_content.append(styled_table)
        else:
            md_content.append("*No correct cases found in cross/single categories.*\n")
        md_content.append("")

    # Sample Cases - Group 3: Non-2-Turn Cases
    non_2_turn_cases = df[df['turn_depth'] != 2]
    if len(non_2_turn_cases) > 0:
        md_content.append("### 3. Non-2-Turn Cases (Turn Depth ≠ 2)\n")

        sample_non_2_turn = non_2_turn_cases.sample(min(8, len(non_2_turn_cases)), random_state=46)
        md_content.append(f"**Total cases with turn depth ≠ 2:** {len(non_2_turn_cases)} (showing {len(sample_non_2_turn)} samples)\n")

        non_2_turn_display = sample_non_2_turn[display_columns].copy().rename(columns=column_names)
        styled_table = generate_styled_table(non_2_turn_display, 'non-2-turn-cases-table', formatters)
        md_content.append(styled_table)
        md_content.append("")
    # DataFrame Schema
    md_content.append("## DataFrame Schema\n")
    md_content.append("| Column | Data Type | Description |")
    md_content.append("|--------|-----------|-------------|")
    md_content.append("| index | integer | Dataset index |")
    md_content.append("| image_filename | string | Image filename |")
    md_content.append("| question | string | Evaluation question |")
    md_content.append("| correct_answer_choice | string | Correct choice (A/B/C/D) |")
    md_content.append("| correct_answer_text | string | Correct answer text |")
    md_content.append("| predicted_choice | string | Predicted choice (A/B/C/D) |")
    md_content.append("| predicted_answer | string | Full predicted answer |")
    md_content.append("| answer_choices | JSON string | All answer choices |")
    md_content.append("| full_pred_output | string | Full conversation (formatted as dialogue) |")
    md_content.append("| is_correct | boolean | Whether prediction is correct |")
    md_content.append("| category | string | Question category |")
    md_content.append("| status | string | Processing status |")
    md_content.append("| turn_depth | integer | Number of conversation turns |")
    md_content.append("| full_image_saved_path | string | Path to saved image |")
    md_content.append("| annotated_image_path | string | Path to annotated image |")
    md_content.append("| cropped_images_paths | JSON string | List of cropped image paths |")
    md_content.append("| bbox_coordinates | JSON string | Predicted bounding box coordinates |")
    md_content.append("| crop_visualization_paths | JSON string | List of paths to crop visualizations with bounding boxes and 256px context |")
    md_content.append("| thinking_conversation | string | Full conversation text |")
    md_content.append("")

    return "\n".join(md_content)

# Generate markdown report
md_report = create_markdown_report(df, summary_stats, output_base_dir)
md_path = os.path.join(output_base_dir, 'analysis_report.md')
with open(md_path, 'w', encoding='utf-8') as f:
    f.write(md_report)

print(f"Markdown report saved to: {md_path}")
print(f"Images saved to: {os.path.join(output_base_dir, 'images')}")

# %%
# Display summary information
print(f"\nDataFrame shape: {df.shape}")
print(f"Columns: {list(df.columns)}")