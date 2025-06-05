# %%
import os
import json
import glob
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
            vis_filename = f"{image_name.replace('.jpg', '').replace('.png', '')}_crop_visualization_{bbox_idx}.jpg"
            vis_path = os.path.join(output_dir, 'images', 'cropped', vis_filename)
            Path(vis_path).parent.mkdir(parents=True, exist_ok=True)
            crop_vis_image.save(vis_path)

            visualization_paths.append(vis_path)

    except Exception as e:
        print(f"Error creating crop visualization for {image_name}: {e}")
        return []

    return visualization_paths

def extract_text_content(content):
    """Extract text content from various content formats"""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get('type') == 'text':
                text_parts.append(item.get('text', ''))
            elif isinstance(item, str):
                text_parts.append(item)
        return ' '.join(text_parts)
    else:
        return str(content)

def extract_thinking_data(ori_image, conversation, image_name, output_dir):
    """Extract thinking process data and save cropped images"""
    turn_depth = 0
    thinking_turns = []
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
                    # Extract just the question part
                    question_part = _content.split('\n')[0] if '\n' in _content else _content
                    thinking_turns.append({"role": "user", "content": question_part})
        elif conv_role == 'assistant':
            _content = extract_text_content(content)
            if not _content:
                continue

            turn_depth += 1
            user_last = False
            if "<answer>" in _content and "</answer>" in _content:
                _content = _content[:_content.rfind("</answer>")]
                thinking_turns.append({"role": "assistant", "content": _content})
                break
            # Look for tool calls with image_zoom_in_tool
            if "<tool_call>" in _content and "image_zoom_in_tool" in _content:
                try:
                    _bbox_str = _content.split("<tool_call>")[1].split("</tool_call>")[0]
                    _bbox = json.loads(_bbox_str)['arguments']

                    if 'bbox_2d' in _bbox:
                        _box = _bbox['bbox_2d']
                        x1, y1, x2, y2 = _box
                        bbox_data.append(_box)

                        # Crop and save image
                        _crop_img = ori_image.crop((x1, y1, x2, y2))

                        # Save cropped image
                        idx = len(cropped_paths)
                        crop_filename = f"{image_name.replace('.jpg', '').replace('.png', '')}_crop_{idx}.jpg"
                        crop_path = os.path.join(output_dir, 'cropped', crop_filename)
                        Path(crop_path).parent.mkdir(parents=True, exist_ok=True)
                        _crop_img.save(crop_path)
                        cropped_paths.append(crop_path)
                except Exception as e:
                    print(f"Error processing bbox for {image_name}: {e}")

            thinking_turns.append({"role": "assistant", "content": _content})
        else:
            _content = extract_text_content(content)
            if not _content:
                continue
            thinking_turns.append({"role": "user", "content": _content})
            user_last = True
    if user_last:
        print("Wrong")
    return {
        'turn_depth': turn_depth,
        'thinking_conversation': thinking_turns,
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

        # Use the shared content extraction function
        content_text = extract_text_content(content)

        # Replace actual newlines with literal \n
        content_text = content_text.rstrip('\n')
        content_text = content_text.replace('\n', '\\n')

        # Format the turn with bold and red role
        role_html = f'<span style="color: red; font-weight: bold;">{role}</span>'
        # Escape angle brackets to prevent HTML tag rendering in content_text
        content_text_escaped = content_text.replace('<', '&lt;').replace('>', '&gt;')

        dialogue_parts.append(f'{role_html}: "{content_text_escaped}"')

    return '<br>'.join(dialogue_parts)

def extract_original_image_from_conversation(conversation):
    """Extract the original base64 image from the first user message"""
    for turn in conversation:
        if turn['role'] == 'user' and isinstance(turn.get('content'), list):
            for content_item in turn['content']:
                if (content_item.get('type') == 'image_url' and
                    'image_url' in content_item and
                    'url' in content_item['image_url']):
                    return content_item['image_url']['url']
    return None

# %%
# Configuration - you can change these paths as needed
output_base_dir = '/home/ubuntu/work/laughing-potato/eval/output_data/mme'

# Multi-threading configuration
MAX_WORKERS = None  # Set to None for auto-detection, or specify a number (e.g., 16)
os.makedirs(os.path.join(output_base_dir, 'images', 'full'), exist_ok=True)
os.makedirs(os.path.join(output_base_dir, 'images', 'cropped'), exist_ok=True)
os.makedirs(os.path.join(output_base_dir, 'images', 'annotated'), exist_ok=True)
os.makedirs(os.path.join(output_base_dir, 'cropped'), exist_ok=True)

# Paths
json_path = '/home/ubuntu/work/laughing-potato/eval_results/hgf/mme/result_test_qwen.jsonl'
mme_data_path = '/home/ubuntu/work/eval_data/hgf/mme/data'

# %%
# Load evaluation results
print("Loading MME evaluation results...")
with open(json_path, 'r') as f:
    lines = f.readlines()
    results = [json.loads(line) for line in lines]

print(f"Number of result entries: {len(results)}")

# Load MME dataset images
print("Loading MME dataset images...")
import glob
parquet_files = glob.glob(os.path.join(mme_data_path, '*.parquet'))
mme_data = []
for pf in parquet_files:
    df = pd.read_parquet(pf)
    mme_data.append(df)

mme_df = pd.concat(mme_data, ignore_index=True)
print(f"Loaded {len(mme_df)} images from MME dataset")

# Create image lookup dictionary
image_lookup = {}
for _, row in mme_df.iterrows():
    image_lookup[row['question_id']] = row['image']

# %%
def process_case(result_idx, result_data, output_dir):
    """Process a single MME case and return data for dataframe"""

    # Extract data from result
    question_id = result_data['question_id']
    question = result_data['question']
    correct_answer = result_data['answer']
    pred_ans = result_data['pred_ans']
    pred_output = result_data['pred_output']
    category = result_data.get('category', 'unknown')
    status = result_data.get('status', 'unknown')

    # Generate clean image filename from question_id
    image_name = f"{question_id.replace('/', '_').replace('.png', '').replace('.jpg', '')}_case_{result_idx}.jpg"

    # Load image from MME dataset
    if question_id in image_lookup:
        try:
            image_data = image_lookup[question_id]
            if 'bytes' in image_data:
                # Image is stored as bytes in the dataset
                image_bytes = image_data['bytes']
                actual_image = Image.open(io.BytesIO(image_bytes))
                # Convert to RGB if necessary
                if actual_image.mode != 'RGB':
                    actual_image = actual_image.convert('RGB')
            else:
                # Fallback to dummy image
                actual_image = Image.new('RGB', (800, 600), color='white')
        except Exception as e:
            print(f"Error loading image for {question_id}: {e}")
            actual_image = Image.new('RGB', (800, 600), color='white')
    else:
        print(f"No image found for question_id: {question_id}")
        actual_image = Image.new('RGB', (800, 600), color='white')

    # Check if prediction is correct
    # For MME, we need to handle various answer formats
    is_correct = False
    if pred_ans and correct_answer:
        # Clean both answers for comparison
        pred_clean = pred_ans.strip().lower()
        correct_clean = correct_answer.strip().lower()

        # Direct match
        if pred_clean == correct_clean:
            is_correct = True
        # Handle Yes/No cases
        elif correct_clean in ['yes', 'no']:
            if (correct_clean == 'yes' and any(x in pred_clean for x in ['yes', 'a. yes'])) or \
               (correct_clean == 'no' and any(x in pred_clean for x in ['no', 'b. no', 'a. no'])):
                is_correct = True
        # Handle other exact matches after cleaning
        elif pred_clean.startswith(correct_clean) or correct_clean in pred_clean:
            is_correct = True

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

    return {
        'question_id': question_id,
        'image_filename': image_name,
        'question': question,
        'correct_answer': correct_answer,
        'predicted_answer': pred_ans,
        'is_correct': is_correct,
        'category': category,
        'status': status,
        'turn_depth': thinking_data['turn_depth'],
        'full_image_saved_path': full_image_path,
        'annotated_image_path': annotated_path,
        'cropped_images_paths': json.dumps(thinking_data['cropped_images_paths']),
        'bbox_coordinates': json.dumps(thinking_data['bbox_coordinates']),
        'crop_visualization_paths': json.dumps(crop_visualization_paths),
        'thinking_conversation': format_dialogue_string(thinking_data['thinking_conversation']),
        'full_pred_output': format_dialogue_string(thinking_data['thinking_conversation']),
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

# Sort results by question_id to maintain order
all_data.sort(key=lambda x: x['question_id'])

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
    md_content.append("# MME Evaluation Analysis Report\n")
    md_content.append(f"**Generated on:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    md_content.append("**Dataset:** MME (Multi-Modal Evaluation)\n")
    md_content.append("---\n")

    # Summary Statistics
    md_content.append("## Summary Statistics\n")
    md_content.append("| Metric | Value |")
    md_content.append("|--------|-------|")
    md_content.append(f"| Total Cases | {summary_stats['total_cases']} |")
    md_content.append(f"| Correct Predictions | {summary_stats['correct_predictions']} |")
    md_content.append(f"| Accuracy | {summary_stats['accuracy']:.3f} ({summary_stats['accuracy']*100:.1f}%) |")
    md_content.append(f"| Average Turn Depth | {summary_stats['avg_turn_depth']:.2f} |")
    md_content.append("")

    # Dataset Info
    md_content.append("## Dataset Information\n")
    md_content.append(f"- **Total Questions:** {len(df)}")
    md_content.append(f"- **Data Source:** `{json_path}`")
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

    # Sample Cases
    md_content.append("## Sample Cases\n")

    # Helper functions
    def format_image_path(path, max_width=150, max_height=150):
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

    # Table configuration
    display_columns = ['question_id', 'question', 'correct_answer', 'predicted_answer',
                      'full_pred_output', 'is_correct', 'turn_depth', 'category',
                      'full_image_saved_path', 'cropped_images_paths', 'crop_visualization_paths']

    column_names = {
        'question_id': 'Question ID',
        'question': 'Question',
        'correct_answer': 'Correct Answer',
        'predicted_answer': 'Predicted Answer',
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

    # Sample Cases - Group 1: Incorrect Cases by Category
    wrong_cases = df[df['is_correct'] == False]
    if len(wrong_cases) > 0:
        md_content.append("### 1. Incorrect Cases by Category\n")

        # Sample from each category with incorrect predictions
        sample_wrong_cases = []
        for category in df['category'].unique():
            cat_wrong = wrong_cases[wrong_cases['category'] == category]
            if len(cat_wrong) > 0:
                sample_wrong_cases.append(cat_wrong.sample(min(2, len(cat_wrong)), random_state=42))

        if sample_wrong_cases:
            combined_wrong = pd.concat(sample_wrong_cases, ignore_index=True)
            md_content.append(f"**Showing samples from {len(sample_wrong_cases)} categories with incorrect predictions**\n")
            wrong_display = combined_wrong[display_columns].copy().rename(columns=column_names)
            styled_table = generate_styled_table(wrong_display, 'wrong-cases-table', formatters)
            md_content.append(styled_table)
        else:
            md_content.append("*No incorrect cases found.*\n")
        md_content.append("")

    # Sample Cases - Group 2: Correct Cases by Category
    correct_cases = df[df['is_correct'] == True]
    if len(correct_cases) > 0:
        md_content.append("### 2. Correct Cases by Category\n")

        # Sample from each category with correct predictions
        sample_correct_cases = []
        for category in df['category'].unique():
            cat_correct = correct_cases[correct_cases['category'] == category]
            if len(cat_correct) > 0:
                sample_correct_cases.append(cat_correct.sample(min(2, len(cat_correct)), random_state=44))

        if sample_correct_cases:
            combined_correct = pd.concat(sample_correct_cases, ignore_index=True)
            md_content.append(f"**Showing samples from {len(sample_correct_cases)} categories with correct predictions**\n")
            correct_display = combined_correct[display_columns].copy().rename(columns=column_names)
            styled_table = generate_styled_table(correct_display, 'correct-cases-table', formatters)
            md_content.append(styled_table)
        else:
            md_content.append("*No correct cases found.*\n")
        md_content.append("")

    # Sample Cases - Group 3: Multi-Turn Cases
    multi_turn_cases = df[df['turn_depth'] > 2]
    if len(multi_turn_cases) > 0:
        md_content.append("### 3. Multi-Turn Cases (Turn Depth > 2)\n")

        sample_multi_turn = multi_turn_cases.sample(min(3, len(multi_turn_cases)), random_state=46)
        md_content.append(f"**Total cases with turn depth > 2:** {len(multi_turn_cases)} (showing {len(sample_multi_turn)} samples)\n")

        multi_turn_display = sample_multi_turn[display_columns].copy().rename(columns=column_names)
        styled_table = generate_styled_table(multi_turn_display, 'multi-turn-cases-table', formatters)
        md_content.append(styled_table)
        md_content.append("")

    # DataFrame Schema
    md_content.append("## DataFrame Schema\n")
    md_content.append("| Column | Data Type | Description |")
    md_content.append("|--------|-----------|-------------|")
    md_content.append("| question_id | string | Original question identifier |")
    md_content.append("| image_filename | string | Saved image filename |")
    md_content.append("| question | string | Evaluation question |")
    md_content.append("| correct_answer | string | Correct answer |")
    md_content.append("| predicted_answer | string | Model's predicted answer |")
    md_content.append("| full_pred_output | string | Full conversation (formatted as dialogue) |")
    md_content.append("| is_correct | boolean | Whether prediction is correct |")
    md_content.append("| category | string | MME evaluation category |")
    md_content.append("| status | string | Processing status |")
    md_content.append("| turn_depth | integer | Number of conversation turns |")
    md_content.append("| full_image_saved_path | string | Path to saved image |")
    md_content.append("| annotated_image_path | string | Path to annotated image |")
    md_content.append("| cropped_images_paths | JSON string | List of cropped image paths |")
    md_content.append("| bbox_coordinates | JSON string | Predicted bounding box coordinates |")
    md_content.append("| crop_visualization_paths | JSON string | List of paths to crop visualizations with bounding boxes |")
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