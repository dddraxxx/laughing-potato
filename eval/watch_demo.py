# %%
import os
import json
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw
import numpy as np

# %%
def format_print_str(long_str, line_length=175):
    formatted_str = '\n'.join([long_str[i:i+line_length] for i in range(0, len(long_str), line_length)])
    return formatted_str

def calculate_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)

    if x2_inter > x1_inter and y2_inter > y1_inter:
        area_inter = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    else:
        area_inter = 0

    area_box1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area_box2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    area_union = area_box1 + area_box2 - area_inter

    if area_union == 0:
        return 0
    iou = area_inter / area_union
    return iou

def print_content(content):
    lines = content.splitlines()
    for line in lines:
        print(format_print_str(line))


def extract_thinking_data(ori_image, conversation, ori_gt_boxes, image_name, output_dir):
    """Extract thinking process data and save cropped images"""
    turn_depth = 0
    thinking_text = []
    bbox_data = []
    cropped_paths = []
    iou_scores = []

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
                _bbox_str = _content.split("<tool_call>")[1].split("</tool_call>")[0]
                _bbox = eval(_bbox_str)['arguments']
                for bbox_idx, _box in enumerate([_bbox]):
                    _box = _box['bbox_2d']
                    x1, y1, x2, y2 = _box
                    bbox_data.append(_box)

                    # Crop and save image
                    _crop_img = ori_image.crop((x1, y1, x2, y2))

                    # Calculate max IoU
                    max_iou = 0.
                    for _gt_box in ori_gt_boxes:
                        _gt_x1, gt_y1, gt_w, gt_h = _gt_box
                        _gt_x2, gt_y2 = _gt_x1 + gt_w, gt_y1 + gt_h
                        _gt_box = (_gt_x1, gt_y1, _gt_x2, gt_y2)
                        iou = calculate_iou(_box, _gt_box)
                        if iou > max_iou:
                            max_iou = iou

                    iou_scores.append(max_iou)

                    # Save cropped image
                    crop_filename = f"{image_name.replace('.jpg', '')}_crop_{bbox_idx}_iou{max_iou:.3f}.jpg"
                    crop_path = os.path.join(output_dir, 'cropped', crop_filename)
                    Path(crop_path).parent.mkdir(parents=True, exist_ok=True)
                    _crop_img.save(crop_path)
                    cropped_paths.append(crop_path)

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
        'cropped_images_paths': cropped_paths,
        'iou_scores': iou_scores,
        'max_iou_score': max(iou_scores) if iou_scores else 0.0
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
        content_text = content_text.replace('\n', '\\n')

        # Format the turn with bold and red role
        role_html = f'<span style="color: red; font-weight: bold;">{role}</span>'
        # Escape angle brackets to prevent HTML tag rendering in content_text
        content_text_escaped = content_text.replace('<', '&lt;').replace('>', '&gt;')
        dialogue_parts.append(f'{role_html}: "{content_text_escaped}"')

    return '<br>'.join(dialogue_parts)

# %%
# Create output directories
output_base_dir = '/home/ubuntu/work/laughing-potato/eval/output_data'
os.makedirs(os.path.join(output_base_dir, 'images', 'full'), exist_ok=True)
os.makedirs(os.path.join(output_base_dir, 'images', 'cropped'), exist_ok=True)
os.makedirs(os.path.join(output_base_dir, 'images', 'annotated'), exist_ok=True)

root_path = '/home/ubuntu/work/eval_data/vstar_bench'
json_path = '/home/ubuntu/work/laughing-potato/eval_results/vstar/qwen/result_direct_attributes_qwen.jsonl'

if 'direct_attributes' in json_path:
    root_path = os.path.join(root_path, 'direct_attributes')
else:
    root_path = os.path.join(root_path, 'relative_position')


# %%
with open(json_path, 'r') as f:
    lines = f.readlines()
    lines = [json.loads(line) for line in lines]
line_map = {}
image_list = []
for line in lines:
    line_map[line['image']] = line
    image_list.append(line['image'])

# %%

def process_case(line_id, output_dir):
    """Process a single case and return data for dataframe"""
    tosee_img = image_list[line_id]
    img_path = os.path.join(root_path, tosee_img)
    question = line_map[tosee_img]['question']
    answer = line_map[tosee_img]['answer']
    pred_ans = line_map[tosee_img]['pred_ans']
    pred_output = line_map[tosee_img]['pred_output']
    correct = answer.lower() in pred_ans.lower()

    # Load image and ground truth data
    ori_image_path = os.path.join(root_path, tosee_img)
    ori_json_path = os.path.join(root_path, tosee_img.replace('.jpg', '.json'))
    ori_json = json.load(open(ori_json_path, 'r'))
    ori_gt_name = ori_json['target_object']
    ori_gt_boxes = ori_json['bbox']

    ori_image = Image.open(ori_image_path)

    # Save full image
    full_image_path = os.path.join(output_dir, 'images', 'full', tosee_img)
    ori_image.save(full_image_path)

    # Create annotated image with bounding boxes
    annotated_image = ori_image.copy()
    draw = ImageDraw.Draw(annotated_image)
    # Draw ground truth boxes in green
    min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')
    offset = 20
    for idx, _box in enumerate(ori_gt_boxes):
        x1, y1, w, h = _box
        x2, y2 = x1 + w, y1 + h
        draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
        min_x = min(min_x, x1)
        min_y = min(min_y, y1)
        max_x = max(max_x, x2)
        max_y = max(max_y, y2)
    min_x, min_y, max_x, max_y = max(0, min_x - offset), max(0, min_y - offset), min(ori_image.width, max_x + offset), min(ori_image.height, max_y + offset)
    annotated_image = annotated_image.crop((min_x, min_y, max_x, max_y))

    # Save annotated image
    annotated_path = os.path.join(output_dir, 'images', 'annotated', tosee_img)
    annotated_image.save(annotated_path)

    # Extract thinking data and save cropped images
    thinking_data = extract_thinking_data(ori_image, pred_output, ori_gt_boxes, tosee_img, output_dir)

    # Prepare ground truth bbox data
    gt_bbox_formatted = []
    for _box in ori_gt_boxes:
        x1, y1, w, h = _box
        gt_bbox_formatted.append([x1, y1, x1 + w, y1 + h])

    return {
        'image_filename': tosee_img,
        'question': question,
        'ground_truth_answer': answer,
        'predicted_answer': pred_ans,
        'full_pred_output': format_dialogue_string(pred_output),
        'is_correct': correct,
        'turn_depth': thinking_data['turn_depth'],
        'full_image_saved_path': full_image_path,
        'annotated_image_path': annotated_path,
        'cropped_images_paths': json.dumps(thinking_data['cropped_images_paths']),
        'bbox_coordinates': json.dumps(thinking_data['bbox_coordinates']),
        'gt_bbox_coordinates': json.dumps(gt_bbox_formatted),
        'max_iou_score': thinking_data['max_iou_score'],
        'all_iou_scores': json.dumps(thinking_data['iou_scores']),
        'thinking_conversation': thinking_data['thinking_conversation'],
        'gt_object_names': json.dumps(ori_gt_name)
    }

# Main processing loop
print("Processing all cases and creating dataframe...")
all_data = []

for line_id in range(len(image_list)):
    if line_id % 10 == 0:
        print(f"Processing {line_id}/{len(image_list)}")

    case_data = process_case(line_id, output_base_dir)
    all_data.append(case_data)

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
    'avg_max_iou': df['max_iou_score'].mean(),
    'high_iou_cases': (df['max_iou_score'] > 0.5).sum()
}

# Create markdown report
def create_markdown_report(df, summary_stats, output_dir):
    """Create a comprehensive markdown report"""
    md_content = []

    # Title and summary
    md_content.append("# Visual Reasoning Evaluation Analysis Report\n")
    md_content.append(f"**Generated on:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    md_content.append("---\n")

    # Summary Statistics
    md_content.append("## Summary Statistics\n")
    md_content.append("| Metric | Value |")
    md_content.append("|--------|-------|")
    md_content.append(f"| Total Cases | {summary_stats['total_cases']} |")
    md_content.append(f"| Correct Predictions | {summary_stats['correct_predictions']} |")
    md_content.append(f"| Accuracy | {summary_stats['accuracy']:.3f} ({summary_stats['accuracy']*100:.1f}%) |")
    md_content.append(f"| Average Turn Depth | {summary_stats['avg_turn_depth']:.2f} |")
    md_content.append(f"| Average Max IoU | {summary_stats['avg_max_iou']:.3f} |")
    md_content.append(f"| High IoU Cases (>0.5) | {summary_stats['high_iou_cases']} |")
    md_content.append("")

    # Dataset Info
    md_content.append("## Dataset Information\n")
    md_content.append(f"- **Total Images:** {len(df)}")
    md_content.append(f"- **Data Source:** `{json_path}`")
    md_content.append(f"- **Image Source:** `{root_path}`")
    md_content.append(f"- **Output Directory:** `{output_dir}`")
    md_content.append("")

    # Performance Analysis
    md_content.append("## Performance Analysis\n")

    # IoU Distribution
    iou_ranges = [
        (-1.0, 0.0, "Zero"),
        (0.0, 0.1, "Very Low"),
        (0.1, 0.3, "Low"),
        (0.3, 0.5, "Medium"),
        (0.5, 0.7, "High"),
        (0.7, 1.0, "Very High")
    ]

    md_content.append("### IoU Score Distribution\n")
    md_content.append("| IoU Range | Description | Count | Percentage |")
    md_content.append("|-----------|-------------|-------|------------|")

    for min_iou, max_iou, desc in iou_ranges:
        count = ((df['max_iou_score'] > min_iou) & (df['max_iou_score'] <= max_iou)).sum()
        if min_iou == 0.7:  # Include 1.0 in the last range
            count = (df['max_iou_score'] > min_iou).sum()
        percentage = (count / len(df)) * 100
        md_content.append(f"| {min_iou:.1f} - {max_iou:.1f} | {desc} | {count} | {percentage:.1f}% |")

    md_content.append("")

    # Turn Depth Analysis
    md_content.append("### Turn Depth Analysis\n")
    turn_depth_stats = df['turn_depth'].value_counts().sort_index()
    md_content.append("| Turn Depth | Count | Percentage |")
    md_content.append("|------------|-------|------------|")
    for depth, count in turn_depth_stats.items():
        percentage = (count / len(df)) * 100
        md_content.append(f"| {depth} | {count} | {percentage:.1f}% |")
    md_content.append("")

    # Accuracy by IoU ranges
    md_content.append("### Accuracy by IoU Score Range\n")
    md_content.append("| IoU Range | Correct | Total | Accuracy |")
    md_content.append("|-----------|---------|-------|----------|")

    for min_iou, max_iou, desc in iou_ranges:
        if min_iou == 0.7:
            mask = df['max_iou_score'] > min_iou
        else:
            mask = (df['max_iou_score'] > min_iou) & (df['max_iou_score'] <= max_iou)

        subset = df[mask]
        if len(subset) > 0:
            correct = subset['is_correct'].sum()
            total = len(subset)
            acc = correct / total
            md_content.append(f"| {min_iou:.1f} - {max_iou:.1f} | {correct} | {total} | {acc:.3f} ({acc*100:.1f}%) |")

    md_content.append("")

    # Sample Cases
    md_content.append("## Sample Cases\n")

    # Helper function to create image HTML
    def format_image_path(path, max_width=150, max_height=150, use_base64=True):
        """Convert image path to HTML img tag with preserved aspect ratio"""
        if pd.isna(path) or path == '':
            return ''

        if use_base64:
            # Embed image as base64 for self-contained markdown
            try:
                import base64
                with open(path, 'rb') as img_file:
                    img_data = base64.b64encode(img_file.read()).decode()
                    img_ext = path.split('.')[-1].lower()
                    mime_type = f'image/{img_ext}' if img_ext in ['png', 'jpg', 'jpeg', 'gif'] else 'image/jpeg'
                    return f'<img src="data:{mime_type};base64,{img_data}" style="max-width: {max_width}px; max-height: {max_height}px; height: auto; width: auto; object-fit: contain; border: 1px solid #ddd; border-radius: 4px;">'
            except Exception as e:
                print(f"Error encoding image {path}: {e}")
                return f'<span style="color: red;">Image load error</span>'
        else:
            # Use relative path (requires images to be accessible relative to markdown file)
            try:
                rel_path = os.path.relpath(path, output_dir)
                return f'<img src="{rel_path}" style="max-width: {max_width}px; max-height: {max_height}px; height: auto; width: auto; object-fit: contain; border: 1px solid #ddd; border-radius: 4px;">'
            except:
                return f'<span style="color: red;">Path error</span>'

    def format_cropped_images(cropped_paths_json, max_width=100, max_height=100, use_base64=True):
        """Convert cropped image paths JSON to HTML img tags with preserved aspect ratio"""
        if pd.isna(cropped_paths_json) or cropped_paths_json == '':
            return ''
        try:
            paths = json.loads(cropped_paths_json)
            img_tags = []
            for i, path in enumerate(paths[:3]):  # Show max 3 cropped images
                if use_base64:
                    try:
                        import base64
                        with open(path, 'rb') as img_file:
                            img_data = base64.b64encode(img_file.read()).decode()
                            img_ext = path.split('.')[-1].lower()
                            mime_type = f'image/{img_ext}' if img_ext in ['png', 'jpg', 'jpeg', 'gif'] else 'image/jpeg'
                            img_tag = f'<img src="data:{mime_type};base64,{img_data}" style="max-width: {max_width}px; max-height: {max_height}px; height: auto; width: auto; object-fit: contain; border: 1px solid #ddd; margin: 2px; border-radius: 3px;" title="Crop {i+1}">'
                            img_tags.append(img_tag)
                    except Exception as e:
                        print(f"Error encoding cropped image {path}: {e}")
                        img_tags.append(f'<span style="color: red;">Error</span>')
                else:
                    rel_path = os.path.relpath(path, output_dir)
                    img_tag = f'<img src="{rel_path}" style="max-width: {max_width}px; max-height: {max_height}px; height: auto; width: auto; object-fit: contain; border: 1px solid #ddd; margin: 2px; border-radius: 3px;" title="Crop {i+1}">'
                    img_tags.append(img_tag)

            # Return images in a flex container for better layout
            return f'<div style="display: flex; flex-wrap: wrap; gap: 2px; align-items: center;">{" ".join(img_tags)}</div>'
        except Exception as e:
            print(f"Error processing cropped images: {e}")
            return '<span style="color: red;">Cropped images error</span>'

    def format_correctness(is_correct):
        """Format correctness as emoji"""
        return '✅' if is_correct else '❌'

    def format_iou_score(score):
        """Format IoU score with color coding"""
        if pd.isna(score):
            return ''
        color = '#28a745' if score > 0.7 else '#ffc107' if score > 0.3 else '#dc3545'
        return f'<span style="color: {color}; font-weight: bold;">{score:.3f}</span>'

    # Shared function to create table formatters
    def get_table_formatters():
        return {
            'Full Image': lambda x: format_image_path(x, 150, 150, use_base64=False),
            'Annotated': lambda x: format_image_path(x, 150, 150, use_base64=False),
            'Cropped Regions': lambda x: format_cropped_images(x, 100, 100, use_base64=False),
            'Correct': format_correctness,
            'IoU Score': format_iou_score,
            'Question': lambda x: x if not pd.isna(x) else '',
            'Ground Truth': lambda x: x if not pd.isna(x) else '',
            'Prediction': lambda x: x if not pd.isna(x) else '',
            'Full Dialogue': lambda x: x if not pd.isna(x) else ''
        }

    # Shared function to generate styled table
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
/* Image column styling - columns 9, 10, 11 (Full Image, Annotated, Cropped Regions) */
#{table_id} td:nth-child(9),
#{table_id} td:nth-child(10),
#{table_id} td:nth-child(11) {{
    text-align: center;
    vertical-align: top;
    min-width: 120px;
    max-width: 180px;
    padding: 8px;
}}
</style>

{html_table}
"""
        return styled_table

    # Column configuration
    display_columns = ['image_filename', 'question', 'ground_truth_answer', 'predicted_answer',
                      'full_pred_output', 'is_correct', 'max_iou_score', 'turn_depth', 'full_image_saved_path',
                      'annotated_image_path', 'cropped_images_paths']

    column_names = {
        'image_filename': 'Image',
        'question': 'Question',
        'ground_truth_answer': 'Ground Truth',
        'predicted_answer': 'Prediction',
        'full_pred_output': 'Full Dialogue',
        'is_correct': 'Correct',
        'max_iou_score': 'IoU Score',
        'turn_depth': 'Turns',
        'full_image_saved_path': 'Full Image',
        'annotated_image_path': 'Annotated',
        'cropped_images_paths': 'Cropped Regions'
    }

    formatters = get_table_formatters()

    # Best cases (high IoU and correct)
    best_cases = df[(df['is_correct'] == True) & (df['max_iou_score'] > 0.7)]
    if len(best_cases) > 0:
        print(f"Best cases: {len(best_cases)}")
        best_cases = best_cases.sample(5, random_state=42)
        md_content.append("### Best Performing Cases (Correct + High IoU)\n")

        best_display = best_cases[display_columns].copy().rename(columns=column_names)
        styled_table = generate_styled_table(best_display, 'best-cases-table', formatters)
        md_content.append(styled_table)
        md_content.append("")

    # Wrong cases (incorrect)
    wrong_cases = df[(df['is_correct'] == False)]
    if len(wrong_cases) > 0:
        print(f"Wrong cases: {len(wrong_cases)}")
        # wrong_cases = wrong_cases.sample(5, random_state=42)
        md_content.append("### Wrong Cases (Incorrect)\n")

        wrong_display = wrong_cases[display_columns].copy().rename(columns=column_names)
        styled_table = generate_styled_table(wrong_display, 'wrong-cases-table', formatters)
        md_content.append(styled_table)
        md_content.append("")

    # Challenging cases (low IoU)
    challenging_cases = df[(df['max_iou_score'] < 0.2)]
    if len(challenging_cases) > 0:
        challenging_cases = challenging_cases.sample(5, random_state=42)
        print(f"Challenging cases: {len(challenging_cases)}")
        md_content.append("### Challenging Cases (Low IoU)\n")

        challenging_display = challenging_cases[display_columns].copy().rename(columns=column_names)
        styled_table = generate_styled_table(challenging_display, 'challenging-cases-table', formatters)
        md_content.append(styled_table)
        md_content.append("")

    # DataFrame Schema
    md_content.append("## DataFrame Schema\n")
    md_content.append("| Column | Data Type | Description |")
    md_content.append("|--------|-----------|-------------|")
    md_content.append("| image_filename | string | Original image filename |")
    md_content.append("| question | string | Evaluation question |")
    md_content.append("| ground_truth_answer | string | Correct answer |")
    md_content.append("| predicted_answer | string | Model's prediction |")
    md_content.append("| full_pred_output | string | Full predicted conversation (formatted as dialogue) |")
    md_content.append("| is_correct | boolean | Whether prediction is correct |")
    md_content.append("| turn_depth | integer | Number of conversation turns |")
    md_content.append("| full_image_saved_path | string | Path to saved full image |")
    md_content.append("| annotated_image_path | string | Path to annotated image |")
    md_content.append("| cropped_images_paths | JSON string | List of cropped image paths |")
    md_content.append("| bbox_coordinates | JSON string | Predicted bounding box coordinates |")
    md_content.append("| gt_bbox_coordinates | JSON string | Ground truth bounding box coordinates |")
    md_content.append("| max_iou_score | float | Highest IoU score achieved |")
    md_content.append("| all_iou_scores | JSON string | All IoU scores |")
    md_content.append("| thinking_conversation | string | Full conversation text |")
    md_content.append("| gt_object_names | JSON string | Ground truth object names |")
    md_content.append("")

    # File Structure
    md_content.append("## Generated Files Structure\n")
    md_content.append("```")
    md_content.append("output_data/")
    md_content.append("├── analysis_results.csv       # Main dataframe")
    md_content.append("├── analysis_report.md         # This report")
    md_content.append("└── images/")
    md_content.append("    ├── full/                  # Original images")
    md_content.append("    ├── cropped/               # Cropped regions with IoU scores")
    md_content.append("    └── annotated/             # Images with GT boxes drawn")
    md_content.append("```")

    return "\n".join(md_content)

# Generate markdown report
md_report = create_markdown_report(df, summary_stats, output_base_dir)
md_path = os.path.join(output_base_dir, 'analysis_report.md')
with open(md_path, 'w', encoding='utf-8') as f:
    f.write(md_report)

print(f"Markdown report saved to: {md_path}")
print(f"Images saved to: {os.path.join(output_base_dir, 'images')}")

# %%
# Display first few rows of the dataframe
# print("\nFirst 5 rows of the dataframe:")
# print(df.head())

# %%
# Display column info
print(f"\nDataFrame shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
