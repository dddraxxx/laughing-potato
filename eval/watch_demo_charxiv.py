# %%
import os
import json
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw
import numpy as np
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# %%
def format_print_str(long_str, line_length=175):
    formatted_str = '\n'.join([long_str[i:i+line_length] for i in range(0, len(long_str), line_length)])
    return formatted_str

def print_content(content):
    lines = content.splitlines()
    for line in lines:
        print(format_print_str(line))

def extract_thinking_data(ori_image, conversation, image_name, output_dir):
    """Extract thinking process data and save cropped images"""
    turn_depth = 0
    thinking_text = []
    bbox_data = []
    cropped_paths = []
    bbox_idx = 0

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
                try:
                    _bbox = eval(_bbox_str)['arguments']
                    for _box in [_bbox]:
                        _box = _box['bbox_2d']
                        x1, y1, x2, y2 = _box
                        bbox_data.append(_box)

                        # Crop and save image
                        _crop_img = ori_image.crop((x1, y1, x2, y2))

                        # Save cropped image
                        crop_filename = f"{image_name.replace('.jpg', '').replace('.png', '')}_crop_{bbox_idx}.jpg"
                        bbox_idx += 1
                        crop_path = os.path.join(output_dir, 'cropped', crop_filename)
                        Path(crop_path).parent.mkdir(parents=True, exist_ok=True)
                        _crop_img.save(crop_path)
                        cropped_paths.append(crop_path)
                except Exception as e:
                    print(f"Error parsing bbox for {image_name}: {e}")

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
        content_text = content_text.replace('\n', '\\n')

        # Format the turn with bold and red role
        role_html = f'<span style="color: red; font-weight: bold;">{role}</span>'
        # Escape angle brackets to prevent HTML tag rendering in content_text
        content_text_escaped = content_text.replace('<', '&lt;').replace('>', '&gt;')
        dialogue_parts.append(f'{role_html}: "{content_text_escaped}"')

    return '<br>'.join(dialogue_parts)

# %%
# Create output directories
def main():
    parser = argparse.ArgumentParser(description='Process Charxiv evaluation results and generate analysis report')
    parser.add_argument('--home-dir', type=str, default='/scratch/doqihu',
                        help='Home directory path (default: /scratch/doqihu)')
    parser.add_argument('--model_name', type=str, default='trained_152steps',
                        help='Model name for evaluation results (default: trained_152steps)')
    parser.add_argument('--dataset_version', '-dsv', type=str, default='validation',
                        help='Dataset version (default: validation)')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of worker threads for parallel processing (default: 8)')

    args = parser.parse_args()

    home_dir = args.home_dir
    model_name = args.model_name

    output_base_dir = os.path.join(home_dir, f'laughing-potato/eval/output_data/{model_name}')
    output_base_dir = os.path.join(output_base_dir, 'charxiv')
    root_path = os.path.join(home_dir, 'work/eval_data/hgf/charxiv')
    json_path = os.path.join(home_dir, f'laughing-potato/eval_results/charxiv/{model_name}/result_{args.dataset_version}_qwen_acc.jsonl')

    output_base_dir = os.path.join(output_base_dir, args.dataset_version)
    os.makedirs(os.path.join(output_base_dir, 'images', 'full'), exist_ok=True)
    os.makedirs(os.path.join(output_base_dir, 'images', 'cropped'), exist_ok=True)

    # %%
    with open(json_path, 'r') as f:
        lines = f.readlines()
        lines = [json.loads(line) for line in lines]
    line_map = {}
    image_list = []
    for line in lines:
        # Extract image filename from the data
        if 'image' in line:
            image_key = line['image']
            line_map[image_key] = line
            image_list.append(image_key)

    # %%

    def process_case(line_id, output_dir):
        """Process a single case and return data for dataframe"""
        tosee_img = image_list[line_id]
        img_path = os.path.join(root_path, tosee_img)
        question = line_map[tosee_img]['question']
        answer = line_map[tosee_img]['answer']
        pred_ans = line_map[tosee_img]['pred_ans']
        pred_output = line_map[tosee_img]['pred_output']
        accuracy = line_map[tosee_img].get('acc', 0.0)
        correct = accuracy == 1.0

        # Load image
        ori_image_path = os.path.join(root_path, tosee_img)

        try:
            ori_image = Image.open(ori_image_path)
        except Exception as e:
            print(f"Error loading image {ori_image_path}: {e}")
            return None

        # Save full image
        full_image_path = os.path.join(output_dir, 'images', 'full', tosee_img)
        Path(full_image_path).parent.mkdir(parents=True, exist_ok=True)
        ori_image.save(full_image_path)

        # Extract thinking data and save cropped images
        thinking_data = extract_thinking_data(ori_image, pred_output, tosee_img, output_dir)

        return {
            'image_filename': tosee_img,
            'question': question,
            'ground_truth_answer': answer,
            'predicted_answer': pred_ans,
            'full_pred_output': format_dialogue_string(pred_output),
            'is_correct': correct,
            'accuracy_score': accuracy,
            'turn_depth': thinking_data['turn_depth'],
            'full_image_saved_path': full_image_path,
            'cropped_images_paths': json.dumps(thinking_data['cropped_images_paths']),
            'bbox_coordinates': json.dumps(thinking_data['bbox_coordinates']),
            'thinking_conversation': thinking_data['thinking_conversation']
        }

    # Main processing loop
    print(f"Processing all cases with {args.num_workers} worker threads...")
    all_data = []

    # Progress tracking variables
    completed_count = 0
    total_count = len(image_list)
    progress_lock = threading.Lock()

    def process_with_progress(line_id):
        """Wrapper function that includes progress tracking"""
        nonlocal completed_count
        case_data = process_case(line_id, output_base_dir)

        with progress_lock:
            completed_count += 1
            if completed_count % 50 == 0 or completed_count == total_count:
                print(f"Processing progress: {completed_count}/{total_count} ({completed_count/total_count*100:.1f}%)")

        return case_data

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        # Use map to maintain order and process all cases
        results = list(executor.map(process_with_progress, range(len(image_list))))

        # Filter out None results
        all_data = [case_data for case_data in results if case_data is not None]

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
        'bbox_usage_cases': (df['bbox_coordinates'].apply(lambda x: len(json.loads(x)) > 0 if x else False)).sum()
    }

    # Create markdown report
    def create_markdown_report(df, summary_stats, output_dir):
        """Create a comprehensive markdown report"""
        md_content = []

        # Title and summary
        md_content.append("# Charxiv Evaluation Analysis Report\n")
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
        md_content.append(f"| Cases Using Tool | {summary_stats['bbox_usage_cases']} |")
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

        # Turn Depth Analysis
        md_content.append("### Turn Depth Analysis\n")
        turn_depth_stats = df['turn_depth'].value_counts().sort_index()
        md_content.append("| Turn Depth | Count | Percentage |")
        md_content.append("|------------|-------|------------|")
        for depth, count in turn_depth_stats.items():
            percentage = (count / len(df)) * 100
            md_content.append(f"| {depth} | {count} | {percentage:.1f}% |")
        md_content.append("")

        # Tool Usage Analysis
        md_content.append("### Tool Usage Analysis\n")
        bbox_usage = df['bbox_coordinates'].apply(lambda x: len(json.loads(x)) > 0 if x else False)
        tool_used_count = bbox_usage.sum()
        tool_not_used_count = len(df) - tool_used_count

        md_content.append("| Tool Usage | Count | Percentage |")
        md_content.append("|------------|-------|------------|")
        md_content.append(f"| Used Tool | {tool_used_count} | {(tool_used_count/len(df)*100):.1f}% |")
        md_content.append(f"| No Tool | {tool_not_used_count} | {(tool_not_used_count/len(df)*100):.1f}% |")
        md_content.append("")

        # Accuracy by Tool Usage
        md_content.append("### Accuracy by Tool Usage\n")
        tool_used_df = df[bbox_usage]
        tool_not_used_df = df[~bbox_usage]

        md_content.append("| Tool Usage | Correct | Total | Accuracy |")
        md_content.append("|------------|---------|-------|----------|")

        if len(tool_used_df) > 0:
            tool_acc = tool_used_df['is_correct'].mean()
            md_content.append(f"| Used Tool | {tool_used_df['is_correct'].sum()} | {len(tool_used_df)} | {tool_acc:.3f} ({tool_acc*100:.1f}%) |")

        if len(tool_not_used_df) > 0:
            no_tool_acc = tool_not_used_df['is_correct'].mean()
            md_content.append(f"| No Tool | {tool_not_used_df['is_correct'].sum()} | {len(tool_not_used_df)} | {no_tool_acc:.3f} ({no_tool_acc*100:.1f}%) |")

        md_content.append("")

        # Sample Cases
        md_content.append("## Sample Cases\n")

        # Helper function to create image HTML
        def format_image_path(path, max_width=150, max_height=150, use_base64=False):
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
                # Use relative path
                try:
                    rel_path = os.path.relpath(path, output_dir)
                    return f'<img src="{rel_path}" style="max-width: {max_width}px; max-height: {max_height}px; height: auto; width: auto; object-fit: contain; border: 1px solid #ddd; border-radius: 4px;">'
                except:
                    return f'<span style="color: red;">Path error</span>'

        def format_cropped_images(cropped_paths_json, max_width=100, max_height=100, use_base64=False):
            """Convert cropped image paths JSON to HTML img tags"""
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

                return f'<div style="display: flex; flex-wrap: wrap; gap: 2px; align-items: center;">{" ".join(img_tags)}</div>'
            except Exception as e:
                print(f"Error processing cropped images: {e}")
                return '<span style="color: red;">Cropped images error</span>'

        def format_correctness(is_correct):
            """Format correctness as emoji"""
            return '✅' if is_correct else '❌'

        def format_accuracy_score(score):
            """Format accuracy score with color coding"""
            if pd.isna(score):
                return ''
            color = '#28a745' if score == 1.0 else '#dc3545'
            return f'<span style="color: {color}; font-weight: bold;">{score:.1f}</span>'

        # Table formatters
        def get_table_formatters():
            return {
                'Full Image': lambda x: format_image_path(x, 150, 150, use_base64=False),
                'Cropped Regions': lambda x: format_cropped_images(x, 100, 100, use_base64=False),
                'Correct': format_correctness,
                'Accuracy': format_accuracy_score,
                'Question': lambda x: x if not pd.isna(x) else '',
                'Ground Truth': lambda x: x if not pd.isna(x) else '',
                'Prediction': lambda x: x if not pd.isna(x) else '',
                'Full Dialogue': lambda x: x if not pd.isna(x) else ''
            }

        # Column configuration
        display_columns = ['image_filename', 'question', 'ground_truth_answer', 'predicted_answer',
                          'full_pred_output', 'is_correct', 'accuracy_score', 'turn_depth', 'full_image_saved_path',
                          'cropped_images_paths']

        column_names = {
            'image_filename': 'Image',
            'question': 'Question',
            'ground_truth_answer': 'Ground Truth',
            'predicted_answer': 'Prediction',
            'full_pred_output': 'Full Dialogue',
            'is_correct': 'Correct',
            'accuracy_score': 'Accuracy',
            'turn_depth': 'Turns',
            'full_image_saved_path': 'Full Image',
            'cropped_images_paths': 'Cropped Regions'
        }

        formatters = get_table_formatters()

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
</style>

{html_table}
"""
            return styled_table

        # Wrong cases (incorrect)
        wrong_cases = df[(df['is_correct'] == False)]
        if len(wrong_cases) > 0:
            print(f"Wrong cases: {len(wrong_cases)}")
            sample_size = min(10, len(wrong_cases))
            wrong_cases_sample = wrong_cases.sample(sample_size, random_state=42)
            md_content.append("### Wrong Cases (Incorrect)\n")

            wrong_display = wrong_cases_sample[display_columns].copy().rename(columns=column_names)
            styled_table = generate_styled_table(wrong_display, 'wrong-cases-table', formatters)
            md_content.append(styled_table)
            md_content.append("")

        # Correct cases with tool usage
        correct_with_tool = df[(df['is_correct'] == True) & (df['bbox_coordinates'].apply(lambda x: len(json.loads(x)) > 0 if x else False))]
        if len(correct_with_tool) > 0:
            print(f"Correct cases with tool: {len(correct_with_tool)}")
            sample_size = min(5, len(correct_with_tool))
            correct_sample = correct_with_tool.sample(sample_size, random_state=42)
            md_content.append("### Best Performing Cases (Correct + Used Tool)\n")

            best_display = correct_sample[display_columns].copy().rename(columns=column_names)
            styled_table = generate_styled_table(best_display, 'best-cases-table', formatters)
            md_content.append(styled_table)
            md_content.append("")

        # cases with more than 2 turns
        # for each turn number > 2, pick up to 3 images
        max_turns = df['turn_depth'].max()
        # For each turn number > 2, pick up to 3 images and display as a table
        turn_samples = []
        for turn_num in range(3, int(max_turns) + 1):
            turn_cases = df[df['turn_depth'] == turn_num]
            if len(turn_cases) == 0:
                continue
            sample_size = min(3, len(turn_cases))
            turn_sample = turn_cases.sample(sample_size, random_state=42)
            turn_samples.append(turn_sample)
        if len(turn_samples) > 0:
            turn_samples_df = pd.concat(turn_samples)
            md_content.append("### Cases with Turn Depth > 2\n")
            turn_display = turn_samples_df[display_columns].copy().rename(columns=column_names)
            styled_table = generate_styled_table(turn_display, 'multi-turn-cases-table', formatters)
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
        md_content.append("| accuracy_score | float | Accuracy score from judge (0.0 or 1.0) |")
        md_content.append("| turn_depth | integer | Number of conversation turns |")
        md_content.append("| full_image_saved_path | string | Path to saved full image |")
        md_content.append("| cropped_images_paths | JSON string | List of cropped image paths |")
        md_content.append("| bbox_coordinates | JSON string | Predicted bounding box coordinates |")
        md_content.append("| thinking_conversation | string | Full conversation text |")
        md_content.append("")

        # File Structure
        md_content.append("## Generated Files Structure\n")
        md_content.append("```")
        md_content.append(f"{output_base_dir}/")
        md_content.append("├── analysis_results.csv       # Main dataframe")
        md_content.append("├── analysis_report.md         # This report")
        md_content.append("└── images/")
        md_content.append("    ├── full/                  # Original images")
        md_content.append("    └── cropped/               # Cropped regions from tool usage")
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
    # Display column info
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

if __name__ == "__main__":
    main()
