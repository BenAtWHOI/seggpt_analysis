import argparse
import atexit
import csv
import os
import time
import cv2 as cv
from file_utils import clear_directory, get_subfolder

def denormalize(line, img_width, img_height):
    # Convert YOLO format to pixel coordinates
    box = line.split(' ')
    x = float(box[1]) * img_width
    y = float(box[2]) * img_height
    w = float(box[3]) * img_width
    h = float(box[4]) * img_height
    x1 = int(x - w / 2)
    y1 = int(y - h / 2)
    x2 = int(x + w / 2)
    y2 = int(y + h / 2)
    return [x1, y1, x2, y2]

def calculate_iou(box1, box2, img_width, img_height):
    # Denormalize yolo data
    box1 = denormalize(box1, img_width, img_height)
    box2 = denormalize(box2, img_width, img_height)

    # Calculate intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate IoU
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

def compare_yolo_files(frame_name, seggpt_file, yolo_file, img_width, img_height, iou_threshold):
    # Dictionary of collected analysis information
    analysis_data = {
        'frame_name': frame_name,
        'seggpt_boxes': [line.strip() for line in seggpt_file],
        'yolo_boxes': [line.strip() for line in yolo_file],
        'intersecting_boxes': []
    }
    
    # Count the number of intersecting bounding boxes between the two files
    for seggpt_box in analysis_data['seggpt_boxes']:
        for yolo_box in analysis_data['yolo_boxes']: 
            iou = calculate_iou(seggpt_box, yolo_box, img_width, img_height)
            if iou >= iou_threshold:
                # Record box intersection
                analysis_data['intersecting_boxes'].append({
                    'seggpt_box': seggpt_box,
                    'yolo_box': yolo_box,
                    'iou': iou
                })
                # Remove boxes from analysis data for easier processing down the line
                analysis_data['seggpt_boxes'].remove(seggpt_box)
                analysis_data['yolo_boxes'].remove(yolo_box)
                # Count each box from the seggpt file only once to avoid duplicates
                break
    
    # Measure and record accuracy
    intersecting_box_count = len(analysis_data['intersecting_boxes'])
    seggpt_box_count = len(analysis_data['seggpt_boxes'])
    analysis_data['accuracy'] = intersecting_box_count / seggpt_box_count if seggpt_box_count > 0 else 0

    return analysis_data

def format_box(box):
    # Format box for readability
    return f"[{box[0]} {box[1]} {box[2]} {box[3]}]"

def retrieve_files(directory):
    # Pull all files out of subfolders in the directory
    return set(
        file
        for _, _, files in os.walk(directory)
        for file in files
    )

def main():
    # Time execution of program
    start_time = time.time()
    atexit.register(lambda: print(f"\nTime elapsed: {time.time() - start_time:.2f}s"))
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--seggpt_directory', type=str, default='seggpt_yolo_files', help='Directory of seggpt yolo files to compare')
    parser.add_argument('--yolo_directory', type=str, default='YOLO_output', help='Directory of yolo run yolo files to compare')
    parser.add_argument('--masks_directory', type=str, default='output', help='Directory of masks to retrieve image size from')
    parser.add_argument('--output_directory', type=str, default='analysis_results', help='Directory to output text analysis data to')
    parser.add_argument('--iou_threshold', type=float, default=0.50, help='The value representing how much of the areas of two bounding boxes must intersect to be considered \'overlapping\'')
    parser.add_argument('--limit', type=int, help='Limit on number of results to process')
    args = parser.parse_args()
    
    # Store files to match results
    seggpt_directory = args.seggpt_directory
    yolo_directory = args.yolo_directory
    masks_directory = args.masks_directory
    output_directory = args.output_directory
    
    seggpt_files = retrieve_files(seggpt_directory)
    yolo_files = retrieve_files(yolo_directory)
    mask_files = retrieve_files(masks_directory)

    # Apply limit argument
    seggpt_files = seggpt_files[:args.limit] if args.limit else seggpt_files

    # Remove existing contents of analysis directory
    clear_directory(output_directory)

    results = []
    for seggpt_filename in seggpt_files:
        # Try to match corresponding yolo file and output mask file, ignoring file extensions
        matching_yolo_file = next((file for file in yolo_files if file == seggpt_filename), None)
        matching_output_file = next((file for file in mask_files if file[:-4] == seggpt_filename[:-4]), None)
        
        if matching_yolo_file and matching_output_file:
            # Compare file contents
            img = cv.imread(os.path.join(masks_directory, get_subfolder(seggpt_filename), matching_output_file))
            h, w, _ = img.shape
            with open(os.path.join(seggpt_directory, get_subfolder(seggpt_filename), seggpt_filename), 'r') as s, open(os.path.join(yolo_directory, matching_yolo_file), 'r') as y:
                # Analyze and store data
                analysis_data = compare_yolo_files(seggpt_filename[:-4], s, y, w, h, args.iou_threshold)
                results.append(analysis_data)

    # Write all results to file
    matched_iou, matched_iou_count = 0.0, 0
    for result in results:
        # Add get_subfolder() call after output_directory to structure output 
        file = result['frame_name']
        with open(os.path.join(output_directory, get_subfolder(file), f'{file}.csv'), 'w') as output_file:
            fieldnames = ['frame_name', 'seggpt_class', 'seggpt_x', 'seggpt_y', 'seggpt_w', 'seggpt_h', 'yolo_class', 'yolo_x', 'yolo_y', 'yolo_w', 'yolo_h', 'matching', 'iou']
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()

            # Utility function to write line
            def writerow(frame_name, seggpt_box, yolo_box, matching=False, iou=0.0):
                seggpt_parts = seggpt_box.split() if seggpt_box else [''] * 5
                yolo_parts = yolo_box.split() if yolo_box else [''] * 5
                writer.writerow({
                    'frame_name': frame_name,
                    'seggpt_class': seggpt_parts[0],
                    'seggpt_x': seggpt_parts[1],
                    'seggpt_y': seggpt_parts[2],
                    'seggpt_w': seggpt_parts[3],
                    'seggpt_h': seggpt_parts[4],
                    'yolo_class': yolo_parts[0],
                    'yolo_x': yolo_parts[1],
                    'yolo_y': yolo_parts[2],
                    'yolo_w': yolo_parts[3],
                    'yolo_h': yolo_parts[4],
                    'matching': matching,
                    'iou': iou
                })

            # Write all boxes to the file
            frame_name = result['frame_name']
            for intersecting_box in result['intersecting_boxes']:
                matched_iou += intersecting_box['iou']
                writerow(frame_name, intersecting_box['seggpt_box'], intersecting_box['yolo_box'], True, intersecting_box['iou'])
            for seggpt_box in result['seggpt_boxes']:
                writerow(frame_name, seggpt_box, '', False, 0.0)
            for yolo_box in result['yolo_boxes']:
                writerow(frame_name, '', yolo_box, False, 0.0)
        
        # Add iou to the running total
        matched_iou_count += len(result['intersecting_boxes'])

    # Write various metrics across all results to file
    with open(os.path.join(output_directory, 'analysis.txt'), 'w') as file:
        output = 'Analysis across all results:\n\n'
        output += f'Overall accuracy: {sum(result['accuracy'] for result in results) / len(results) if len(results) > 0 else 0}\n'
        output += f'Average IoU across matched boxes: {matched_iou / matched_iou_count if matched_iou_count > 0 else 0}\n'
        file.write(output)

if __name__ == '__main__':
    main()