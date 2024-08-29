import argparse
import atexit
import os
import time
import cv2 as cv
import numpy as np

def read_yolo_file(file, img_width, img_height):
    # Extract all bounding boxes from the file
    boxes = []
    for line in file:
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
        boxes.append([x1, y1, x2, y2])
    return np.array(boxes)

def read_yolo_file_optimized(file, img_width, img_height):
    # Read all lines at once
    lines = np.loadtxt(file, delimiter=' ', usecols=(1,2,3,4))
    
    # Vectorized calculations
    lines[:, [0,2]] *= img_width
    lines[:, [1,3]] *= img_height
    
    # Calculate corners
    x = lines[:, 0]
    y = lines[:, 1]
    w = lines[:, 2]
    h = lines[:, 3]
    
    boxes = np.column_stack([
        (x - w/2).astype(int),
        (y - h/2).astype(int),
        (x + w/2).astype(int),
        (y + h/2).astype(int)
    ])
    
    return boxes

def calculate_iou(box1, box2):
    # Calculate intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    # Calculate and return IoU
    return intersection / union if union > 0 else 0

def compare_yolo_files(seggpt_file, yolo_file, img_width, img_height, iou_threshold):
    seggpt_boxes = read_yolo_file(seggpt_file, img_width, img_height)
    yolo_boxes = read_yolo_file(yolo_file, img_width, img_height)
    
    # Count the number of intersecting bounding boxes between the two files
    intersecting_boxes = 0
    for seggpt_box in seggpt_boxes:
        match = False
        for yolo_box in yolo_boxes: 
            iou = calculate_iou(seggpt_box, yolo_box)
            if iou > iou_threshold:
                # Count each box from the seggpt file only once to avoid duplicates
                print(f'Box intersection: {seggpt_box}, {yolo_box} ({iou})')
                intersecting_boxes += 1
                match = True
                break 
        if not match:
            print(f'No match for SegGPT box: {seggpt_box}')
   
    accuracy = intersecting_boxes / len(seggpt_boxes) if len(seggpt_boxes) > 0 else 0
    print(f'Intersecting boxes: {intersecting_boxes} / {len(seggpt_boxes)} ({accuracy})')
    return intersecting_boxes

def main():
    # Time execution of program
    start_time = time.time()
    atexit.register(lambda: print(f"\nTime elapsed: {time.time() - start_time:.2f}s"))
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--seggpt_directory', type=str, help='Directory of seggpt yolo files to compare')
    parser.add_argument('--yolo_directory', type=str, help='Directory of yolo run yolo files to compare')
    parser.add_argument('--output_directory', type=str, help='Directory of output masks to retrieve image size from')
    parser.add_argument('--iou_threshold', type=float, default=0.50, help='The value representing how much of the areas of two bounding boxes must intersect to be considered \'overlapping\'')
    args = parser.parse_args()

    # Store directories and filenames to match results
    seggpt_directory = args.seggpt_directory if args.seggpt_directory else 'seggpt_yolo_files'
    yolo_directory = args.yolo_directory if args.yolo_directory else 'YOLO_output'
    output_directory = args.output_directory if args.output_directory else 'output'
    
    seggpt_files = set(os.listdir(seggpt_directory))
    yolo_files = set(os.listdir(yolo_directory))
    output_files = set(os.listdir(output_directory))

    # Compare each seggpt mask to the files in the yolo output directories
    matched_count = 0
    accuracy = 0.0
    for seggpt_filename in seggpt_files:
        print('========================')
        print(f'Matching file: {seggpt_filename}')
        
        # Try to match corresponding yolo file and output mask file, ignoring file extensions
        matching_yolo_file = next((file for file in yolo_files if file == seggpt_filename), None)
        matching_output_file = next((file for file in output_files if file[:-4] == seggpt_filename[:-4]), None)
        
        if matching_yolo_file and matching_output_file:
            # Compare file contents
            print(f'Corresponding yolo and mask file found!')
            img = cv.imread(os.path.join(output_directory, matching_output_file))
            h, w, _ = img.shape
            with open(os.path.join(seggpt_directory, seggpt_filename), 'r') as s, open(os.path.join(yolo_directory, matching_yolo_file), 'r') as y:
                # Measure total accuracy over all files
                accuracy += compare_yolo_files(s, y, w, h, args.iou_threshold)
            matched_count += 1
        else:
            # Skip analysis as there is no corresponding files
            print('No match found.')
    
    print('========================')
    print(f'{matched_count} file(s) matched between {len(seggpt_files)} SegGPT generated yolo files, {len(yolo_files)} YOLO run results, and {len(output_files)} mask files.')
    print(f'Overall accuracy: {accuracy / matched_count}')

if __name__ == '__main__':
    main()