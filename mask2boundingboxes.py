import argparse
import atexit
import multiprocessing as mp
import os
import time
import cv2 as cv
from file_utils import clear_directory, get_subfolder

def boxes_should_merge(box1, box2, merge_distance):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    horizontal_overlap = (x1 <= x2 <= x1 + w1) or (x2 <= x1 <= x2 + w2)
    vertical_overlap = (y1 <= y2 <= y1 + h1) or (y2 <= y1 <= y2 + h2)

    horizontal_distance = min(abs(x1 - (x2 + w2)), abs(x2 - (x1 + w1)))
    vertical_distance = min(abs(y1 - (y2 + h2)), abs(y2 - (y1 + h1)))

    # Boxes should merge if overlapping or any edge of either box is within merge distance of the other box
    return (horizontal_overlap or horizontal_distance <= merge_distance) and (vertical_overlap or vertical_distance <= merge_distance)

def merge_boxes(contours, merge_distance):
    # Merge every possible combination of contours
    while True:
        merged = False
        combined_contours = []
        # Loop through list of remaining contours
        while contours:
            current = contours.pop(0)
            for other in contours[:]:
                if boxes_should_merge(current, other, merge_distance):
                    x = min(current[0], other[0])
                    y = min(current[1], other[1])
                    w = max(current[0] + current[2], other[0] + other[2]) - x
                    h = max(current[1] + current[3], other[1] + other[3]) - y
                    current = (x, y, w, h)
                    contours.remove(other)
                    merged = True
            combined_contours.append(current)
        # All combinations of contours have been merged
        if not merged:
            break
        contours = combined_contours
    return combined_contours

def generate_bounding_boxes(args):
    directory, file, min_area, mask_output_directory, yolo_output_directory, padding, merge_distance = args

    # Convert the image to binary and find the mask contours
    img = cv.imread(os.path.join(directory, file), cv.IMREAD_GRAYSCALE)
    _, img = cv.threshold(img, 0, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Save the binary image and reopen it as a color image to draw bounding boxes
    cv.imwrite(os.path.join(mask_output_directory, get_subfolder(file), file), img)
    img = cv.imread(os.path.join(mask_output_directory, get_subfolder(file), file), cv.IMREAD_COLOR)
    img_height, img_width, _ = img.shape

    # Combine overlapping and nearby contours
    bounding_rects = [cv.boundingRect(c) for c in contours]
    combined_contours = merge_boxes(bounding_rects, merge_distance)

    # Generate and record the bounding boxes with padding
    yolos = []
    for x, y, w, h in combined_contours:
        # Only draw and record contours that are greater than the min area
        if w * h >= min_area:
            # Draw the bounding box with padding
            point1 = (max(x - padding, 0), max(y - padding, 0))
            point2 = (min(x + w + padding, img_width), min(y + h + padding, img_height))
            outline_color = (0, 255, 0)
            cv.rectangle(img, point1, point2, outline_color, 2)

            # Convert to YOLO format and record box
            x_center = (x + w / 2) / img_width
            y_center = (y + h / 2) / img_height
            width = w / img_width
            height = h / img_height
            yolos.append(f'0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}')

    # Save image to directory and write all bounding boxes to file in YOLO format
    cv.imwrite(os.path.join(mask_output_directory, get_subfolder(file), file), img)
    with open(os.path.join(yolo_output_directory, get_subfolder(file), f'{file[:-4]}.txt'), 'w') as file: 
        for record in yolos:
            file.write(f'{record}\n')

def main():
    # Time execution of program
    start_time = time.time()
    atexit.register(lambda: print(f"\nTime elapsed: {time.time() - start_time:.2f}s"))

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_directory', type=str, default='masks', help='Directory of masks to process')
    parser.add_argument('--mask_output_directory', type=str, default='output', help='Directory containing the processed images')
    parser.add_argument('--yolo_output_directory', type=str, default='seggpt_yolo_files', help='Directory containing the generated masks in yolo format')
    parser.add_argument('--min_area', type=int, default=25, help='The minimum area in pixels for a bounding box to be drawn')
    parser.add_argument('--padding', type=int, default=0, help='The amount of empty space in pixels to include around each bounding box')
    parser.add_argument('--merge_area', type=int, default=50, help='The minimum amount of pixels between two bounding boxes to be considered separate boxes and thus not merged')
    parser.add_argument('--limit', type=int, help='Limit on number of masks to process')
    args = parser.parse_args()

    # Remove existing contents of output directories
    mask_dir, yolo_dir = args.mask_output_directory, args.yolo_output_directory
    clear_directory(mask_dir)
    clear_directory(yolo_dir)

    # Apply limit argument
    lim, input_dir = args.limit, args.input_directory
    files = os.listdir(input_dir)
    files = files[:lim] if lim else files

    # Set number of processes to CPU core count
    core_count = mp.cpu_count()
    pool = mp.Pool(processes=core_count)

    # Process images in parallel
    processes = [(input_dir, file, args.min_area, mask_dir, yolo_dir, args.padding, args.merge_area) for file in files]
    pool.map(generate_bounding_boxes, processes)
    pool.close()
    pool.join()

if __name__ == '__main__':
    main()