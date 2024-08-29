import argparse
import atexit
import os
import time
import cv2 as cv

def generate_bounding_boxes(directory, file, gray_threshold, mask_output_directory, yolo_output_directory):
    print(f'Generating bounding boxes for mask: \'{file}\'')

    # Find the contours of the mask
    img = cv.imread(f'./{directory}/{file}', cv.IMREAD_COLOR)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, gray_threshold, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Skip processing if no bounding boxes can be formed
    if (len(contours) == 0):
        print(f'No contours found in \'{file}\', skipping')
        print('\n=========================\n')
        return
    
    # Save image dimensions for bounding box calculations
    img_height, img_width, _ = img.shape
    print(f'Image dimensions: {img_width} x {img_height}')

    # Generate a bounding box, then draw and record it
    yolos = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv.boundingRect(contour)

        # Only process bounding boxes that are at least 10x10 pixels
        if w >= 10 and h >= 10:
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Convert to YOLO format
            x_center = (x + w / 2) / img_width
            y_center = (y + h / 2) / img_height
            width = w / img_width
            height = h / img_height

            print(f'Recording bounding box starting at {x}, {y} with dimensions {w} x {h} px ({x_center:.6f}, {y_center:.6f}, {width:.6f}, {height:.6f})')
            
            # Append YOLO format string (class x_center y_center width height)
            yolos.append(f'0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}')
    
    # Save the image
    cv.imwrite(os.path.join(mask_output_directory, file), img)
    
    # Write all bounding boxes to file in yolo format
    with open(os.path.join(yolo_output_directory, f'{file[:-4]}.txt'), 'w') as file:
        for record in yolos:
            file.write(f'{record}\n')

    print('\n=========================\n')

def clear_directory(directory):
    # Clear all files in the directoryZ
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            os.unlink(file_path)

def main():
    # Time execution of program
    start_time = time.time()
    atexit.register(lambda: print(f"\nTime elapsed: {time.time() - start_time:.2f}s"))

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_directory', type=str, default='masks', help='Directory of masks to process')
    parser.add_argument('--mask_output_directory', type=str, default='output', help='Directory containing the processed images')
    parser.add_argument('--yolo_output_directory', type=str, default='seggpt_yolo_files', help='Directory containing the generated masks in yolo format')
    parser.add_argument('--gray_threshold', type=int, default=100, help='The minimum pixel light level (0-255) to draw bounding boxes around')
    parser.add_argument('--limit', type=int, help='Limit on number of masks to process')
    args = parser.parse_args()

    # Remove existing contents of output directories
    mask_dir, yolo_dir = args.mask_output_directory, args.yolo_output_directory
    clear_directory(mask_dir)
    clear_directory(yolo_dir)

    # Generate bounding boxes for the specified number of masks
    lim, input_dir = args.limit, args.input_directory
    files = os.listdir(input_dir)
    if lim:
        files = files[:lim]
    for file in files:
        generate_bounding_boxes(input_dir, file, args.gray_threshold, mask_dir, yolo_dir)


if __name__ == '__main__':
    main()