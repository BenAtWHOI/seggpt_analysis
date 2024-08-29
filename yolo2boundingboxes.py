import argparse
import os
import cv2 as cv

def calculate_dimensions(image_width, image_height, center_x, center_y, width, height):
    x = int((center_x - width/2) * image_width)
    y = int((center_y - height/2) * image_height)
    w = int(width * image_width)
    h = int(height * image_height)
    return (x, y, w, h)

def generate_bounding_boxes(mask_directory, mask_file, yolo_directory, yolo_file_path, output_directory):
    print(f'Generating bounding boxes for mask: \'{mask_file}\'')

    # Read the image and save dimensions for bounding box calculations
    img = cv.imread(os.path.join(mask_directory, mask_file), cv.IMREAD_COLOR)
    img_height, img_width, _ = img.shape

    # Read all bounding boxes from the corresponding yolo file
    with open(os.path.join(yolo_directory, yolo_file_path), 'r') as yolo_file:
        for line in yolo_file:
            box = line.split(' ')
            print(f'LINE: {line}')
            x, y, w, h = calculate_dimensions(img_width, img_height, float(box[1]), float(box[2]), float(box[3]), float(box[4]))
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            print(f'Recording bounding box  starting at {x}, {y} with dimensions {w} x {h} px')
    
    # Save the image
    cv.imwrite(os.path.join(output_directory, mask_file), img)

    print('\n=========================\n')


def clear_directory(directory):
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            os.unlink(file_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_directory', type=str, help='Directory of yolo files to process')
    parser.add_argument('--mask_directory', type=str, help='The directory of masks to draw the yolo bounding boxes on')
    parser.add_argument('--output_directory', type=str, help='Where to output the processed images')
    parser.add_argument('--limit', type=str, help='Limit on number of masks to process')
    args = parser.parse_args()

    # Remove existing contents of output directories
    output_directory = args.output_directory if args.output_directory else 'output'
    clear_directory(output_directory)

    # Generate bounding boxes for the specified number of masks
    limit = args.limit
    yolo_files = set(os.listdir(args.yolo_directory))
    mask_files = os.listdir(args.mask_directory)
    if limit:
        mask_files = mask_files[:int(limit)]
    for mask_file in mask_files:
        for file in yolo_files:
            if file[:-4] == mask_file[:-4]:
                print(f'{mask_file[:-4]}, {file[:-4]}')
        yolo_file = next((file for file in yolo_files if file[:-4] == mask_file[:-4]), None)
        if yolo_file:
            generate_bounding_boxes(args.mask_directory, mask_file, args.yolo_directory, yolo_file, output_directory)


if __name__ == '__main__':
    main()