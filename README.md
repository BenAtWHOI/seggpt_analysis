# Usage

Convert SegGPT masks to bounding boxes using OpenCV
```
python mask2boundingboxes.py
  --input_directory {directory of masks to draw bounding boxes around}
  --mask_output_directory {directory to output masks to with bounding boxes drawn}
  --yolo_output_directory {directory to output text files with the bounding boxes in YOLO format}
  --gray_threshold {the level of light (0-255) to detect as part of the mask and draw a bounding box around, default=100}
  --limit {a limit on the number of masks to attempt to process}
```

Analyze the accuracy of the bounding boxes drawn around the SegGPT masks, compared to a directory of YOLO text files from previous runs
```
python compare_results.py
  --seggpt_directory {directory containing the bounding box YOLO files outputted by mask2boundingboxes.py}
  --yolo_directory {directory containing the bounding box YOLO files to compare to}
  --output_directory {directory containing mask images (processed or unprocessed) to retrieve image dimensions from for denormalization of YOLO data}
  --iou_threshold {a float between 0-1 representing what percentage of the volume of a bounding box must overlap to consider "matching", default=0.50}
```

Manually draw bounding boxes on masks given existing YOLO data (for visual inspection)
```
python yolo2boundingboxes.py
  --yolo_directory {directory containing text files of bounding box data in YOLO format}
  --mask_directory {directory of masks to draw bounding boxes around}
  --output_directory {directory to output masks to with bounding boxes drawn}
  --limit {a limit on the number of masks to attempt to process}
```
