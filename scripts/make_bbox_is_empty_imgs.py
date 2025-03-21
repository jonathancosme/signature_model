import os
import cv2
import yaml
import random
import numpy as np
import argparse
from tqdm import tqdm

def main(input_images_dir, input_labels_dir, annotation_yaml, output_dir):
	# Load class names from annotation YAML
	with open(annotation_yaml, 'r') as f:
	    yaml_data = yaml.safe_load(f)

	class_id_to_name = yaml_data['names']

	# Assign unique colors to each class
	num_classes = len(class_id_to_name)
	random.seed(42)  # Ensures same colors each run
	class_colors = {i: tuple(np.random.randint(0, 255, 3).tolist()) for i in range(num_classes)}

	# Function to convert YOLO bbox format to pixel coordinates
	def yolo_to_xyxy(bbox, img_w, img_h):
	    x_c, y_c, w, h = bbox
	    x_c *= img_w
	    y_c *= img_h
	    w *= img_w
	    h *= img_h
	    x_min = int(x_c - w / 2)
	    y_min = int(y_c - h / 2)
	    x_max = int(x_c + w / 2)
	    y_max = int(y_c + h / 2)
	    return x_min, y_min, x_max, y_max

	# Function to draw labeled bounding boxes on an image
	def draw_bboxes(image_path, label_path, output_path):
	    image = cv2.imread(image_path)
	    if image is None:
	        tqdm.write(f"Error: Cannot read image {image_path}")
	        return

	    img_h, img_w, _ = image.shape

	    # Read label file
	    if not os.path.exists(label_path):
	        tqdm.write(f"Warning: No label file found for {image_path}")
	        return

	    with open(label_path, 'r') as f:
	        lines = f.readlines()

	    for line in lines:
	        parts = line.strip().split()
	        if len(parts) < 5:
	            continue

	        class_id = int(parts[0])
	        bbox = list(map(float, parts[1:5]))

	        if class_id not in class_id_to_name:
	            tqdm.write(f"Warning: Class ID {class_id} not found in annotation YAML")
	            continue

	        class_name = class_id_to_name[class_id]
	        x_min, y_min, x_max, y_max = yolo_to_xyxy(bbox, img_w, img_h)
	        color = class_colors[class_id]
	        if 'not_empty' in class_name:
	            color = (0, 255, 0)
	            class_name = 'is_filled'
	        else:
	            color = (0, 0, 255)
	            class_name = 'not_filled'

	        # Draw bounding box
	        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

	        # Display label text
	        label_text = f"{class_name}"
	        font_scale = 0.65
	        font_thickness = 1
	        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
	        text_x, text_y = x_min, y_min - 5

	        # Draw background rectangle for text
	        cv2.rectangle(image, (text_x, text_y - text_size[1] - 2), (text_x + text_size[0] + 4, text_y), color, -1)

	        # Put text on image
	        cv2.putText(image, label_text, (text_x + 2, text_y - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

	    # Save output image
	    os.makedirs(os.path.dirname(output_path), exist_ok=True)
	    cv2.imwrite(output_path, image)
	    tqdm.write(f"Saved: {output_path}")

	# Create output directory if it doesn't exist
	os.makedirs(output_dir, exist_ok=True)

	# Get list of .txt files in input_dir
	img_files = [f for f in os.listdir(input_images_dir)]
	img_files.sort()
	txt_files = [f for f in os.listdir(input_labels_dir)]
	txt_files.sort()

	for img_file, txt_file in tqdm(zip(img_files, txt_files), position=0, leave=False):
		img_path = os.path.join(input_images_dir, img_file)
		label_path = os.path.join(input_labels_dir, txt_file)

		output_img_path = os.path.join(output_dir, img_file)
		draw_bboxes(img_path, label_path, output_img_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process text files.")
    parser.add_argument("--input-images-dir", required=True, help="Path to the input images directory")
    parser.add_argument("--input-labels-dir", required=True, help="Path to the input labels directory")
    parser.add_argument("--output-dir", required=True, help="Path to the output directory")
    parser.add_argument("--annotation-yaml", required=True, help="Path to the annotation yaml file")
    args = parser.parse_args()
    
    main(args.input_images_dir, args.input_labels_dir, args.annotation_yaml, args.output_dir)