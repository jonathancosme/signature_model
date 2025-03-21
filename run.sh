#!/bin/bash

# Function to print usage
usage() {
    echo "Usage: $0 --input <input_folder> --form <form_folder>"
    exit 1
}

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --input) INPUT_FOLDER="$2"; shift ;;
        --form) FORM_FOLDER="$2"; shift ;;
        *) usage ;;
    esac
    shift
done

# Ensure both arguments are provided
if [[ -z "$INPUT_FOLDER" || -z "$FORM_FOLDER" ]]; then
    usage
fi

# Define the full paths
INPUT_PATH="./inputs/$INPUT_FOLDER"
FORM_PATH="./model_weights/$FORM_FOLDER"

# Check if input directory exists
if [[ ! -d "$INPUT_PATH" ]]; then
    echo "Error: Input directory '$INPUT_PATH' does not exist."
    exit 1
fi

# Check if form directory exists
if [[ ! -d "$FORM_PATH" ]]; then
    echo "Error: Form directory '$FORM_PATH' does not exist."
    exit 1
fi

# Iterate through subfolders in numeric ascending order
for dir in $(ls -v "$INPUT_PATH" | grep -E '^[0-9]+$'); do
    FULL_PAGE_PATH="$INPUT_PATH/$dir"
    if [[ -d "$FULL_PAGE_PATH" ]]; then
        export PAGE="$dir"
        echo "Processing directory: $FULL_PAGE_PATH"

        DETECT_FORM_PATH="$FORM_PATH/$PAGE/detect_form"
        DETECT_FIELDS_PATH="$FORM_PATH/$PAGE/detect_fields"

        echo "    $DETECT_FORM_PATH"

        # Variables for detect_form
        WEIGHTS="$DETECT_FORM_PATH/best.pt"
        SOURCE="$INPUT_PATH/$PAGE"
        PROJECT="./runs/$INPUT_FOLDER/intermediate/crop_form_imgs"
        NAME="$PAGE"

        echo "        WEIGHTS = $WEIGHTS"
        echo "        SOURCE = $SOURCE"
        echo "        PROJECT = $PROJECT"
        echo "        NAME = $NAME"

        # Run the YOLOv5 form detection model
        echo "        Running YOLOv5 detection (form)..."
        python ./yolov5/detect.py --max-det 1 --save-crop --nosave --save-txt --save-conf --exist-ok --weights "$WEIGHTS" --source "$SOURCE" --project "$PROJECT" --name "$NAME"

        if [[ $? -ne 0 ]]; then
            echo "        Error: YOLOv5 detection failed for PAGE=$PAGE (form)"
            exit 1
        fi

        # Variables for detect_fields
        WEIGHTS="$DETECT_FIELDS_PATH/best.pt"
        SOURCE="$PROJECT/$PAGE/crops/form"
        PROJECT="./runs/$INPUT_FOLDER/intermediate/is_empty_field_preds"
        NAME="$PAGE"

        echo "        WEIGHTS = $WEIGHTS"
        echo "        SOURCE = $SOURCE"
        echo "        PROJECT = $PROJECT"
        echo "        NAME = $NAME"

        # Run the YOLOv5 field detection model
        echo "        Running YOLOv5 detection (fields)..."
        # 
        python ./yolov5/detect.py --save-txt --iou-thres 0.0 --save-conf --nosave --exist-ok --weights "$WEIGHTS" --source "$SOURCE" --project "$PROJECT" --name "$NAME"

        if [[ $? -ne 0 ]]; then
            echo "        Error: YOLOv5 detection failed for PAGE=$PAGE (fields)"
            exit 1
        fi

        # Variables for adj_is_empty_field_preds.py
        INPUT_DIR="$PROJECT/$PAGE/labels"
        OUTPUT_DIR="./runs/$INPUT_FOLDER/intermediate/adj_is_empty_field_preds/$PAGE/labels"
        ANNOTATION_YAML="$DETECT_FIELDS_PATH/annotations.yaml"

        echo "        INPUT_DIR = $INPUT_DIR"
        echo "        OUTPUT_DIR = $OUTPUT_DIR"
        echo "        ANNOTATION_YAML = $ANNOTATION_YAML"

        # Run the adjustment script
        echo "        Running adj_is_empty_field_preds.py..."
        python ./scripts/adj_is_empty_field_preds.py --input-dir "$INPUT_DIR" --output-dir "$OUTPUT_DIR" --annotation-yaml "$ANNOTATION_YAML"

        if [[ $? -ne 0 ]]; then
            echo "        Error: Adjustment script failed for PAGE=$PAGE"
            exit 1
        fi

        # Variables file_name_agg_adj_is_empty_field_preds.py
        OUTPUT_CSV="./runs/$INPUT_FOLDER/intermediate/run_agg_is_empty_results.csv"

        echo "        INPUT_DIR = $INPUT_DIR"
        echo "        OUTPUT_CSV = $OUTPUT_CSV"
        echo "        ANNOTATION_YAML = $ANNOTATION_YAML"

        # Run the aggregation script
        echo "        Running file_name_agg_adj_is_empty_field_preds.py..."
        python ./scripts/file_name_agg_adj_is_empty_field_preds.py --input-dir "$INPUT_DIR" --output-csv "$OUTPUT_CSV" --annotation-yaml "$ANNOTATION_YAML"

        if [[ $? -ne 0 ]]; then
            echo "        Error: Adjustment script failed for PAGE=$PAGE"
            exit 1
        fi

        # Variables for make_bbox_is_empty_imgs.py
        INPUT_IMAGES_DIR="./runs/$INPUT_FOLDER/intermediate/crop_form_imgs/$PAGE/crops/form"
        INPUT_LABELS_DIR="$OUTPUT_DIR"
        OUTPUT_DIR="./runs/$INPUT_FOLDER/intermediate/bbox_is_empty_imgs/$PAGE"

        echo "        INPUT_IMAGES_DIR = $INPUT_IMAGES_DIR"
        echo "        INPUT_LABELS_DIR = $INPUT_LABELS_DIR"
        echo "        OUTPUT_DIR = $OUTPUT_DIR"

        # Run the image generation script
        echo "        Running make_bbox_is_empty_imgs.py..."
        python ./scripts/make_bbox_is_empty_imgs.py --input-images-dir "$INPUT_IMAGES_DIR" --input-labels-dir "$INPUT_LABELS_DIR" --output-dir "$OUTPUT_DIR" --annotation-yaml "$ANNOTATION_YAML"

        if [[ $? -ne 0 ]]; then
            echo "        Error: Image generation script failed for PAGE=$PAGE"
            exit 1
        fi

        echo "    $DETECT_FIELDS_PATH"
    fi
done


# Run file_name_agg_bbox_is_empty_imgs.py once after the loop
INPUT_DIR="./runs/$INPUT_FOLDER/intermediate/bbox_is_empty_imgs"
OUTPUT_DIR="./runs/$INPUT_FOLDER/intermediate/file_name_agg_bbox_is_empty_imgs"

echo "Running file_name_agg_bbox_is_empty_imgs.py..."
python ./scripts/file_name_agg_bbox_is_empty_imgs.py --input-dir "$INPUT_DIR" --output-dir "$OUTPUT_DIR"

if [[ $? -ne 0 ]]; then
    echo "Error: file_name_agg_bbox_is_empty_imgs.py failed"
    exit 1
fi

# Move results to final output directory
FINAL_DIR="./runs/$INPUT_FOLDER/final"
mkdir -p "$FINAL_DIR"

# Move and rename final directory
mv "./runs/$INPUT_FOLDER/intermediate/file_name_agg_bbox_is_empty_imgs" "$FINAL_DIR/pdfs_with_is_missing_bbox"

# Move CSV results
mv "./runs/$INPUT_FOLDER/intermediate/run_agg_is_empty_results.csv" "$FINAL_DIR/"

echo "Final results moved to $FINAL_DIR"