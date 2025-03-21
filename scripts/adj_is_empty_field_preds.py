import argparse
import os
from tqdm import tqdm
import polars as pl
import yaml

def main(input_dir, output_dir, annotation_yaml):
    # Ensure input directory exists
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")

    # Load YAML file
    with open(annotation_yaml, "r") as file:
        yaml_data = yaml.safe_load(file)

    # Extract class mapping from 'names'
    class_mapping = {int(k): v for k, v in yaml_data["names"].items()}
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get list of .txt files in input_dir
    txt_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]

    # Progress bar with tqdm
    for filename in tqdm(txt_files, desc="Processing files", position=0, leave=False):
        tqdm.write(f"Processing file: {filename}")
        input_path = os.path.join(input_dir, filename)

        # Read txt file into a Polars DataFrame
        df = pl.read_csv(
            input_path,
            glob=False,
            separator=" ",
            has_header=False,
            new_columns=["class_id", "center_x", "center_y", "width", "height", "confidence"]
        )

        # # Filter out duplicates by keeping only the highest confidence for each class_id
        # df = (
        #     df.sort("confidence", descending=True)
        #     .unique(subset=["class_id"], keep="first")
        # )

        
        ## add class string
        df = df.with_columns(pl.col("class_id").alias("fieldName"))

        # Ensure class_id is of type Int64 to match dictionary keys
        df = df.with_columns(pl.col("fieldName").cast(pl.String))

        # Add the fieldName column by mapping class_id to class_mapping
        df = df.with_columns(
            pl.col("fieldName").replace(class_mapping)
        )

        ## determin field emptiness

        df = df.rename({"fieldName": "is_empty_fieldName"})

        # Create the fieldName column by stripping "is_empty_" or "not_empty_"
        df = df.with_columns(
            pl.col("is_empty_fieldName").str.replace(r"^(is_empty_|not_empty_)", "").alias("fieldName")
        )

        # Remove duplicates: Keep the first occurrence (highest confidence) for each fieldName
        df = df.unique(subset=["fieldName"], keep="first")

        # cleanup

        df = df.select(["class_id", "center_x", "center_y", "width", "height", "confidence"])

        ## save file
        output_path = os.path.join(output_dir, filename)

        df.write_csv(output_path, include_header=False, separator=" ")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process text files.")
    parser.add_argument("--input-dir", required=True, help="Path to the input directory")
    parser.add_argument("--output-dir", required=True, help="Path to the output directory")
    parser.add_argument("--annotation-yaml", required=True, help="Path to the annotation yaml file")
    args = parser.parse_args()
    
    main(args.input_dir, args.output_dir, args.annotation_yaml)
