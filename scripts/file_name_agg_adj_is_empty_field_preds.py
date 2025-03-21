import argparse
import os
from tqdm import tqdm
import polars as pl
import yaml

def main(input_dir, output_csv, annotation_yaml):
	# Ensure input directory exists
	if not os.path.isdir(input_dir):
		raise ValueError(f"Input directory does not exist: {input_dir}")

	# Load YAML file
	with open(annotation_yaml, "r") as file:
		yaml_data = yaml.safe_load(file)

	# Extract class mapping from 'names'
	class_mapping = {int(k): v for k, v in yaml_data["names"].items()}

	df = pl.scan_csv(f"{input_dir}/**/*.txt", 
		include_file_paths='path', 
		separator=' ', 
		has_header=False, 
		new_columns=["class_id", "center_x", "center_y", "width", "height", "confidence"]).collect()

	df = df.with_columns(
		pl.col('path').str.split('/').list.get(-3).alias('page'),
		pl.col('path').str.split('/').list.get(-1).str.replace('.txt', '.pdf').alias('file_name'),
		)
	df = df.drop('path')

	# Ensure class_id is of type Int64 to match dictionary keys
	df = df.with_columns(pl.col("class_id").cast(pl.String))

	# Add the class_string column by mapping class_id to class_mapping
	df = df.with_columns(
		pl.col("class_id").replace(class_mapping).alias("fieldName"),
	)

	# Ensure class_id is of type Int64 to match dictionary keys
	df = df.with_columns(pl.col("class_id").cast(pl.Int64))

	# Sort by file_name and class_id (ascending)
	df = df.sort(["file_name", "class_id"])

	df = df.rename({"fieldName": "is_empty_fieldName"})

	# Create the fieldName column by stripping "is_empty_" or "not_empty_"
	df = df.with_columns(
		pl.col("is_empty_fieldName").str.replace(r"^(is_empty_|not_empty_)", "").alias("fieldName")
	)

	# Add the is_empty column: True if "is_empty_" is in is_empty_fieldName, False otherwise
	df = df.with_columns(
		pl.when(pl.col("is_empty_fieldName").str.contains("is_empty_"))
		.then(pl.lit(True))
		.otherwise(pl.lit(False))
		.alias("is_empty")
	)

	# drop is_empty_fieldName
	df = df.drop('is_empty_fieldName')

	# Ensure class_id is of type Int64 to match dictionary keys
	df = df.with_columns(pl.col("class_id").cast(pl.Int64))

	# Get unique file names
	unique_file_names = df["file_name"].unique()
	df_fields = pl.DataFrame({"fieldName": list(yaml_data["names"].values())})
	df_fields = df_fields.with_columns(pl.col('fieldName').str.replace('is_empty_', '').str.replace('not_empty_', ''))
	df_fields = df_fields.unique()

	# Create a Cartesian product of unique file names and df_fields
	df_file_names = pl.DataFrame({"file_name": unique_file_names})
	df_expanded = df_file_names.join(df_fields, how="cross")

	# Left join df_expanded with df_input on file_name and fieldName
	df = df_expanded.join(
		df, on=["file_name", "fieldName"], how="left"
	)

	# Add the `not_detected` column: True if confidence is missing, False otherwise
	df = df.with_columns(
		pl.col("class_id").is_null().alias("not_detected")
	)

	# Sort by file_name and class_id (ascending)
	df = df.sort(["file_name", "class_id"])


	# Save the updated DataFrame to CSV
	df.write_csv(output_csv)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Process text files.")
	parser.add_argument("--input-dir", required=True, help="Path to the input directory")
	parser.add_argument("--output-csv", required=True, help="Path to the output csv")
	parser.add_argument("--annotation-yaml", required=True, help="Path to the annotation yaml file")
	args = parser.parse_args()

	main(args.input_dir, args.output_csv, args.annotation_yaml)

