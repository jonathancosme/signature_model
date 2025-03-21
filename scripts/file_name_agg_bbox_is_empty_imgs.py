import argparse
import os
from pathlib import Path
from PIL import Image
from collections import defaultdict

def images_to_pdf(image_paths, pdf_path):
    """Combines multiple images into a single PDF file."""
    images = [Image.open(path).convert('RGB') for path in sorted(image_paths)]
    if images:
        images[0].save(pdf_path, save_all=True, append_images=images[1:])

def collect_images_by_filename(input_dir):
    """Collects image paths grouped by filename across all subdirectories."""
    input_path = Path(input_dir)
    files_dict = defaultdict(list)

    for subdir in sorted(input_path.iterdir()):
        if subdir.is_dir():
            for image_path in subdir.glob("*.jpg"):
                files_dict[image_path.name].append(image_path)

    return files_dict

def main(input_dir, output_dir):

    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    images_by_file = collect_images_by_filename(input_dir)

    for filename, image_paths in images_by_file.items():
        base_name = os.path.splitext(filename)[0]
        output_pdf_path = os.path.join(output_dir, f"{base_name}.pdf")
        images_to_pdf(image_paths, output_pdf_path)
        print(f"Saved PDF: {output_pdf_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine images of same file across pages into a PDF.")
    parser.add_argument("--input-dir", required=True, help="Path to input directory containing page subfolders.")
    parser.add_argument("--output-dir", required=True, help="Path to output directory to save PDFs.")
    args = parser.parse_args()
  
    main(args.input_dir, args.output_dir)
