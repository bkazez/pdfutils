import os
from pdf2image import convert_from_path
import argparse

def convert_pdf_to_png(input_pdf, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = convert_from_path(input_pdf)

    num_digits = len(str(len(images)))  # Calculate the number of digits for zero-padding
    for i, image in enumerate(images):
        output_file = os.path.join(output_dir, f"page_{str(i).zfill(num_digits)}.png")
        image.save(output_file, 'PNG')
        print(f"Saved: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PDF to PNG")
    parser.add_argument("--input", required=True, help="Input PDF file path")
    parser.add_argument("--output", required=True, help="Output directory for PNG files")
    args = parser.parse_args()

    convert_pdf_to_png(args.input, args.output)
