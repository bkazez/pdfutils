"""
Usage:
python pdf_to_images.py <path_to_pdf> <output_folder> <dpi>

Example:
python pdf_to_images.py document.pdf images_output 300

- <path_to_pdf>: Path to the PDF file to be converted.
- <output_folder>: Directory where the converted images will be saved.
- <dpi>: Dots Per Inch (resolution) for the converted images.

This script converts each page of a PDF file into separate image files. The images are stored in the specified output folder with a resolution determined by the dpi parameter.
"""

import sys
from pdf2image import convert_from_path
import os

def convert_pdf_to_images(pdf_path, output_folder, dpi=300):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    convert_from_path(pdf_path, dpi=dpi, output_folder=output_folder, fmt='png')

def main():
    if len(sys.argv) != 4:
        print("Usage: python pdf_to_images.py <path_to_pdf> <output_folder> <dpi>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_folder = sys.argv[2]
    dpi = int(sys.argv[3])
    convert_pdf_to_images(pdf_path, output_folder, dpi)

if __name__ == '__main__':
    main()
