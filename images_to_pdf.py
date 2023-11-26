"""
Usage:
python images_to_pdf.py <input_folder> <output_folder> <pages_per_pdf>

Example:
python images_to_pdf.py images_input images_output 5

- <input_folder>: Directory containing the images to be combined into PDFs.
- <output_folder>: Directory where the resulting PDF files will be saved.
- <pages_per_pdf>: Number of images to include in each PDF file.

This script combines images into PDF files. Each PDF file contains the specified number of images, sorted by filename. Supported image formats include PNG, JPG, JPEG, TIF, and TIFF.
"""

import sys
from PIL import Image
import os
import re

def numerical_sort_key(filename):
    """
    Extract numerical values from filename for sorting.
    """
    numbers = map(int, re.findall(r'\d+', filename))
    return tuple(numbers)

def is_image_file(filename):
    # List of supported image formats
    supported_formats = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    return filename.lower().endswith(supported_formats)

def combine_images_to_pdf(input_folder, output_folder, pages_per_pdf):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Filter and sort image files numerically
    images = sorted([img for img in os.listdir(input_folder) if is_image_file(img)], key=numerical_sort_key)
    for i in range(0, len(images), pages_per_pdf):
        imgs = [Image.open(os.path.join(input_folder, img)).convert('RGB') for img in images[i:i+pages_per_pdf]]
        imgs[0].save(os.path.join(output_folder, f'output_{i // pages_per_pdf + 1}.pdf'), save_all=True, append_images=imgs[1:])

def main():
    if len(sys.argv) != 4:
        print("Usage: python images_to_pdf.py <input_folder> <output_folder> <pages_per_pdf>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    pages_per_pdf = int(sys.argv[3])
    combine_images_to_pdf(input_folder, output_folder, pages_per_pdf)

if __name__ == '__main__':
    main()
