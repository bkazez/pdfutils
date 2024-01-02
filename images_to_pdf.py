"""
Usage:
python images_to_pdf.py --input_folder <input_folder> --output_folder <output_folder> [--pages_per_pdf <num>] [--split] [--rotation_angle <angle>] [--prefix_regex <regex>]

This script combines images into PDF files. Supported image formats include PNG, JPG, JPEG, TIF, and TIFF. It can optionally split images into left and right pages and rotate them.

Options:
- --input_folder: Directory containing the images to be combined into PDFs.
- --output_folder: Directory where the resulting PDF files will be saved.
- --pages_per_pdf: Optional. Number of images to include in each PDF file.
- --split: Optional. Split images into left and right pages.
- --rotation_angle: Optional. Rotate pages by a given angle (e.g., 180 for upside-down).
- --prefix_regex: Optional. Regex pattern to remove from the input folder name in the PDF filename.
"""

import argparse
from PIL import Image
import os
import re

def is_image_file(filename):
    supported_formats = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    return filename.lower().endswith(supported_formats)

def numerical_sort_key(filename):
    numbers = map(int, re.findall(r'\d+', filename))
    return tuple(numbers)

def split_image(img):
    width, height = img.size
    left = img.crop((0, 0, width/2, height))
    right = img.crop((width/2, 0, width, height))
    return left, right

def combine_images_to_pdf(args):
    input_folder = args.input_folder
    output_folder = args.output_folder
    pages_per_pdf = args.pages_per_pdf
    split = args.split
    rotation_angle = args.rotation_angle
    prefix_regex = args.prefix_regex

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    images = sorted([img for img in os.listdir(input_folder) if is_image_file(img)], key=numerical_sort_key)

    if pages_per_pdf is None:
        pages_per_pdf = len(images)

    for i in range(0, len(images), pages_per_pdf):
        imgs = []
        for img in images[i:i + pages_per_pdf]:
            image_path = os.path.join(input_folder, img)
            with Image.open(image_path).convert('RGB') as im:
                if rotation_angle:
                    im = im.rotate(rotation_angle, expand=True)
                if split:
                    imgs.extend(split_image(im))
                else:
                    imgs.append(im)

        output_name = os.path.basename(input_folder)
        if prefix_regex:
            output_name = re.sub(prefix_regex, '', output_name)
        pdf_filename = f'{output_name}_{i // pages_per_pdf + 1}.pdf' if pages_per_pdf != len(images) else f'{output_name}.pdf'
        imgs[0].save(os.path.join(output_folder, pdf_filename), save_all=True, append_images=imgs[1:])

def main():
    parser = argparse.ArgumentParser(description='Combine images into PDF files.')
    parser.add_argument('--input_folder', required=True, help='Directory containing the images.')
    parser.add_argument('--output_folder', required=True, help='Directory to save PDF files.')
    parser.add_argument('--pages_per_pdf', type=int, help='Number of images per PDF file.')
    parser.add_argument('--split', action='store_true', help='Split images into left and right pages.')
    parser.add_argument('--rotation_angle', type=int, help='Rotate pages by this angle.')
    parser.add_argument('--prefix_regex', type=str, help='Regex to remove from PDF filename.')

    args = parser.parse_args()
    combine_images_to_pdf(args)

if __name__ == '__main__':
    main()
