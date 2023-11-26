import os
import sys
from PyPDF2 import PdfFileReader, PdfFileWriter
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def convert_pdf_to_images(pdf_path, output_folder):
    from pdf2image import convert_from_path
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    return convert_from_path(pdf_path, output_folder=output_folder, fmt='png')

def find_histogram_bounds(image_path, lower_percentile, upper_percentile):
    img = Image.open(image_path).convert('L')
    hist = img.histogram()

    # Normalize the histogram
    hist_norm = [float(h) / sum(hist) for h in hist]

    # Cumulative distribution
    cum_dist = np.cumsum(hist_norm)

    # Find lower and upper bounds based on percentiles
    lower_bound = np.percentile(range(256), lower_percentile, interpolation='nearest', weights=cum_dist)
    upper_bound = np.percentile(range(256), upper_percentile, interpolation='nearest', weights=cum_dist)

    return lower_bound, upper_bound

def adjust_contrast(image_path, lower_bound, upper_bound):
    img = Image.open(image_path)
    img = img.point(lambda p: p * (255 / (upper_bound - lower_bound)) + lower_bound)
    return img

def main(pdf_path, pages_per_pdf):
    images_folder = 'temp_images'
    processed_images_folder = 'processed_images'

    # Convert PDF to images
    images = convert_pdf_to_images(pdf_path, images_folder)

    # Process each image
    if not os.path.exists(processed_images_folder):
        os.makedirs(processed_images_folder)

    for i, image_path in enumerate(images):
        lower_bound, upper_bound = find_histogram_bounds(image_path, 0.05, 0.95)
        img = adjust_contrast(image_path, lower_bound, upper_bound)
        img.save(f'{processed_images_folder}/page_{i:04d}.png')

    # Combine processed images into PDFs
    for i in range(0, len(images), pages_per_pdf):
        img_paths = [f'{processed_images_folder}/page_{j:04d}.png' for j in range(i, min(i + pages_per_pdf, len(images)))]
        imgs = [Image.open(x) for x in img_paths]
        imgs[0].save(f'output_{i // pages_per_pdf + 1}.pdf', save_all=True, append_images=imgs[1:])

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python script.py <path_to_pdf> <pages_per_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    pages_per_pdf = int(sys.argv[2])
    main(pdf_path, pages_per_pdf)
