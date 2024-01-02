import os
import re
from PIL import Image

from .img import is_image_file

def images_to_pdf(input_folder, output_folder, pages_per_pdf=None, prefix_regex=None):
    pages_per_pdf = pages_per_pdf or float('inf')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    images = sorted([img for img in os.listdir(input_folder) if is_image_file(os.path.join(input_folder, img))], key=lambda x: x.lower())

    for i in range(0, len(images), pages_per_pdf):
        imgs = []
        for img in images[i:i+pages_per_pdf]:
            image_path = os.path.join(input_folder, img)
            with Image.open(image_path).convert('RGB') as im:
                imgs.append(im)

        output_name = os.path.basename(input_folder)
        if prefix_regex:
            output_name = re.sub(prefix_regex, '', output_name)
        imgs[0].save(os.path.join(output_folder, f'{output_name}_{i // pages_per_pdf + 1}.pdf'), save_all=True, append_images=imgs[1:])
