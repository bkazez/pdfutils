from PIL import Image
import os

def images_to_pdf(input_folder, output_folder, output_name, pages_per_pdf=None):
    images = sorted([img for img in os.listdir(input_folder) if img.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))], key=lambda x: x.lower())

    if not pages_per_pdf:
        pages_per_pdf = len(images)

    # Handle the case when there are no images or pages_per_pdf is None/zero
    if not images:
        print("*** No images to process!")
        return

    pages_per_pdf = min(pages_per_pdf, len(images))  # Ensure pages_per_pdf doesn't exceed the number of images

    for i in range(0, len(images), pages_per_pdf):
        imgs = []
        for img in images[i:i + pages_per_pdf]:
            image_path = os.path.join(input_folder, img)
            with Image.open(image_path).convert('RGB') as im:
                imgs.append(im)

        # Format the output file name
        output_pdf_path = os.path.join(output_folder, f'{output_name}.pdf') if pages_per_pdf >= len(images) else os.path.join(output_folder, f'{output_name}_{i // pages_per_pdf + 1}.pdf')

        # Saving the PDF only if there are images
        if imgs:
            imgs[0].save(output_pdf_path, save_all=True, append_images=imgs[1:])

# Example usage
# images_to_pdf('path/to/input_folder', 'path/to/output_folder', 'output_name', pages_per_pdf=None)
