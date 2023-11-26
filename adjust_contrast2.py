import cv2
import numpy as np
import sys
import os

def refine_contrast_bounds(hist, low_val, high_val):
    # Refine lower bound
    while low_val < high_val and hist[int(low_val)] == 0:
        low_val += 1

    # Refine upper bound
    while high_val > low_val and hist[int(high_val)] == 0:
        high_val -= 1

    return low_val, high_val

def adjust_contrast_using_median(image_path, num_std=2, channel='green'):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Unable to open the image file.")

    # Select the specified color channel
    if channel == 'green':
        img_channel = img[:, :, 1]  # Green channel
    elif channel == 'red':
        img_channel = img[:, :, 2]  # Red channel
    elif channel == 'blue':
        img_channel = img[:, :, 0]  # Blue channel
    else:
        img_channel = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    median = np.median(img_channel)
    std = np.std(img_channel)

    lower_bound = max(median - num_std * std, 0)
    upper_bound = min(median + num_std * std, 255)

    # Calculate histogram
    hist = cv2.calcHist([img_channel], [0], None, [256], [0, 256]).ravel()

    # Refine bounds based on actual data in the histogram
    lower_bound, upper_bound = refine_contrast_bounds(hist, lower_bound, upper_bound)

    # Apply contrast stretching
    adjusted_img = np.clip((img_channel - lower_bound) * 255 / (upper_bound - lower_bound), 0, 255)

    return adjusted_img.astype(np.uint8)

def process_images(input_folder, output_folder, num_std=2, channel='green'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for img_file in os.listdir(input_folder):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, img_file)
            output_path = os.path.join(output_folder, img_file)

            adjusted_image = adjust_contrast_using_median(input_path, num_std, channel)
            cv2.imwrite(output_path, adjusted_image)
            print(f"Processed: {img_file}")

def main():
    if len(sys.argv) != 5:
        print("Usage: python adjust_contrast.py <input_folder> <output_folder> <num_std> <channel>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    num_std = float(sys.argv[3])
    channel = sys.argv[4]
    process_images(input_folder, output_folder, num_std, channel)

if __name__ == '__main__':
    main()
