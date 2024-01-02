import cv2
import numpy as np
import argparse
import os

def find_border_contour(image, threshold_value=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour

def draw_contour(image, contour):
    cv2.drawContours(image, [contour], -1, (0, 0, 255), 2)
    return image

def autocrop_image(image, threshold_value, contraction_percent, debug=False):
    border_contour = find_border_contour(image, threshold_value)
    image_with_contour = draw_contour(image.copy(), border_contour) if debug else image.copy()

    # Bounding rectangle
    x_bound, y_bound, w_bound, h_bound = cv2.boundingRect(border_contour)

    # Check if crop area is less than 50% of the original image
    original_area = image.shape[0] * image.shape[1]
    cropped_area = w_bound * h_bound
    if cropped_area < 0.5 * original_area:
        print(f"***Cropped image would be only {w_bound}x{h_bound}. Skipping crop.")
        return image_with_contour, image  # Return the original image if cropping condition is not met

    # Fit rectangle
    rect = cv2.minAreaRect(border_contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    x_fit, y_fit, w_fit, h_fit = cv2.boundingRect(box)

    # Average of the coordinates and dimensions
    x_avg = (x_bound + x_fit) // 2
    y_avg = (y_bound + y_fit) // 2
    w_avg = (w_bound + w_fit) // 2
    h_avg = (h_bound + h_fit) // 2

    # Apply contraction
    contract_pixels_w = int(w_avg * contraction_percent / 100)
    contract_pixels_h = int(h_avg * contraction_percent / 100)

    # Cropped image
    cropped_image = image[y_avg + contract_pixels_h:y_avg + h_avg - contract_pixels_h,
                          x_avg + contract_pixels_w:x_avg + w_avg - contract_pixels_w]

    # Debug mode: Save the image with contour
    if debug:
        image_with_contour = draw_contour(image.copy(), border_contour)
    else:
        image_with_contour = image.copy()

    return image_with_contour, cropped_image

def process_folder(input_folder, output_folder, threshold_value, contraction_percent, debug):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.   path.join(output_folder, filename)

            image = cv2.imread(input_path)
            if image is None:
                print(f"Could not open {input_path}. Skipping.")
                continue

            image_with_contour, processed_image = autocrop_image(image, threshold_value, contraction_percent, debug)

            if debug:
                contour_output_path = os.path.splitext(output_path)[0] + '_contoured.png'
                cv2.imwrite(contour_output_path, image_with_contour)
                print(f"Contoured image saved to {contour_output_path}")

            cv2.imwrite(output_path, processed_image)
            print(f"Processed {filename}")



def main():
    parser = argparse.ArgumentParser(description="Auto-crop borders in images within a folder.")
    parser.add_argument("input_folder", help="Directory containing the images to be processed.")
    parser.add_argument("output_folder", help="Directory where the processed images will be saved.")
    parser.add_argument("--threshold", type=int, default=150, help="Threshold value for border detection.")
    parser.add_argument("--contract", type=float, default=1, help="Percentage to contract the crop area.")
    parser.add_argument("--debug", action='store_true', help="Enable debug mode.")

    args = parser.parse_args()

    process_folder(args.input_folder, args.output_folder, args.threshold, args.contract, args.debug)

if __name__ == "__main__":
    main()
