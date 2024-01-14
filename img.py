import cv2
import numpy as np
import argparse
import os
from scipy.signal import savgol_filter, find_peaks

# is_image_file

def is_image_file(filename):
    supported_formats = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    return filename.lower().endswith(supported_formats)

# adjust_contrast_peaks

def apply_unsharp_mask(image, kernel_size, sharpening_factor):
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return cv2.addWeighted(image, 1 + sharpening_factor, blurred, -sharpening_factor, 0)

def get_center_crop(image, percent):
    if percent >= 100:
        return image
    height, width = image.shape[:2]
    x_start = int(width * (1 - percent / 100) / 2)
    x_end = width - x_start
    y_start = int(height * (1 - percent / 100) / 2)
    y_end = height - y_start
    return image[y_start:y_end, x_start:x_end]

def get_histogram_peak(hist):
    max_density = np.max(hist)
    peak = np.argmax(hist)
    return peak

def get_black_white_peaks(peaks):
    if len(peaks) < 2:
        if len(peaks) == 1:
            return peaks[0]
        else:
            return None, None
    else:
        black_peak = peaks[0]
        white_peak = peaks[-1]
        print(f"{black_peak}, {white_peak}")
        return float(black_peak), float(white_peak)

def smooth_histogram(hist, kernel_size=5):
    """
    Smooth the histogram using a moving average (convolution with a uniform kernel).

    Args:
    - hist: The histogram array.
    - kernel_size: The size of the convolution kernel (window size for moving average).
    """
    kernel = np.ones(kernel_size) / kernel_size
    hist_smooth = np.convolve(hist, kernel, mode='same')
    return hist_smooth

def save_histogram(hist, peaks, lower_bound, upper_bound, output_path):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(hist, color='green')
    #plt.plot(hist_smooth, color='lightgreen')
    plt.plot(peaks, hist[peaks], 'x')
    plt.axvline(x=lower_bound, color='black', linestyle='--')
    plt.axvline(x=upper_bound, color='gray', linestyle='--')
    plt.title('Histogram with Black and White Points')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')

    hist_output_path = os.path.splitext(output_path)[0] + '_histogram.png'
    plt.savefig(hist_output_path)
    plt.close()

def adjust_contrast_peaks(img, text_black_crop_percent, text_white_crop_percent, analysis_area_percent, peak_prominence=500, secondary_peak_ratio=0.2, debug=False):
    if img.dtype != 'uint8':
        img = cv2.convertScaleAbs(img)

    # Convert to grayscale if it is a multi-channel image
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Analyze the center crop for histogram adjustments
    analysis_img = get_center_crop(img, analysis_area_percent) if analysis_area_percent < 100 else img

    hist = cv2.calcHist([analysis_img], [0], None, [256], [0, 256]).ravel()

    # Find peaks in the histogram
    peaks, _ = find_peaks(hist, peak_prominence)
    if len(peaks) == 0:
        print("***No peaks found in histogram.")
        return img
    peak_values = hist[peaks]

    if len(peaks) == 1:
        print("Only one peak")
        lower_bound = float(0.0)
        upper_bound = float(peaks[0]) * (1 - text_white_crop_percent/100.0)
        print(f"    => {lower_bound}..{upper_bound}")
        #save_histogram(hist, peaks, lower_bound, upper_bound, "img")
    else:
        lower_bound = float(peaks[0])
        upper_bound = float(peaks[-1])

        # Search for significant secondary peak, which is probably a grayscale image
        # Without a significant secondary peak is probably text-based and benefits from more aggressive histogram cropping
        main_peak_value = np.max(peak_values)
        significant_peaks = peak_values[peak_values >= secondary_peak_ratio * main_peak_value]
        has_significant_secondary_peak = len(significant_peaks) > 1
        if not has_significant_secondary_peak:
            print("  No significant secondary peak")
            print(f"    {lower_bound}..{upper_bound}")
            width = upper_bound - lower_bound
            lower_bound += width * text_black_crop_percent/100.0
            upper_bound -= width * text_white_crop_percent/100.0
            print(f"    => {lower_bound}..{upper_bound}")

    # Apply contrast adjustments to the entire image
    adjusted_image = np.clip((img - lower_bound) * 255 / (upper_bound - lower_bound), 0, 255)

    return adjusted_image

# adjust_contrast_stddev

def refine_contrast_bounds(hist, total_pixels, low_val, high_val, pixel_threshold):
    # Refine lower bound
    while low_val < high_val and hist[int(low_val)] < pixel_threshold:
        low_val += 1

    # Refine upper bound
    while high_val > low_val and hist[int(high_val)] < pixel_threshold:
        high_val -= 1

    return low_val, high_val

def adjust_contrast_stddev(img, num_std=2, pixel_percentage=None):
    median = np.median(img)
    std = np.std(img_channel)

    lower_bound = max(median - num_std * std, 0)
    upper_bound = min(median + num_std * std, 255)

    # Calculate histogram
    if img.dtype != 'uint8':
        img = cv2.convertScaleAbs(img)
    hist = cv2.calcHist([img_channel], [0], None, [256], [0, 256]).ravel()
    total_pixels = sum(hist)

    # Set pixel threshold based on the percentage or use a default value
    pixel_threshold = total_pixels * pixel_percentage if pixel_percentage is not None else 1

    # Refine bounds based on actual data in the histogram
    lower_bound, upper_bound = refine_contrast_bounds(hist, total_pixels, lower_bound, upper_bound, pixel_threshold)

    # Apply contrast stretching
    img = np.clip((img - lower_bound) * 255 / (upper_bound - lower_bound), 0, 255)

    return img.astype(np.uint8)

# autocrop

def find_border_contour(image, threshold=150):
    # Check whether the image is already grayscale
    if len(image.shape) == 2 or image.shape[2] == 1:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.uint8)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour

def draw_contour(image, contour):
    cv2.drawContours(image, [contour], -1, (0, 0, 255), 2)
    return image

def autocrop(image, threshold, contraction_percent, debug=False):
    border_contour = find_border_contour(image, threshold)
    if border_contour is None:
        print("***No contour found. Returning original image.")
        return image

    # Calculate bounding rectangle and contract it
    x, y, w, h = cv2.boundingRect(border_contour)
    contraction = contraction_percent / 100.0
    new_width = int(w * (1 - contraction))
    new_height = int(h * (1 - contraction))
    new_x = x + int((w - new_width) / 2)
    new_y = y + int((h - new_height) / 2)

    # Debug mode: write the contour and rectangle on the image
    if debug:
        import uuid

        image = image.copy()

        # Convert the image to 8-bit if it's not
        if image.dtype == 'float64':  # Check if image depth is 64F
            image_8bit = cv2.convertScaleAbs(image)
        else:
            image_8bit = image

        # Convert grayscale image to BGR color image for contour drawing
        if len(image_8bit.shape) == 2:  # Grayscale image has 2 dimensions
            image_color = cv2.cvtColor(image_8bit, cv2.COLOR_GRAY2BGR)
        else:
            image_color = image_8bit.copy()

        cv2.drawContours(image_color, [border_contour], -1, (0, 255, 0), 3)  # Green contour
        cv2.rectangle(image_color, (new_x, new_y), (new_x + new_width, new_y + new_height), (0, 0, 255), 3)  # red rectangle
        debug_filename = os.path.expanduser(f'~/Desktop/{str(uuid.uuid4())}.png')
        cv2.imwrite(debug_filename, image_color)

    # Check if the resulting crop would be < 0.5 * original_area
    original_area = image.shape[0] * image.shape[1]
    cropped_area = new_width * new_height
    if cropped_area < 0.5 * original_area:
        print(f"***Cropped image would be only {new_width}x{new_height}. Skipping crop.")
        return image

    # Crop and return the image
    cropped_image = image[new_y:new_y+new_height, new_x:new_x+new_width]
    if cropped_image.size == 0:
        print("***Cropping resulted in an empty image. Returning original image.")
        return image

    return cropped_image

# rotate

def rotate(img, angle):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)

    # Get the rotation matrix and then apply the affine warp
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated

# split

def split(img):
    height, width = img.shape[:2]
    left = img[:, 0:width//2]
    right = img[:, width//2:width]
    return left, right
