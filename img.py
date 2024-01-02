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
        return None, None
    else:
        black_peak = peaks[0]
        white_peak = peaks[-1]
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

def adjust_contrast_peaks(img, output_path, text_black_crop_percent, text_white_crop_percent, analysis_area_percent, peak_prominence=500, secondary_peak_ratio=0.2, debug=False):
    # Analyze the center crop for histogram adjustments
    analysis_img = get_center_crop(img, analysis_area_percent) if analysis_area_percent < 100 else img
    hist = cv2.calcHist([analysis_img], [0], None, [256], [0, 256]).ravel()

    # Find peaks in the histogram
    peaks, _ = find_peaks(hist, peak_prominence)
    peak_values = hist[peaks]

    # Identify the main peak and check for significant secondary peak
    main_peak_value = np.max(peak_values)
    main_peak = peaks[np.argmax(peak_values)]

    lower_bound, upper_bound = get_black_white_peaks(peaks)
    if lower_bound is None:
        lower_bound = np.percentile(analysis_img, 1)
    if upper_bound is None:
        upper_bound = np.percentile(analysis_img, 99)

    # Filter out peaks less than the specified ratio of the main peak
    significant_peaks = peak_values[peak_values >= secondary_peak_ratio * main_peak_value]
    has_significant_secondary_peak = len(significant_peaks) > 1
    if not has_significant_secondary_peak:
        print("  Has significant secondary peak")
        lower_bound = upper_bound * (text_black_crop_percent/100.0) # move it right a bit
        upper_bound = (upper_bound - lower_bound) * (1 - text_white_crop_percent/100.0) + lower_bound # move it left a bit

    print(f"    {lower_bound}..{upper_bound}")

    # Apply contrast adjustments to the entire image
    adjusted_image = np.clip((img_channel_full - lower_bound) * 255 / (upper_bound - lower_bound), 0, 255)

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

def adjust_contrast_stddev(image, num_std=2, channel='green', pixel_percentage=None):
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
    total_pixels = sum(hist)

    # Set pixel threshold based on the percentage or use a default value
    pixel_threshold = total_pixels * pixel_percentage if pixel_percentage is not None else 1

    # Refine bounds based on actual data in the histogram
    lower_bound, upper_bound = refine_contrast_bounds(hist, total_pixels, lower_bound, upper_bound, pixel_threshold)

    # Apply contrast stretching
    adjusted_img = np.clip((img_channel - lower_bound) * 255 / (upper_bound - lower_bound), 0, 255)

    return adjusted_img.astype(np.uint8)

# autocrop

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

# rotate

def rotate(img)
    img.rotate(rotation_angle, expand=True)

# split

def split(img):
    width, height = img.size
    left = img.crop((0, 0, width/2, height))
    right = img.crop((width/2, 0, width, height))
    return left, right

