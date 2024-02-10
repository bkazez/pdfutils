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

def save_histogram(output_path, hist, peaks, lower_bound, upper_bound):
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

def adjust_contrast_peaks(img, text_black_crop_percent=37, text_white_crop_percent=11, analysis_area_percent=80, peak_prominence=25000, min_distance_between_peaks=25, secondary_peak_ratio=0.05, maintain_color=False, min_histogram_width=40, debug=False, output_path=None):
    if img.dtype != 'uint8':
        img = cv2.convertScaleAbs(img)

    if maintain_color:
        original_img = img.copy()
    else:
        original_img = None

    # Convert to grayscale if required, otherwise work on L channel of LAB
    if maintain_color:
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0]
    else:
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Analyze the center crop for histogram adjustments
    analysis_img = get_center_crop(img, analysis_area_percent) if analysis_area_percent < 100 else img

    hist = cv2.calcHist([analysis_img], [0], None, [256], [0, 256]).ravel()

    # Find peaks in the histogram
    peaks, _ = find_peaks(hist, prominence=peak_prominence, distance=min_distance_between_peaks)
    if len(peaks) == 0:
        print("***No peaks found in histogram.")
        return img
    peak_values = hist[peaks]

    if len(peaks) == 1:
        # Text
        print("  Only one peak")
        lower_bound = 60 # start a quarter of the way in
        upper_bound = float(peaks[0])

        # Crop more aggressively, based on the width
        width = upper_bound - lower_bound
        if width > min_histogram_width:
            lower_bound += width * text_black_crop_percent/100.0
            upper_bound -= width * text_white_crop_percent/100.0
    else:
        # Photo
        lower_bound = 60 # start a quarter of the way in
        upper_bound = np.percentile(analysis_img, 97)

    if (upper_bound - lower_bound) < min_histogram_width:
        # Crop based on percentiles
        print(f"  too narrow: [{lower_bound}, {upper_bound}]")
        lower_bound = np.percentile(analysis_img, 3)
        upper_bound = np.percentile(analysis_img, 97)
        print(f"    recalculated: [{lower_bound}, {upper_bound}]")

    if debug and output_path:
        save_histogram(output_path, hist, peaks, lower_bound, upper_bound)

    # Apply contrast adjustments
    if maintain_color:
        l_channel = np.clip((img.astype(float) - lower_bound) * 255 / (upper_bound - lower_bound), 0, 255).astype('uint8')
        lab_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2LAB)
        lab_img[..., 0] = l_channel
        adjusted_image = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)
    else:
        adjusted_image = np.clip((img.astype(float) - lower_bound) * 255 / (upper_bound - lower_bound), 0, 255).astype('uint8')

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

# adjust_contrast_percentile

def adjust_contrast_percentile(image, lower_percentile, upper_percentile):
    """
    Adjust the contrast of an image based on the lower and upper percentiles in the LAB color space.
    Ensures that the lower percentile is less than the upper percentile for effective contrast stretching.
    """
    assert 0 <= lower_percentile < upper_percentile <= 100, "Percentiles must be in the range [0, 100] with lower < upper."

    # Convert the image to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Extract the L channel
    l_channel = lab[:, :, 0]

    # Compute the lower and upper bounds based on percentiles
    lower_bound = np.percentile(l_channel, lower_percentile)
    upper_bound = np.percentile(l_channel, upper_percentile)

    print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")  # Debugging print

    # Stretch the contrast based on the calculated bounds
    l_channel_stretched = np.clip((l_channel - lower_bound) * 255 / (upper_bound - lower_bound), 0, 255).astype(np.uint8)

    # Update the L channel in the LAB image
    lab[:, :, 0] = l_channel_stretched

    # Convert the LAB image back to BGR color space
    adjusted_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return adjusted_image

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

def autocrop(image, output_path, threshold, contraction_percent, debug=False):
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

# color correction

def match_color(target_img, reference_img):
    # Convert both images to LAB color space to separate luminance and color channels
    target_lab = cv2.cvtColor(target_img, cv2.COLOR_BGR2LAB)
    reference_lab = cv2.cvtColor(reference_img, cv2.COLOR_BGR2LAB)

    # Compute the mean and std dev for the color channels (A and B) of both images
    mean_tar, stddev_tar = cv2.meanStdDev(target_lab[:,:,1:3])
    mean_ref, stddev_ref = cv2.meanStdDev(reference_lab[:,:,1:3])

    # Adjust the A and B channels of the target image based on the reference image
    for i in range(1, 3): # Loop through A and B channels
        target_lab[:,:,i] = np.clip(((target_lab[:,:,i] - mean_tar[i-1]) * (stddev_ref[i-1] / stddev_tar[i-1])) + mean_ref[i-1], 0, 255).astype(np.uint8)

    # Convert the LAB image back to BGR color space
    corrected_img = cv2.cvtColor(target_lab, cv2.COLOR_LAB2BGR)

    return corrected_img

# contrast matching

def match_contrast(image, reference_img):
    """
    Match the contrast of an image to that of a reference image using the LAB color space.

    Parameters:
    - image: The target image to adjust, in BGR color space.
    - reference_img: The reference image to match, in BGR color space.

    Returns:
    - The image with its contrast matched to that of the reference image, in BGR color space.
    """
    # Convert both images to LAB color space and extract the L channel
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    reference_lab = cv2.cvtColor(reference_img, cv2.COLOR_BGR2LAB)
    l_image, a_image, b_image = cv2.split(image_lab)
    l_reference, _, _ = cv2.split(reference_lab)

    # Calculate the histogram and CDF of the L channel of both images
    hist_image, bins = np.histogram(l_image.flatten(), 256, [0, 256])
    cdf_image = hist_image.cumsum()
    cdf_image_normalized = cdf_image * (np.max(l_reference) / cdf_image.max())

    hist_reference, bins = np.histogram(l_reference.flatten(), 256, [0, 256])
    cdf_reference = hist_reference.cumsum()
    cdf_reference_normalized = cdf_reference / cdf_reference.max()

    # Create a lookup table to map pixel values based on the CDFs
    lookup_table = np.zeros(256)
    for i in range(256):
        idx = np.abs(cdf_image_normalized[i] - cdf_reference_normalized).argmin()
        lookup_table[i] = idx

    # Map the L channel of the target image using the lookup table
    l_image_matched = lookup_table[l_image]

    # Merge the adjusted L channel back and convert to BGR
    image_matched_lab = cv2.merge([l_image_matched.astype(np.uint8), a_image, b_image])
    image_matched = cv2.cvtColor(image_matched_lab, cv2.COLOR_LAB2BGR)

    return image_matched

def adjust_levels_in_lab(image, black_point, white_point, output_min=0, output_max=255):
    """
    Adjust the levels of the L channel in the LAB color space.

    Parameters:
    - image: Input image in BGR color space.
    - black_point: The lower bound of the input levels to be mapped to output_min.
    - white_point: The upper bound of the input levels to be mapped to output_max.
    - output_min: The lower bound of the output levels (default is 0).
    - output_max: The upper bound of the output levels (default is 255).

    Returns:
    - The image with adjusted L channel, in BGR color space.
    """
    # Convert the image to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Split the LAB image to different channels
    l, a, b = cv2.split(lab)

    # Apply levels adjustment to the L channel
    l_float = l.astype(np.float32)
    l_adjusted = np.clip((l_float - black_point) * (output_max - output_min) / (white_point - black_point) + output_min, output_min, output_max)

    # Merge the adjusted L channel back with the a and b channels
    l_adjusted = l_adjusted.astype(np.uint8)
    updated_lab = cv2.merge((l_adjusted, a, b))

    # Convert the adjusted LAB image back to BGR color space
    adjusted_image = cv2.cvtColor(updated_lab, cv2.COLOR_LAB2BGR)

    return adjusted_image

def adjust_hue_saturation(image, hue, saturation):
    """
    Adjust the hue and saturation of an image, with hue wrapping around.

    Parameters:
    - image: Input image in BGR color space.
    - hue: Hue adjustment value (-179 to 179 in OpenCV). Positive values increase hue; negative values decrease hue.
    - saturation: Saturation adjustment value (-255 to 255 in OpenCV). Positive values increase saturation; negative values decrease saturation.

    Returns:
    - The image with adjusted hue and saturation, in BGR color space.
    """
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Adjust hue with wrapping
    hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + hue) % 180

    # Adjust saturation with clamping
    hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(int) + saturation, 0, 255)

    # Convert the adjusted HSV image back to BGR color space
    adjusted_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return adjusted_image

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

# scale

import cv2

def scale(image, scale_percent):
    # Calculate the new dimensions
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    new_dim = (width, height)

    # Resize the image
    resized_image = cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)

    return resized_image
