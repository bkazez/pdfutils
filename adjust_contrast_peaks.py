
def process_images(args):
    input_folder = args.input_folder
    output_folder = args.output_folder

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for img_file in os.listdir(input_folder):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, img_file)
            output_path = os.path.join(output_folder, os.path.splitext(img_file)[0] + '.png')

            adjusted_image = adjust_contrast(input_path, output_path, args.channel, args.sharpening_factor, args.text_black_crop_percent, args.text_white_crop_percent, args.analysis_area_percent, debug=args.debug)

            print(f"Processed: {img_file}")

def main():
    parser = argparse.ArgumentParser(description='Adjust image contrast with optional unsharp mask.')
    parser.add_argument('--input_folder', required=True, help='Directory containing the images.')
    parser.add_argument('--output_folder', required=True, help='Directory to save adjusted images.')
    parser.add_argument('--channel', choices=['red', 'green', 'blue', 'gray'], required=True, help='Color channel for contrast adjustment.')
    parser.add_argument('--text_black_crop_percent', type=float, help='Where to crop the black point of text docs (percent of histogram until white point)')
    parser.add_argument('--text_white_crop_percent', type=float, help='Where to crop the white point of text docs (percent of histogram from above black point to desired white point)')
    parser.add_argument('--sharpening_factor', type=int, help='Amount to sharpen (0, 1, 2, ...)')
    parser.add_argument('--analysis_area_percent', type=float, default=100, help='Percentage of the image area to analyze (center crop).')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode to output histograms.')

    args = parser.parse_args()
    process_images(args)

if __name__ == '__main__':
    main()

