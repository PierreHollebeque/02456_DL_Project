from PIL import Image, ImageChops
import numpy as np
import os, shutil, argparse
import random

def create_list(N):
    n_true = int(N * 0.8)
    n_false = N - n_true
    lst = [True] * n_true + [False] * n_false
    random.shuffle(lst)
    return lst

from scipy.ndimage import gaussian_filter


dir_low = 'datasets/data_original/CFRP_60_low/'
dir_high = 'datasets/data_original/CFRP_60_high/'
dir_result = 'datasets/ifr_images'
in_size = (40, 128)
out_size = (80, 256)


def crop_relevant_zone(image_path: str, crop_size: tuple[int, int], tiff=False):
    try:
        raw_image = Image.open(image_path)
        # Apply specific processing if tiff is True
        if tiff:
            try:
                raw_image.seek(0)
                p1 = raw_image.copy().convert("RGB")
                
                raw_image.seek(1)
                p2 = raw_image.copy().convert("RGB")
                
                # The image to work on becomes the product of the two frames
                original_image = ImageChops.multiply(p1, p2)
            except EOFError:
                print(f"Error: The TIFF file '{image_path}' does not contain enough frames.")
                return None
        else:
            # Otherwise use the image as is
            original_image = raw_image
    except FileNotFoundError:
        print(f"Error: The file '{image_path}' was not found.")
        return None
    except Exception as e:
        print(f"Error opening or reading the image: {e}")
        return None

    if original_image.mode not in ['I', 'F', 'L']:
        grayscale_image = original_image.convert('L') # 'L' converts to 8-bit grayscale, which is fine for finding the location of max brightness
    else:
        grayscale_image = original_image

    image_array = np.array(grayscale_image)
    if image_array.size == 0 or image_array.max() == image_array.min():
        print(f"Error: The image '{image_path}' is empty or has no variation in pixel values.")
        return None
    window_size = 150
    filtered_image = gaussian_filter(image_array.astype(np.float32), sigma= window_size/6)

    max_loc = np.unravel_index(np.argmax(filtered_image), filtered_image.shape)
    center_y, center_x = max_loc

    crop_width, crop_height = crop_size

    left = center_x - crop_width // 2
    upper = center_y - crop_height // 2
    right = left + crop_width
    lower = upper + crop_height

    left = max(0, left)
    upper = max(0, upper)
    right = min(original_image.width, right)
    lower = min(original_image.height, lower)

    crop_box = (left, upper, right, lower)
    cropped_image = original_image.crop(crop_box)
    return cropped_image

def reformate_size(image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    current_size = image.size
    new_image = Image.new("RGB", target_size)
    if target_size[1] != current_size[1]:
        image = image.resize(out_size)
        current_size = image.size
    x_coord = (target_size[0] - current_size[0]) //2
    new_image.paste(image, (x_coord, 0))
    return new_image


def flip(image: Image.Image, horizontal: bool = False, vertical: bool = False) -> Image.Image:
    if horizontal:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if vertical:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
    return image



def data_generator(average=False, force=False):
    flip_possible = [(False, False), (True, False), (False, True), (True, True)]

    # Create output directory / reset it 
    if not os.path.exists(dir_result):
        os.makedirs(dir_result)
    elif not force :
        print(f"Directory {dir_result} already exists. Use force=True to overwrite.")
        return
    else : 
        shutil.rmtree(dir_result)
        os.makedirs(dir_result)
    os.makedirs(os.path.join(dir_result,'testA'))
    os.makedirs(os.path.join(dir_result,'testB'))
    os.makedirs(os.path.join(dir_result,'trainA'))
    os.makedirs(os.path.join(dir_result,'trainB'))

    if not average :
        print('Processing without averaging...')

        # Process low-resolution images
        print("--- Processing low-resolution images ---")
        image_count = 0
        number_of_images = len([name for name in os.listdir(dir_low) if name.endswith('.tiff')]) * 4 
        list_train_test = create_list(number_of_images)
        for filename in os.listdir(dir_low):
            if filename.endswith('.tiff'):
                inputh_path = os.path.join(dir_low, filename)
                cropped_image = crop_relevant_zone(inputh_path, in_size, tiff=True)
                cropped_image = reformate_size(cropped_image, (256,256))
                if cropped_image is not None:
                    for h_flip, v_flip in flip_possible:
                        augmented_image = flip(cropped_image, horizontal=h_flip, vertical=v_flip)
                        if list_train_test[image_count]:
                            output_path = os.path.join(dir_result,'trainA', f"{image_count}.png")
                        else :
                            output_path = os.path.join(dir_result,'testA', f"{image_count}.png")
                        augmented_image.save(output_path)
                        image_count += 1
                        print(f"Processed and saved image: {image_count}/{number_of_images}", end='\r')



        
        # Process high-resolution images
        print("--- Processing high-resolution images ---")
        image_count = 0
        for filename in os.listdir(dir_high):
            if filename.endswith('.png'):
                inputh_path = os.path.join(dir_high, filename)
                cropped_image = crop_relevant_zone(inputh_path, out_size)
                cropped_image = reformate_size(cropped_image, (256,256))
                for h_flip, v_flip in flip_possible:
                    augmented_image = flip(cropped_image, horizontal=h_flip, vertical=v_flip)
                    if list_train_test[image_count]:
                      output_path = os.path.join(dir_result,'trainB', f"{image_count}.png")
                    else :
                      output_path = os.path.join(dir_result,'testB', f"{image_count}.png")
                    augmented_image.save(output_path)
                    image_count += 1
                    print(f"Processed and saved image: {image_count}/{number_of_images}", end='\r')

    else :
        print('Processing with averaging...')

        # Process low-resolution images
        print("--- Processing low-resolution images ---")
        image_count = 0
        number_of_images = len([name for name in os.listdir(dir_low) if name.endswith('.tiff')])*4*(len([name for name in os.listdir(dir_low) if name.endswith('.tiff')])-1)
        list_train_test = create_list(number_of_images)
        for filename1 in os.listdir(dir_low):
            for filename2 in os.listdir(dir_low):
                if filename1.endswith('.tiff') and filename2.endswith('.tiff') and filename1 != filename2:
                    inputh_path = os.path.join(dir_low, filename1)
                    cropped_image1 = crop_relevant_zone(inputh_path, in_size, tiff=True)
                    inputh_path = os.path.join(dir_low, filename2)
                    cropped_image2 = crop_relevant_zone(inputh_path, in_size, tiff=True)
                    if cropped_image1 is not None or cropped_image2 is not None:
                        average_image = Image.blend(cropped_image1, cropped_image2, alpha=0.5)
                        average_image = reformate_size(average_image, (256,256))
                        for h_flip, v_flip in flip_possible:
                            augmented_image = flip(average_image, horizontal=h_flip, vertical=v_flip)
                            if list_train_test[image_count]:
                                output_path = os.path.join(dir_result,'trainA', f"{image_count}.png")
                            else :
                                output_path = os.path.join(dir_result,'testA', f"{image_count}.png")
                            augmented_image.save(output_path)
                            image_count += 1
                            print(f"Processed and saved image: {image_count}/{number_of_images}", end='\r')
        
        # Process high-resolution images
        print("--- Processing high-resolution images ---")
        image_count = 0
        for filename1 in os.listdir(dir_high):
            for filename2 in os.listdir(dir_high):
                if filename1.endswith('.png') and filename2.endswith('.png') and filename1 != filename2:
                    inputh_path = os.path.join(dir_high, filename1)
                    cropped_image1 = crop_relevant_zone(inputh_path, out_size)
                    inputh_path = os.path.join(dir_high, filename2)
                    cropped_image2 = crop_relevant_zone(inputh_path, out_size)
                    average_image = Image.blend(cropped_image1, cropped_image2, alpha=0.5)
                    average_image = reformate_size(average_image, (256,256))
                    for h_flip, v_flip in flip_possible:
                        augmented_image = flip(average_image, horizontal=h_flip, vertical=v_flip)
                        if list_train_test[image_count]:
                          output_path = os.path.join(dir_result,'trainB', f"{image_count}.png")
                        else :
                          output_path = os.path.join(dir_result,'testB', f"{image_count}.png")
                        augmented_image.save(output_path)
                        image_count += 1
                        print(f"Processed and saved image: {image_count}/{number_of_images}", end='\r')



def main() -> None:
    # Read parameters from configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("--average",
                        type=bool,
                        default=False,
                        required=False,
                        help="Create the average datasets N*(N-1) elements")
    parser.add_argument("--force_create",
                        type=bool,
                        default=False,
                        required=False,
                        help="Overwrite the existing dataset folder if it exists")
    args = parser.parse_args()

    data_generator(average=args.average,force = args.force_create)

if __name__ == "__main__":
    print('Starting data cropping and augmentation...')
    main()
    # image = crop_relevant_zone('datasets/data_original/CFRP_60_low/Record_2025-11-11_10-49-48.tiff', in_size, tiff=True)
    # image =reformate_size(image, (256,256))
    # image.save('test_crop.png')
    # image = crop_relevant_zone('datasets/data_original/CFRP_60_high/prst_A_stat_03_5_230149.png', out_size, tiff=False)
    # image =reformate_size(image, (256,256))
    # image.save('test_crop2.png')
    
