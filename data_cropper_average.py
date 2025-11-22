from PIL import Image, ImageDraw
import numpy as np
import os, shutil, argparse


from scipy.ndimage import gaussian_filter

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageChops
from scipy.ndimage import gaussian_filter


dir_low = 'data/CFRP_60_low/'
dir_high = 'data/CFRP_60_high/'
dir_result = 'data/enhanced/'
in_size = (40, 120)
out_size = (80, 240)


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
        grayscale_image = original_image.convert('L') # 'L' converts to 8-bit grayscale
    else:
        grayscale_image = original_image

    image_array = np.array(grayscale_image)
    
    # Safety check to avoid errors on empty/black images
    if image_array.size == 0 or image_array.max() == image_array.min():
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


def flip(image: Image.Image, horizontal: bool = False, vertical: bool = False) -> Image.Image:
    if horizontal:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if vertical:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
    return image



def data_generator(average=False):
    flip_possible = [(False, False), (True, False), (False, True), (True, True)]

    # Create output directory / reset it 
    if not os.path.exists(dir_result):
        os.makedirs(dir_result)
    else : 
        shutil.rmtree(dir_result)
        os.makedirs(dir_result)
    
    os.makedirs(os.path.join(dir_result,'low'))
    os.makedirs(os.path.join(dir_result,'high'))

    if not average :
        print('Processing without averaging...')

        # Process low-resolution images
        print("--- Processing low-resolution images ---")
        image_count = 0
        # Note: Le calcul de number_of_images est approximatif ici (dépend du filtre) mais utile pour la barre de progression
        tiff_files = [name for name in os.listdir(dir_low) if name.endswith('.tiff')]
        number_of_images = len(tiff_files) * 4 
        
        for filename in tiff_files:
            inputh_path = os.path.join(dir_low, filename)
            # On garde tiff=True ici pour l'overlay
            cropped_image = crop_relevant_zone(inputh_path, in_size, tiff=True)
            
            if cropped_image is not None:
                for h_flip, v_flip in flip_possible:
                    augmented_image = flip(cropped_image, horizontal=h_flip, vertical=v_flip)
                    output_path = os.path.join(dir_result, 'low', f"{image_count}.png")
                    augmented_image.save(output_path)
                    image_count += 1
                    print(f"Processed and saved image: {image_count}/{number_of_images}", end='\r')
        print("") # Retour à la ligne après la boucle
        
        # Process high-resolution images
        print("--- Processing high-resolution images ---")
        image_count = 0
        png_files = [name for name in os.listdir(dir_high) if name.endswith('.png')]
        number_of_images = len(png_files) * 4
        
        for filename in png_files:
            inputh_path = os.path.join(dir_high, filename)
            cropped_image = crop_relevant_zone(inputh_path, out_size)
            
            if cropped_image is not None:
                for h_flip, v_flip in flip_possible:
                    augmented_image = flip(cropped_image, horizontal=h_flip, vertical=v_flip)
                    output_path = os.path.join(dir_result, 'high', f"{image_count}.png")
                    augmented_image.save(output_path)
                    image_count += 1
                    print(f"Processed and saved image: {image_count}/{number_of_images}", end='\r')
        print("")

    else :
        print('Processing with averaging...')

        # Process low-resolution images
        print("--- Processing low-resolution images ---")
        image_count = 0
        tiff_files = [name for name in os.listdir(dir_low) if name.endswith('.tiff')]
        number_of_images = len(tiff_files) * 4 * (len(tiff_files)-1)
        
        for filename1 in tiff_files:
            for filename2 in tiff_files:
                if filename1 != filename2:
                    inputh_path1 = os.path.join(dir_low, filename1)
                    cropped_image1 = crop_relevant_zone(inputh_path1, in_size, tiff=True)
                    
                    inputh_path2 = os.path.join(dir_low, filename2)
                    cropped_image2 = crop_relevant_zone(inputh_path2, in_size, tiff=True)
                    
                    if cropped_image1 is not None and cropped_image2 is not None:
                        average_image = Image.blend(cropped_image1, cropped_image2, alpha=0.5)

                        for h_flip, v_flip in flip_possible:
                            augmented_image = flip(average_image, horizontal=h_flip, vertical=v_flip)
                            output_path = os.path.join(dir_result, 'low', f"{image_count}.png")
                            augmented_image.save(output_path)
                            image_count += 1
                            print(f"Processed and saved image: {image_count}/{number_of_images}", end='\r')
        print("")

        # Process high-resolution images
        print("--- Processing high-resolution images ---")
        image_count = 0
        png_files = [name for name in os.listdir(dir_high) if name.endswith('.png')]
        # Note: Le calcul number_of_images ici est indicatif
        
        for filename1 in png_files:
            for filename2 in png_files:
                if filename1 != filename2:
                    inputh_path1 = os.path.join(dir_high, filename1)
                    # Correction: in_size -> out_size
                    cropped_image1 = crop_relevant_zone(inputh_path1, out_size)
                    
                    inputh_path2 = os.path.join(dir_high, filename2)
                    # Correction: in_size -> out_size
                    cropped_image2 = crop_relevant_zone(inputh_path2, out_size)
                    
                    if cropped_image1 is not None and cropped_image2 is not None:
                        average_image = Image.blend(cropped_image1, cropped_image2, alpha=0.5)

                        for h_flip, v_flip in flip_possible:
                            augmented_image = flip(average_image, horizontal=h_flip, vertical=v_flip)
                            output_path = os.path.join(dir_result, 'high', f"{image_count}.png")
                            augmented_image.save(output_path)
                            image_count += 1
                            print(f"Processed and saved image: {image_count}", end='\r')
        print("")



def main() -> None:
    # Read parameters from configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("--average",
                        type=str,
                        default=False,
                        required=False,
                        help="Path to test config file.")
    args = parser.parse_args()

    data_generator(average=args.average)

if __name__ == "__main__":
    main()
