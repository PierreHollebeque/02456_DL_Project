import kagglehub
import os
import random
from PIL import Image, ImageFilter, ImageEnhance

# Download latest version



def process_and_split_images(source_path, base_dest_path, train_ratio=0.95):
    train_b_path = os.path.join(base_dest_path, 'trainB')
    test_b_path = os.path.join(base_dest_path, 'testB')
    train_a_path = os.path.join(base_dest_path, 'trainA')
    test_a_path = os.path.join(base_dest_path, 'testA')

    os.makedirs(train_b_path, exist_ok=True)
    os.makedirs(test_b_path, exist_ok=True)
    os.makedirs(train_a_path, exist_ok=True)
    os.makedirs(test_a_path, exist_ok=True)
    
    for root, _, files in os.walk(source_path):
        print(files)
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                original_image_path = os.path.join(root, file)
                
                # Decide whether to put in train or test set
                if random.random() < train_ratio:
                    dest_b_folder = train_b_path
                    dest_a_folder = train_a_path
                else:
                    dest_b_folder = test_b_path
                    dest_a_folder = test_a_path

                # Process image for 'A' folders
                try:
                    img = Image.open(original_image_path).convert("RGB")
                    img = img.resize((256, 256))

                    # Apply blur
                    img_a = img.filter(ImageFilter.GaussianBlur(radius=1))
                    
                    # Increase exposure
                    enhancer = ImageEnhance.Brightness(img_a)
                    img_a = enhancer.enhance(1.3) # Increase brightness by 30%

                    # Low pixel aggregation (downscale and upscale)
                    img_a = img_a.resize((128, 128), Image.BILINEAR).resize((256, 256), Image.BILINEAR)

                    img_a.save(os.path.join(dest_a_folder, file))
                    img.save(os.path.join(dest_b_folder, file))
                except Exception as e:
                    print(f"Error processing {file}: {e}")


if __name__ == '__main__':
    # path = kagglehub.dataset_download("dimensi0n/imagenet-256")
    # print("Path to dataset files:", path)
    source_directory = "../../.cache/kagglehub/datasets/dimensi0n/imagenet-256/versions/1"  # Use the downloaded dataset path
    destination_directory = 'datasets/mock_images'
    process_and_split_images(source_directory, destination_directory)