import numpy as np
import scipy.misc as misc
from PIL import Image
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import kagglehub


def read_crop_data(path, batch_size, shape, factor):
    h = shape[0]
    w = shape[1]
    c = shape[2]
    filenames = os.listdir(path)
    rand_selects = np.random.randint(0, filenames.__len__(), [batch_size])
    batch = np.zeros([batch_size, h, w, c])
    downsampled = np.zeros([batch_size, h//factor, w//factor, c])
    for idx, select in enumerate(rand_selects):
        try:
            img = np.array(Image.open(path + filenames[select]))[:, :, :3]
            crop = random_crop(img, h)
            batch[idx, :, :, :] = crop
            downsampled[idx, :, :, :] = misc.imresize(crop, [h // factor, w // factor])
        except:
            img = np.array(Image.open(path + filenames[0]))[:, :, :3]
            crop = random_crop(img, h)
            batch[idx, :, :, :] = crop
            downsampled[idx, :, :, :] = misc.imresize(crop, [h//factor, w//factor])
    return batch, downsampled

def random_crop(img, size):
    h = img.shape[0]
    w = img.shape[1]
    start_x = np.random.randint(0, h - size + 1)
    start_y = np.random.randint(0, w - size + 1)
    return img[start_x:start_x + size, start_y:start_y + size, :]




def load_cars_dataset(batch_size=64, seed=42):
    """
    Loads Cars dataset from Kaggle and splits into train/test.
    Args:
        batch_size (int): batch size for DataLoader
        data_dir (str): path to the root of the dataset directory
        seed (int|None): optional seed for reproducible sampling
    """
    try:
        data_dir = kagglehub.dataset_download("kshitij192/cars-image-dataset")
        if seed is not None:
            torch.manual_seed(seed)

        # Define transformations
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to fixed size
            transforms.ToTensor(),
        ])

        train_dir = os.path.join(data_dir,'Cars Dataset', 'train')
        test_dir = os.path.join(data_dir,'Cars Dataset', 'test')

        train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
        test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        print(f"Using {len(train_loader.dataset)} training and {len(test_loader.dataset)} test images")
        return train_loader, test_loader, data_dir
    except Exception as e:
        print(f"Error loading Cars dataset: {e}")
        return None, None

if __name__ == "__main__":
    train_loader, test_loader = load_cars_dataset(batch_size=32)
    if train_loader and test_loader:
        # Check one batch
        images, labels = next(iter(train_loader))
        print(f"Batch image tensor shape: {images.shape}")
        print(f"Batch labels tensor shape: {labels.shape}")