import argparse
from train_test import train, test, up_scale
from PIL import Image
import numpy as np
import os
import cv2 

# Utility for image resizing (used in test mode)
def imresize_np(img_array, size):
    # img_array is [H, W, C] and [0, 255]. cv2.resize takes (W, H).
    resized_img = cv2.resize(img_array, (size[1], size[0]), interpolation=cv2.INTER_CUBIC)
    return resized_img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch SRGAN Training and Testing Script")

    # Training and Path Arguments
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (Paper: 16).")
    parser.add_argument("--lambd", type=float, default=1e-3, help="Weight for adversarial loss (lambda).")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate.")
    parser.add_argument("--clip_v", type=float, default=0.05, help="Discriminator weight clipping value (WGAN).")
    parser.add_argument("--B", type=int, default=5, help="Number of residual blocks (Paper: 16).")
    parser.add_argument("--max_itr", type=int, default=100000, help="Maximum number of iterations (Paper: 600000).")
    
    # NOTE: path_trainset should point to the directory containing 'LR/' and 'HR/' subfolders
    parser.add_argument("--path_trainset", type=str, default="./data/", help="Path to the custom dataset directory (must contain LR/ and HR/ subfolders).")
    parser.add_argument("--path_vgg", type=str, default="./vgg_para/", help="Path to VGG parameters.")
    parser.add_argument("--path_save_model", type=str, default="./save_para/", help="Path to save model checkpoints.")

    # Mode Argument
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], 
                        help="Execution mode: 'train' or 'test'.")
    
    # Parse initial arguments
    args = parser.parse_args()

    # --- Execution Logic ---
    if args.mode == "test":
        
        # Add test image path argument only in 'test' mode
        parser.add_argument("--path_test_img", type=str, default="./test/0.jpg", 
                            help="Path to the HR (High-Resolution) test image.")
        args = parser.parse_args() # Re-parse to include the new argument
        
        if not os.path.exists(args.path_test_img):
             print(f"❌ Error: Test file not found at {args.path_test_img}.")
             print("Please provide a valid path to an HR image.")
        else:
            print(f"▶️ Test Mode: Loading image {args.path_test_img}")
            
            # Load HR image
            img = np.array(Image.open(args.path_test_img).convert('RGB')) 
            
            # Calculate LR dimensions (down sample factor: 4)
            h, w = img.shape[0] // 4, img.shape[1] // 4 
            
            # Downsample HR image to LR using Bicubic for INFERENCE INPUT
            downsampled_img = imresize_np(img, [h, w])
            
            # Call the test function
            test(downsampled_img, img, args.B)
            
    else: # args.mode == "train"
        print("▶️ Training Mode: Starting SRGAN process.")
        print(f"Dataset Path: {args.path_trainset}")
        print(f"Parameters: Batch={args.batch_size}, B-Blocks={args.B}, Max Iters={args.max_itr}")
        
        # Call the train function
        train(
            batch_size=args.batch_size, 
            lambd=args.lambd, 
            init_lr=args.learning_rate, 
            clip_v=args.clip_v, 
            B=args.B,
            max_itr=args.max_itr, 
            path_trainset=args.path_trainset, 
            path_vgg=args.path_vgg,
            path_save_model=args.path_save_model
        )