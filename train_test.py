from network import Generator, Discriminator, vggnet

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import time
import os
import glob
import warnings
import cv2 

# --- Configuration and Utilities ---

# Suppress metric warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Utility for image resizing (replacement for scipy.misc.imresize)
def imresize_np(img_array, size):
    # img_array is [H, W, C] and [0, 255]. cv2.resize takes (W, H).
    resized_img = cv2.resize(img_array, (size[1], size[0]), interpolation=cv2.INTER_CUBIC)
    return resized_img

# Data Reader (UPDATED to read LR/HR pairs from subfolders)
class SRDataset(Dataset):
    """Reads LR/HR image pairs from 'LR/' and 'HR/' subdirectories."""
    def __init__(self, path_trainset): 
        
        lr_dir = os.path.join(path_trainset, 'LR')
        hr_dir = os.path.join(path_trainset, 'HR')

        # Find all image paths in the LR folder
        # Sort paths to ensure LR and HR are matched correctly by filename
        self.lr_file_paths = sorted(glob.glob(os.path.join(lr_dir, '*.*')))
        
        if not os.path.isdir(hr_dir):
            raise FileNotFoundError(f"HR directory not found at {hr_dir}. Check your data structure.")
            
        # Construct corresponding HR file paths
        self.hr_file_paths = [os.path.join(hr_dir, os.path.basename(p)) for p in self.lr_file_paths]

        if not self.lr_file_paths:
            print(f"Warning: No LR images found in {lr_dir}. Training may fail.")
        
        # Quick check for missing HR files
        for hr_path in self.hr_file_paths:
            if not os.path.exists(hr_path):
                raise FileNotFoundError(f"Corresponding HR image {os.path.basename(hr_path)} is missing in {hr_dir}.")

    def __len__(self):
        return len(self.lr_file_paths)
        
    def __getitem__(self, idx):
        lr_path = self.lr_file_paths[idx]
        hr_path = self.hr_file_paths[idx]
        
        # 1. Read images
        lr_img = Image.open(lr_path).convert('RGB')
        hr_img = Image.open(hr_path).convert('RGB')
        
        # 2. Preprocessing: [H, W, C] -> [C, H, W], Normalize to [-1, 1]
        hr_tensor = torch.from_numpy(np.array(hr_img).astype(np.float32)).permute(2, 0, 1) / 127.5 - 1.0
        lr_tensor = torch.from_numpy(np.array(lr_img).astype(np.float32)).permute(2, 0, 1) / 127.5 - 1.0
        
        return hr_tensor, lr_tensor

# --- Inference Functions ---

def up_scale(downsampled_img, model_path="./save_para/G_model.pth"):
    """
    Upscales a single image using the Generator.
    downsampled_img: numpy array [H, W, C] (0-255)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = Generator(name="generator", B=16).to(device) 

    try:
        G.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Generator loaded from {model_path}")
    except Exception as e:
        print(f"Warning: Could not load model ({e}).")

    G.eval() 

    # Preprocessing
    input_tensor = torch.from_numpy(downsampled_img.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)
    input_tensor = input_tensor / 127.5 - 1.0
    
    with torch.no_grad():
        sr_tensor = G(input_tensor)

    # Post-processing
    sr_img_np = (sr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5
    sr_img_np = np.clip(sr_img_np, 0, 255).astype(np.uint8)

    Image.fromarray(sr_img_np).show(title="Super-Resolved")
    Image.fromarray(np.uint8(downsampled_img)).show(title="Low-Resolution")


def test(downsampled_img, hr_img, B, model_path="./save_para/G_model.pth"):
    """
    Tests the Generator and compares its PSNR/SSIM to Bicubic upsampling.
    downsampled_img: The actual LR image fed to the network.
    hr_img: The original HR image (for metric calculation).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = Generator(name="generator", B=B).to(device)

    try:
        G.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Warning: Could not load model ({e}).")
        
    G.eval()
    
    # Preprocessing
    lr_tensor = torch.from_numpy(downsampled_img.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)
    lr_tensor = lr_tensor / 127.5 - 1.0

    with torch.no_grad():
        sr_tensor = G(lr_tensor)

    # Post-processing for metric calculation
    sr_img_gen = (sr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5
    
    h, w, _ = hr_img.shape
    
    # Bicubic upsampling of the LR input
    bic_img = imresize_np(downsampled_img, (h, w))
    
    # Resize SR output to match HR image size for metric calculation
    sr_img_resized = imresize_np(np.clip(sr_img_gen, 0, 255), (h, w))

    # Calculate metrics
    p_sr = psnr(hr_img, sr_img_resized)
    s_sr = ssim(hr_img, sr_img_resized, channel_axis=-1, data_range=255)
    p_bic = psnr(hr_img, bic_img)
    s_bic = ssim(hr_img, bic_img, channel_axis=-1, data_range=255)

    # --- Save output images instead of showing them ---
    output_dir = "result_bin"
    os.makedirs(output_dir, exist_ok=True)
    # Convert numpy arrays to PIL Images and save
    # Image.fromarray(np.uint8(sr_img_resized)).show(title="SR Image")
    # Image.fromarray(np.uint8(downsampled_img)).show(title="LR Image")
    # Image.fromarray(np.uint8(bic_img)).show(title="Bicubic")
    Image.fromarray(np.uint8(sr_img_resized)).save(os.path.join(output_dir, "sr_image.png"))
    Image.fromarray(np.uint8(downsampled_img)).save(os.path.join(output_dir, "lr_image.png"))
    Image.fromarray(np.uint8(bic_img)).save(os.path.join(output_dir, "bicubic_image.png"))
    Image.fromarray(np.uint8(hr_img)).save(os.path.join(output_dir, "hr_image.png"))
    
    print(f"SR PSNR: {p_sr:.4f}, SR SSIM: {s_sr:.4f}, BIC PSNR: {p_bic:.4f}, BIC SSIM: {s_bic:.4f}")
    print(f"✅ Images saved in '{output_dir}/' directory.")


# --- Training Function ---

def train(batch_size=4, lambd=1e-3, init_lr=1e-4, clip_v=0.05, B=16, max_itr=100000, 
          path_trainset="./data/", path_vgg="./vgg_para/", path_save_model="./save_para/"):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create models
    G = Generator(name="generator", B=B).to(device)
    D = Discriminator(name="discriminator").to(device)
    Phi = vggnet(path_vgg).to(device) # VGG Net
    
    # Optimizers (RMSProp)
    opt_D = optim.RMSprop(D.parameters(), lr=init_lr)
    opt_G = optim.RMSprop(G.parameters(), lr=init_lr)
    
    # Data Loading (Uses the updated SRDataset)
    dataset = SRDataset(path_trainset) 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Perceptual Loss (tf.nn.l2_loss / batch_size -> 0.5 * MSE)
    vgg_loss_fn = nn.MSELoss() 
    
    os.makedirs("./results", exist_ok=True)
    os.makedirs(path_save_model, exist_ok=True)

    # Training loop
    itr = 0
    lr0 = init_lr
    data_iterator = iter(dataloader)
    
    while itr < max_itr:
        # Learning rate update schedule
        if itr == max_itr // 2 or itr == max_itr * 3 // 4:
            lr0 /= 10.0
            for param_group in opt_D.param_groups:
                param_group['lr'] = lr0
            for param_group in opt_G.param_groups:
                param_group['lr'] = lr0
                
        s0 = time.time()
        
        # 1. Read Data 
        try:
            hr_batch, lr_batch = next(data_iterator)
        except StopIteration:
            data_iterator = iter(dataloader)
            hr_batch, lr_batch = next(data_iterator)
            
        hr_batch = hr_batch.to(device)
        lr_batch = lr_batch.to(device)
        e0 = time.time()
        
        s1 = time.time()

        # --- Discriminator Update (WGAN Loss) ---
        D.train()
        G.eval()
        D.zero_grad()

        # Real and Fake Logits
        real_logits = D(hr_batch)
        sr_batch_detached = G(lr_batch).detach()
        fake_logits = D(sr_batch_detached)

        # D_loss = mean(fake) - mean(real)
        d_loss = torch.mean(fake_logits) - torch.mean(real_logits)
        
        d_loss.backward()
        opt_D.step()

        # Weight Clipping (WGAN)
        for p in D.parameters():
            p.data.clamp_(-clip_v, clip_v)

        # --- Generator Update (Perceptual + Adversarial) ---
        G.train()
        D.eval()
        G.zero_grad()

        sr_batch = G(lr_batch)

        # Adversarial Loss: -mean(fake_logits)
        g_adv_loss = -torch.mean(D(sr_batch))

        # Perceptual Loss (VGG Loss)
        phi_sr = Phi(sr_batch)
        phi_gt = Phi(hr_batch).detach()
        
        # g_content_loss = 0.5 * MSE
        g_content_loss = vgg_loss_fn(phi_sr, phi_gt) * 0.5

        # G_loss = -mean(fake) * lambd + VGG_Loss
        g_loss = g_adv_loss * lambd + g_content_loss
        
        g_loss.backward()
        opt_G.step()
        e1 = time.time()
        
        # --- Logging and Saving ---
        if itr % 200 == 0:
            G.eval()
            with torch.no_grad():
                # Post-processing for visualization and metrics (using first batch element)
                raw = np.uint8((hr_batch[0].permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5)
                lr_log = np.uint8((lr_batch[0].permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5)
                gen = np.uint8((sr_batch[0].permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5)
                
                # Bicubic resizing (needs target size, assumed to be same as HR)
                bicub = imresize_np(lr_log, raw.shape[:2])

                # Metrics
                p_val = psnr(raw, gen)
                s_val = ssim(raw, gen, channel_axis=-1, data_range=255)

                print("Iteration: %d, D_loss: %f, G_loss: %e, PSNR: %f, SSIM: %f, Read_time: %f, Update_time: %f" % 
                      (itr, d_loss.item(), g_loss.item(), p_val, s_val, e0 - s0, e1 - s1))
                
                # Save combined image
                combined_img = np.concatenate((raw, bicub, gen), axis=1)
                Image.fromarray(combined_img).save(f"./results/{itr}.jpg")
                
            G.train() 

        if itr % 5000 == 0:
            # Save weights
            torch.save(G.state_dict(), os.path.join(path_save_model, "G_model.pth"))
            torch.save(D.state_dict(), os.path.join(path_save_model, "D_model.pth"))
            print(f"Models saved at iteration {itr}")
            
        itr += 1