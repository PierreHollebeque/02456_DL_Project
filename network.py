import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.hub
import numpy as np

# --- Custom Modules ---

class ResidualBlock(nn.Module):
    """SRGAN-style Residual Block: Conv -> BN -> PReLU -> Conv -> BN -> Skip"""
    def __init__(self, channels=64):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        # Skip connection
        return x + self.block(x) 

class UpsampleBlock(nn.Module):
    """Upsampling Block using PixelShuffle (as used in conv3, conv4)"""
    def __init__(self, in_channels, up_scale_factor):
        super(UpsampleBlock, self).__init__()
        # Output channels = Input channels * r^2. For r=2, 64 * 4 = 256
        channels_for_shuffle = in_channels * (up_scale_factor ** 2)
        
        self.conv = nn.Conv2d(in_channels, channels_for_shuffle, 
                              kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale_factor)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x

# --- Generator Class ---

class Generator(nn.Module):
    """PyTorch implementation of the SRGAN Generator."""
    def __init__(self, name, B, in_channels=3):
        super(Generator, self).__init__()
        self.name = name
        
        # Initial Convolution (conv1)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=9, padding=4)
        self.prelu1 = nn.PReLU()
        
        # Residual Blocks
        res_blocks = [ResidualBlock(64) for _ in range(B)]
        self.res_blocks = nn.Sequential(*res_blocks)
        
        # Mid-network (conv2 + BN + skip addition)
        self.mid_conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.mid_bn = nn.BatchNorm2d(64)
        
        # Upsampling Blocks (x4 total upscaling: 2*2)
        self.upsample1 = UpsampleBlock(64, up_scale_factor=2) 
        self.upsample2 = UpsampleBlock(64, up_scale_factor=2) 

        # Final Convolution (conv5)
        self.conv_final = nn.Conv2d(64, in_channels, kernel_size=9, padding=4)

    def forward(self, inputs, train_phase=True):
        out = self.prelu1(self.conv1(inputs))
        skip_connection = out
        
        out = self.res_blocks(out)
        
        # Skip connection addition
        out = self.mid_bn(self.mid_conv(out))
        out = torch.add(out, skip_connection)
        
        # Upsampling
        out = self.upsample1(out) 
        out = self.upsample2(out) 
        
        out = self.conv_final(out)
        
        # Output normalized to [-1, 1]
        return torch.tanh(out)


# --- Discriminator Class ---

class Discriminator(nn.Module):
    """PyTorch implementation of the SRGAN Discriminator."""
    def __init__(self, name, in_channels=3):
        super(Discriminator, self).__init__()
        self.name = name
        
        # Discriminator blocks (downsampling via stride=2)
        self.net = nn.Sequential(
            # conv1_1: Conv(64, 3, 2) -> Leaky(0.2)
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # conv2_1: Conv(128, 3, 2) -> BN1 -> Leaky(0.2)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # conv3_1: Conv(256, 3, 2) -> BN2 -> Leaky(0.2)
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # conv4_1: Conv(512, 3, 2) -> BN3 -> Leaky(0.2)
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Fully Connected Layers (Final classification)
        self.fc_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # Global average pooling
            nn.Flatten(),
            nn.Linear(512, 1024), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1) # Final output logit
        )

    def forward(self, inputs, train_phase=True):
        out = self.net(inputs)
        output = self.fc_layers(out)
        return output

# --- VGG Feature Extractor ---

class VGGFeatureExtractor(nn.Module):
    """VGG19 feature extractor for Perceptual Loss (extracts from relu2_2)."""
    def __init__(self, vgg_path):
        super(VGGFeatureExtractor, self).__init__()
        
        # Load pre-trained VGG19
        try:
            vgg19 = torch.hub.load('pytorch/vision', 'vgg19', weights='VGG19_Weights.DEFAULT')
        except:
             # Fallback for environments without internet access
            from torchvision.models import vgg19
            vgg19 = vgg19(weights='IMAGENET1K_V1')

        # Extract features up to relu2_2 (index 8 in the feature sequence)
        # Sequence: Conv(0), ReLU(1), Conv(2), ReLU(3), MaxPool(4), Conv(5), ReLU(6), Conv(7), ReLU(8)
        self.features = nn.Sequential(*list(vgg19.features)[:9]) 
        
        # Disable gradient calculation
        for param in self.features.parameters():
            param.requires_grad = False
            
        # ImageNet BGR means (converted to RGB order for PyTorch input subtraction)
        self.mean = torch.tensor([123.68, 116.779, 103.939]).view(1, 3, 1, 1) 
        
    def forward(self, inputs):
        # 1. De-normalize from [-1, 1] to [0, 255]
        inputs = (inputs + 1.0) * 127.5 
        
        # 2. VGG Pre-processing: RGB -> BGR (flip(1)) and Mean Subtraction
        inputs = inputs.flip(1).sub(self.mean.to(inputs.device))
        
        # 3. Feature Extraction
        features = self.features(inputs)
        return features

# Alias to match the original TF naming convention
vggnet = VGGFeatureExtractor