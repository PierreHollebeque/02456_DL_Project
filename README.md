# Project 23 : FEATURE ALIGNMENT FOR UNPAIRED INFRARED IMAGE TRANSLATION IN THE MICROSTRUCTURES OF COMPOSITE MATERIALS
## 02456 DEEP LEARNING, DTU COMPUTE, FALL 2025 - Danish Technical University (DTU)
## Authors : GROUP 130
  - Rémi Berthelot (s254144)
  - Andreas Løvendahl Eefsen (s224223)
  - Pierre Hollebèque (s254136)
  - Maxime Roux (s244314)

## The project
This GitHub project is associated with a project report that contains an explanation of the issue addressed and the methodology implemented.

## Notebook
The notebook containing all the work done is called `main.ipynb`.

## Strucure of the code
The part of the code on CyleGANs is based on another GitHub project, [CycleGAN and pix2pix in PyTorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). This project was taken and adapted to suit our needs.

## How to use the code ?
### Step 1 - Upload the dataset
The first step is to upload the dataset. To do that, you need to place the `CFRP_60_high` and `CFRP_60_low` folders in one named `data_original` and upload it in the `datasets` folder.


```text
datasets/
└── data_original/
    ├── CFRP_60_high/
        └── ...high resolution images
    └── CFRP_60_low/
        └── ...low resolution images
```

### Step 2 - Preprocessing
Use the following bash command : 
 ```text
 python  util/data_cropper_average.py --average True --force_create True --flip true
 ```

- `average` : Create the average datasets N*(N-1) elements
- `flip` : Flip vertically and horizontally images (multiply number of images by 4)
- `force_create` : Overwrite the existing dataset folder if it exists

### Step 3 - Training
```text
python train.py --dataroot ./datasets/ifr_images --name ifr_model_resnet_final --model cycle_gan  --init_type xavier --netG resnet_6blocks --n_epochs 3 --n_epochs_decay 6 --batch_size 8 --lambda_B 5 --pool_size 80 --preprocess none
```

- `name` : name of the experiment. It decides where to store samples and models. Here, we use fir_images_resnet_suqre
- `model` :  which model to use. Here we use `cycle_gan`
- `init_type` : network initialization [normal | xavier | kaiming | orthogonal]
- `netG` : specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]. Here we use resnet_6blocks
- `n_epochs` : number of epochs with the initial learning rate
- `n_epochs_decay` : number of epochs to linearly decay learning rate to zero
- `batch_size` : input batch size
- `pool_size` : the size of image buffer that stores previously generated images
- `preprocess` : here we use `none`because we use our own methods.

(more details in `options\`)

You can find the loss_log (useful to plot training curves) and model performance over epochs in the folder `checkpoints/ifr_model_resnet_final`.

```text
checkpoints/
└── ifr_model_resnet_final/
    ├── log_loss.txt
    └── web/
        ├── images/
        └── index.html
```
`index.html` allow you to navigate through images.

### Step 4 - Testing

```text
python test.py  --dataroot ./datasets/ifr_images --name ifr_model_resnet_final --netG resnet_6blocks --no_dropout --preprocess none
```

(more details in `options\`)

You can find the results of your tests (the ouput images) in the folder `results/ifr_model_resnet_final`

```text
results/
└── ifr_model_resnet_final/
    └── test_latest/
        ├── images/
        └── index.html
```
`index.html` allow you to navigate through images.


### Step 5 - Plot loss

To plot the loss of the CyclaGAN training, use the following bash command:

```text
python util/plot_loss_results.py --name ifr_model_resnet_final
```

`cyclegan_losses_4_levels.png` is saved in 

```text
results/
└── ifr_model_resnet_final/
    ├── cyclegan_losses_4_levels.png
    └── test_latest/
        ├── images/
        └── index.html
```

