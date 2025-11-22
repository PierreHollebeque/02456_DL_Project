python  util/data_cropper_average.py --average True
python train.py --dataroot ./datasets/ifr_images --name ifr_model_unet --model cycle_gan  --init_type xavier --netG unet_256 --netD basic --n_epochs 0 --n_epochs_decay 1 --batch_size 32 --lambda_B 5 --preprocess none
python test.py  --dataroot ./datasets/ifr_images --name ifr_model_unet 
python test.py --dataroot ./datasets/ifr_images/testA --name ifr_model_unet --model_suffix _A --no_dropout
# python test.py --dataroot ./datasets/ifr_images/testB --name ifr_model_unet --model_suffix _B --no_dropout