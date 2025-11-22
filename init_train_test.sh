python  util/data_cropper_average.py --average True
python train.py --dataroot ./datasets/ifr_images --name ifr_model_resnet_square --model cycle_gan  --init_type xavier --netG resnet_6blocks --n_epochs 3 --n_epochs_decay 6 --batch_size 8 --lambda_B 5 --pool_size 80 --preprocess none
python test.py  --dataroot ./datasets/ifr_images --name ifr_model_resnet 
python test.py --dataroot ./datasets/ifr_images/testA --name ifr_model_resnet --model_suffix _A --no_dropout
# python test.py --dataroot ./datasets/ifr_images/testB --name ifr_model_unet --model_suffix _B --no_dropout