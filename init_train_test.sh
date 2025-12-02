python  util/data_cropper_average.py --average True --flip False --force_create True
python train.py --dataroot ./datasets/ifr_images --name ifr_model_resnet_final --model cycle_gan  --init_type xavier --netG resnet_6blocks --n_epochs 60 --n_epochs_decay 240 --batch_size 8 --lambda_B 5 --pool_size 80 --preprocess none
python test.py --dataroot ./datasets/ifr_images/trainA --name ifr_model_resnet_final --model_suffix _A --no_dropout --netG resnet_6blocks




python test.py --dataroot ./datasets/ifr_images/testA --name ifr_model_resnet --model_suffix _A --no_dropout --netG resnet_6blocks
# python test.py --dataroot ./datasets/ifr_images/testB --name ifr_model_unet --model_suffix _B --no_dropout
python train.py --dataroot ./datasets/mock_images --name mock_model_resnet_1 --model cycle_gan  --init_type xavier --netG resnet_6blocks --n_epochs 8 --n_epochs_decay 16 --batch_size 8 --lambda_B 5 --pool_size 80 --preprocess none
python test.py --dataroot ./datasets/mock_images/testA --name mock_model_resnet_1 --model_suffix _A --no_dropout --netG resnet_6blocks
python test.py --dataroot ./datasets/mock_images/testA --name ifr_model_resnet --model_suffix _A --no_dropout --netG resnet_6blocks
