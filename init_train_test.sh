python  util/data_cropper.py --input_dir ./datasets/ifr_images/original --output_dir ./datasets/ifr_images
python train.py --dataroot ./datasets/ifr_images --name ifr_model --model cycle_gan --n_epochs 10 --n_epochs_decay 10
python test.py  --dataroot ./datasets/ifr_images --name ifr_model 
