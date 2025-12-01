#!/bin/sh
### General options
### –- specify queue --
BSUB -q gpua100
### -- set the job Name --
#BSUB -J testjob
### -- ask for number of cores (default: 1) --
BSUB -n 2
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
BSUB -W 20:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=5GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
BSUB -u remiberthelot49@gmail.com
### -- send notification at start --
BSUB -B
### -- send notification at completion--
BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
BSUB -o gpu_%J.out
BSUB -e gpu_%J.err
# -- end of LSF options --
REPO=/zhome/01/9/224462/DL/02456_DL_Project

# # Create job_out if it is not present
# if [[ ! -d ${REPO}/job_out ]]; then
# 	mkdir ${REPO}/job_out
# fi


# date=$(date +%Y%m%d_%H%M)
# mkdir ${REPO}/runs/train/${date}

nvidia-smi
# Activate venv
module purge
module load python3/3.9.23  
module load cuda/12.8.1
source ${REPO}/.venv/bin/activate


# Lancer l'entraînement
python ${REPO}/train.py \
  --dataroot ./datasets/mock_images \
  --name mock_model_resnet \
  --model cycle_gan \
  --init_type xavier \
  --netG resnet_6blocks \
  --n_epochs 6 \
  --n_epochs_decay 18 \
  --batch_size 8 \
  --lambda_B 5 \
  --pool_size 80 \
  --preprocess none

# Lancer le test
python ${REPO}/test.py \
  --dataroot ./datasets/mock_images/testA \
  --name mock_model_resnet \
  --model_suffix _A \
  --no_dropout