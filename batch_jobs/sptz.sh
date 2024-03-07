#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH -t 35:00:00
#SBATCH --gpus=v100-32:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alsmeirelles@gmail.com

#echo commands to stdout
#set -x

cd /ocean/projects/asc130006p/alsm/yolo

echo '[VIRTUALENV]'
source /ocean/projects/asc130006p/alsm/yolo/env/bin/activate

#Load CUDA and set LD_LIBRARY_PATH
module load cuda/11.7.1

echo '[START] training'
date +"%D %T"

#time yolo task=detect mode=train model=yolov8l.pt data=sptz.yaml epochs=100 imgsz=1280 batch=4 plots=True save=True save_period=1 
time python train9_yolol_ger8.py

echo '[FINAL] done training'

deactivate 

date +"%D %T"


