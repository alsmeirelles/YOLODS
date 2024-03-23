#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH -t 35:00:00
#SBATCH --gpus=v100-32:2
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

time python train_yolov8.py -ep 100 -bs 96 -data merged5.yaml -model yolov8l.pt -gpus 2

echo '[FINAL] done training'

deactivate 

date +"%D %T"


