#!/bin/bash
#SBATCH -p gaudi
#SBATCH -q public
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -t 3-00:00:00
#SBATCH -c 8
#SBATCH --mem=48G

APPTAINER_CMD=(../gaudi-apptainer.sh exec)

export PT_HPU_LAZY_MODE=0

"${APPTAINER_CMD[@]}" python /workspace/MedMNIST/engine.py --data_flag chestmnist \
--config /workspace/config.yaml --device hpu \
--as_rgb --model_flag resnet50 --num_epochs 100 --batch_size 64