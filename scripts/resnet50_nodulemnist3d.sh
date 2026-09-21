#!/bin/bash
#SBATCH -p gaudi
#SBATCH -q public
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -t 3-00:00:00
#SBATCH -c 8
#SBATCH --mem=96G

APPTAINER_CMD=(../gaudi-apptainer.sh exec)

export PT_HPU_LAZY_MODE=1

"${APPTAINER_CMD[@]}" python /workspace/MedMNIST3D/engine.py --data_flag nodulemnist3d --config /workspace/config.yaml --as_rgb --size 28 --batch_size 32 --model_flag resnet50 --conv Conv3d --num_epochs 100 --device hpu
