#!/bin/bash
#SBATCH -t 1-00:00:00
#SBATCH -p public
#SBATCH -q public
#SBATCH --cpus-per-task=10

module load mamba/latest
chmod +x gaudi-apptainer.sh
./gaudi-apptainer.sh build

chmod +x cuda-apptainer.sh
./cuda-apptainer.sh build